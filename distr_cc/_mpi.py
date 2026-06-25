# Copyright 2025-2026 The Distributed-CC Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import threading
import time
from mpi4py import MPI

def punctuate_gil(duration: float | None = 0.0001) -> None:
    """Release the GIL briefly so an MPI progress thread can run."""
    if duration is not None and duration > 0:
        time.sleep(duration)

class MPIProgressThread:
    """Background MPI request poller for overlapping nonblocking collectives."""

    def __init__(self, poll_interval: float = 0.001):
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._requests = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._pause_req = False
        self._paused_ack = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_req = False
        if self._thread.is_alive():
            self._thread.join()

    def add_requests(self, reqs) -> None:
        if not reqs:
            return
        with self._lock:
            self._requests.extend(reqs)

    def set_requests(self, reqs) -> None:
        with self._lock:
            self._requests = list(reqs) if reqs else []

    def clear_requests(self) -> None:
        with self._lock:
            self._requests.clear()

    def pause(self) -> None:
        self._paused_ack.clear()
        self._pause_req = True
        self._paused_ack.wait()

    def resume(self) -> None:
        self._pause_req = False

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            if self._pause_req:
                self._paused_ack.set()
                while self._pause_req and not self._stop_event.is_set():
                    time.sleep(self.poll_interval)
                continue

            active_reqs = None
            with self._lock:
                if self._requests:
                    active_reqs = self._requests[:]

            if active_reqs:
                try:
                    MPI.Request.Testany(active_reqs)
                except Exception:
                    pass

            time.sleep(self.poll_interval)

def mpi_progress_thread_supported() -> bool:
    try:
        return MPI.Query_thread() >= MPI.THREAD_MULTIPLE
    except Exception:
        return True

def mpi_thread_level_name(level) -> str:
    names = {MPI.THREAD_SINGLE: "THREAD_SINGLE", MPI.THREAD_FUNNELED: "THREAD_FUNNELED",
            MPI.THREAD_SERIALIZED: "THREAD_SERIALIZED", MPI.THREAD_MULTIPLE: "THREAD_MULTIPLE"}
    return names.get(level, str(level))

def log_mpi_progress_thread_disabled(mycc, reason: str) -> None:
    if getattr(mycc, "_mpi_progress_thread_disabled_logged", False):
        return
    setattr(mycc, "_mpi_progress_thread_disabled_logged", True)
    log_enabled = any(getattr(mycc, flag, False) for flag in ("log_t3_communication", "log_t4_communication"))
    if getattr(mycc, "rank", 0) == 0 and log_enabled:
        from pyscf.lib import logger
        log = logger.Logger(mycc.stdout, logger.INFO)
        log.info("MPI progress thread disabled: %s", reason)

def start_mpi_progress_thread(mycc):
    if mycc.size <= 1 or not getattr(mycc, "use_mpi_progress_thread", False):
        return None
    if not mpi_progress_thread_supported():
        try:
            level = mpi_thread_level_name(MPI.Query_thread())
        except Exception:
            level = "unknown"
        log_mpi_progress_thread_disabled(mycc, "MPI thread level %s is below THREAD_MULTIPLE" % level)
        return None
    progress_thread = MPIProgressThread(poll_interval=getattr(mycc, "mpi_progress_poll_interval", 0.001))
    progress_thread.start()
    return progress_thread

def punctuate_mpi_progress(mycc, progress_thread) -> None:
    if progress_thread is None:
        return
    punctuate_gil(getattr(mycc, "gil_punctuate_duration", 0.0001))
