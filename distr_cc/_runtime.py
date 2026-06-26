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
import os
import warnings
import numpy as np
from pyscf import lib
from pyscf.cc.rccsdt import format_size
from pyscf.lib import logger

_TRUE_ENV_VALUES = {"1", "true", "t", "yes", "y", "on"}

def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in _TRUE_ENV_VALUES

def python_fallback_allowed(obj=None):
    if obj is not None:
        value = getattr(obj, "allow_python_fallback", None)
        if value is not None:
            return bool(value)
    return _env_flag("DISTR_CC_ALLOW_PYTHON_FALLBACK", False)

def _native_unavailable_message(feature, err=None):
    msg = (
        "Native distr_cc C helper is required for %s, but it is unavailable. "
        "Build the native library with `cmake -S . -B build && cmake --build build`. "
        "To explicitly allow the slower Python fallback for debugging, set "
        "DISTR_CC_ALLOW_PYTHON_FALLBACK=1 or set `allow_python_fallback = True`."
    ) % feature
    if err is not None:
        msg += " Original error: %s" % err
    return msg

def warn_python_fallback(feature, obj=None, err=None):
    if not python_fallback_allowed(obj):
        raise RuntimeError(_native_unavailable_message(feature, err=err))
    msg = "Using slow Python fallback for %s instead of a distr_cc C helper." % feature
    if err is not None:
        msg += " Original native-helper error: %s" % err
    warnings.warn(msg, RuntimeWarning, stacklevel=2)

def require_native_c(has_c_lib, feature, obj=None, err=None):
    if has_c_lib:
        return True
    warn_python_fallback(feature, obj=obj, err=err)
    return False

def warn_non_pytblis_backend(obj, method_name=None):
    backend = getattr(obj, "einsum_backend", None)
    if str(backend).lower() == "pytblis":
        return
    if getattr(obj, "rank", 0) != 0:
        return

    warned_key = "_distr_cc_warned_non_pytblis_backend"
    if getattr(obj, warned_key, False):
        return
    try:
        setattr(obj, warned_key, True)
    except Exception:
        pass

    label = " for %s" % method_name if method_name else ""
    logger.warn(
        obj,
        "einsum_backend%s is %r. For Linux HPC runs, `pytblis` is strongly "
        "recommended and can be critical for performance. Use "
        "`mycc.set_einsum_backend('pytblis')` after installing pytblis.",
        label, backend,
    )

def norm_sq_from_norms(norms):
    return sum(float(norm) * float(norm) for norm in norms)

def vector_delta_norm_sq(vector, reference, block_size=1 << 20):
    """Return ||vector - reference||^2 without materializing reference.

    ``reference`` may be an in-memory ndarray, memmap, or h5py dataset used by
    DIIS for out-of-core storage.
    """
    if reference is None:
        return None

    vector = np.asarray(vector).reshape(-1)
    size = vector.size
    if size == 0:
        return 0.0

    try:
        ref_size = int(np.prod(reference.shape))
    except AttributeError:
        reference = np.asarray(reference).reshape(-1)
        ref_size = reference.size

    if ref_size != size:
        raise ValueError("DIIS vector and reference have different sizes")

    total = 0.0
    for p0 in range(0, size, block_size):
        p1 = min(p0 + block_size, size)
        ref_block = np.asarray(reference[p0:p1]).reshape(-1)
        delta = vector[p0:p1] - ref_block
        total += float(np.vdot(delta, delta).real)
    return total

def resolve_nvir_diis(mycc, nvir):
    nvir_diis = getattr(mycc, "nvir_diis", None)
    if nvir_diis is None:
        return nvir
    nvir_diis = int(nvir_diis)
    if nvir_diis <= 0:
        raise ValueError("nvir_diis must be a positive integer")
    return min(nvir_diis, nvir)

def resolve_diis_scratch(mycc):
    scratch = getattr(mycc, "diis_scratch", None)
    if scratch in (None, ""):
        return None
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(scratch))))

def resolve_diis_incore_space(mycc):
    space = int(mycc.diis_space)
    if resolve_diis_scratch(mycc) is None:
        return space
    scratch_start = int(getattr(mycc, "diis_scratch_start", 0))
    return max(0, min(scratch_start, space))

def _ensure_unique_tamps_map(mycc):
    if getattr(mycc, "unique_tamps_map", None) is None:
        mycc.unique_tamps_map = mycc.build_unique_tamps_map()
    return mycc.unique_tamps_map

def _unique_tamp_index(unique_map, order):
    return (*unique_map[0], *[slice(None)] * order)

def pack_unique_replicated_tamps(mycc, tamps, scale=1.0):
    unique_tamps_map = _ensure_unique_tamps_map(mycc)
    chunks = []
    dtype = None
    for i, t in enumerate(tamps):
        order = i + 1
        idx = _unique_tamp_index(unique_tamps_map[i], order)
        chunk = np.asarray(t[idx]).reshape(-1)
        dtype = chunk.dtype
        if scale != 1.0:
            chunk = chunk * scale
        chunks.append(chunk)
    if chunks:
        return np.concatenate(chunks)
    return np.asarray([], dtype=np.float64 if dtype is None else dtype)

def unpack_unique_replicated_tamps(mycc, vector, template_tamps, scale=1.0):
    from pyscf.cc.rccsdt import restore_t_

    unique_tamps_map = _ensure_unique_tamps_map(mycc)
    vector = np.asarray(vector).reshape(-1)
    tamps = []
    offset = 0
    for i, template in enumerate(template_tamps):
        order = i + 1
        t = np.zeros_like(template)
        idx = _unique_tamp_index(unique_tamps_map[i], order)
        unique_shape = t[idx].shape
        size = int(np.prod(unique_shape))
        block = vector[offset:offset + size].reshape(unique_shape)
        if scale != 1.0:
            block = block / scale
        t[idx] = block
        restore_t_(t, mycc.nocc, order=order, do_tri=False, unique_tamps_map=unique_tamps_map[i])
        tamps.append(t)
        offset += size
    return tamps, offset

def contraction_logger(mycc, flag="log_highest_t_contractions", all_ranks_flag="log_highest_t_contractions_all_ranks"):
    if not getattr(mycc, flag, False):
        return logger.Logger(mycc.stdout, 0)
    all_ranks = getattr(mycc, all_ranks_flag, False)
    verbose = logger.DEBUG1 if (getattr(mycc, "rank", 0) == 0 or all_ranks) else 0
    return logger.Logger(mycc.stdout, verbose)

def contraction_message(mycc, message, flag="log_highest_t_contractions",
                        all_ranks_flag="log_highest_t_contractions_all_ranks"):
    if getattr(mycc, flag, False) and getattr(mycc, all_ranks_flag, False):
        return "Rank %d %s" % (mycc.rank, message)
    return message

def memory_logger(mycc):
    if not getattr(mycc, "log_memory", False):
        return logger.Logger(mycc.stdout, 0)
    all_ranks = getattr(mycc, "log_memory_all_ranks", False)
    if getattr(mycc, "rank", 0) != 0 and not all_ranks:
        return logger.Logger(mycc.stdout, 0)
    return logger.Logger(mycc.stdout, logger.DEBUG)

def log_memory(mycc, log=None, label=None, per_iter=False):
    if not getattr(mycc, "log_memory", False):
        return
    if per_iter and not getattr(mycc, "log_memory_per_iter", False):
        return
    all_ranks = getattr(mycc, "log_memory_all_ranks", False)
    if getattr(mycc, "rank", 0) != 0 and not all_ranks:
        return
    if log is None:
        log = memory_logger(mycc)
    suffix = "" if label is None else " after %s" % label
    memory_bytes = int(lib.current_memory()[0] * 1024**2)
    log.debug("        Rank %d memory used%s: %8s", mycc.rank, suffix, format_size(memory_bytes))

def make_standard_diis(mycc):
    if isinstance(mycc.diis, lib.diis.DIIS):
        return mycc.diis
    if not mycc.diis:
        return None
    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.space = mycc.diis_space
    return adiis

def make_mpi_diis(mycc, log=None):
    if not mycc.diis:
        return None
    from distr_cc.diis import DIIS as MPI_DIIS

    diis_scratch = resolve_diis_scratch(mycc)
    adiis = MPI_DIIS(dev=mycc, space=mycc.diis_space, comm=mycc.comm, scratch=diis_scratch,
                    scratch_start=getattr(mycc, "diis_scratch_start", 0),
                    cleanup=getattr(mycc, "diis_scratch_cleanup", True),
                    mmap=getattr(mycc, "diis_scratch_mmap", False))
    if diis_scratch is not None and mycc.rank == 0 and log is not None:
        log.info("DIIS scratch dir = %s; keeping %d of %d slots in memory",
                 diis_scratch, resolve_diis_incore_space(mycc), mycc.diis_space)
    return adiis
