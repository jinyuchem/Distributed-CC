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

import os
import numpy as np
from mpi4py import MPI
from distr_cc._runtime import vector_delta_norm_sq

_ON_DISK = object()

def _safe_name(name):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(name))

class DIIS:
    def __init__(self, dev=None, space=6, comm=None, scratch=None,
                 scratch_start=0, cleanup=False, mmap=False, prefix=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        self.dev = dev
        self.verbose = getattr(dev, 'verbose', 0)
        self.stdout = getattr(dev, 'stdout', None)
        self.space = int(space)
        if self.space <= 0:
            raise ValueError("DIIS space must be a positive integer")

        self.xs, self.es = [None] * self.space, [None] * self.space
        self._head, self._H, self._xprev = 0, None, None
        self.last_delta_norm_sq = None
        self.comm = comm

        self.scratch = None if scratch is None else os.path.abspath(
            os.path.expandvars(os.path.expanduser(str(scratch))))
        self.scratch_start = max(0, min(int(scratch_start), self.space))
        self.cleanup = bool(cleanup)
        self.mmap = bool(mmap)
        self._scratch_slots = set()

        if self.scratch is not None:
            os.makedirs(self.scratch, exist_ok=True)
            if prefix is None:
                name = getattr(dev, 'name', None) or getattr(dev, '__class__', type(self)).__name__
                prefix = 'DIIS-%s-rank%05d-pid%d' % (_safe_name(name), comm.Get_rank(), os.getpid())
            self.prefix = _safe_name(prefix)
        else:
            self.prefix = None

    def __del__(self):
        if getattr(self, 'cleanup', False):
            try:
                self.clean_scratch()
            except Exception:
                pass

    def reset(self):
        self.clean_scratch()
        self.xs, self.es = [None] * self.space, [None] * self.space
        self._head, self._H, self._xprev = 0, None, None
        self.last_delta_norm_sq = None
        self._scratch_slots = set()
        return self

    def _use_scratch(self, idx):
        return self.scratch is not None and idx >= self.scratch_start

    def _slot_path(self, kind, idx):
        return os.path.join(self.scratch, '%s-%s%02d.npy' % (self.prefix, kind, idx))

    def _save_array(self, kind, idx, arr):
        path = self._slot_path(kind, idx)
        tmp_path = '%s.tmp' % path
        with open(tmp_path, 'wb') as fout:
            np.save(fout, np.asarray(arr))
        os.replace(tmp_path, path)
        self._scratch_slots.add((kind, idx))

    def _load_array(self, kind, idx):
        mmap_mode = 'r' if self.mmap else None
        return np.load(self._slot_path(kind, idx), mmap_mode=mmap_mode, allow_pickle=False)

    def _slot_array(self, slots, kind, idx):
        arr = slots[idx]
        if arr is _ON_DISK:
            return self._load_array(kind, idx)
        return arr

    def _spill_slot(self, idx):
        if not self._use_scratch(idx):
            return
        self._save_array('X', idx, self.xs[idx])
        self._save_array('E', idx, self.es[idx])
        self.xs[idx], self.es[idx] = _ON_DISK, _ON_DISK

    def update(self, x, xerr=None):
        x_shape = np.shape(x)
        x = np.asarray(x).reshape(-1)

        if self._xprev is None:
            self._xprev = x.copy()
            self.last_delta_norm_sq = None
            return x.reshape(x_shape)

        err = np.asarray(xerr).reshape(-1) if xerr is not None else x - self._xprev
        self.xs[self._head], self.es[self._head] = x, err
        nd = sum(v is not None for v in self.xs)

        if self._H is None:
            self._H = np.zeros((self.space + 1, self.space + 1), err.dtype)
            self._H[0, 1:] = self._H[1:, 0] = 1

        new_row = np.zeros(nd, dtype=err.dtype)
        for i in range(nd):
            ei = err if i == self._head else self._slot_array(self.es, 'E', i)
            new_row[i] = np.dot(err, np.asarray(ei).reshape(-1))
            ei = None

        self.comm.Allreduce(MPI.IN_PLACE, new_row, op=MPI.SUM)

        for i, val in enumerate(new_row):
            self._H[self._head + 1, i + 1] = val
            self._H[i + 1, self._head + 1] = val.conj()

        self._spill_slot(self._head)
        self._head = (self._head + 1) % self.space

        xnew = self.extrapolate(nd)
        delta_norm_sq = np.array(vector_delta_norm_sq(xnew, self._xprev), dtype=np.float64)
        self.comm.Allreduce(MPI.IN_PLACE, delta_norm_sq, op=MPI.SUM)
        self.last_delta_norm_sq = max(float(delta_norm_sq), 0.0)
        self._xprev = xnew.copy()
        return xnew.reshape(x_shape)

    def clean_scratch(self):
        if self.scratch is None:
            return
        for kind, idx in list(self._scratch_slots):
            try:
                os.remove(self._slot_path(kind, idx))
            except FileNotFoundError:
                pass
            self._scratch_slots.discard((kind, idx))

    def extrapolate(self, nd):
        from pyscf.lib import logger

        h = self._H[:nd + 1, :nd + 1]
        g = np.zeros(nd + 1, h.dtype)
        g[0] = 1

        w, v = np.linalg.eigh(h)
        if np.any(abs(w) < 1e-14):
            logger.debug(self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w) > 1e-14
            c = np.dot(v[:, idx] * (1 / w[idx]), np.dot(v[:, idx].T.conj(), g))
        else:
            c = np.linalg.solve(h, g)
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = np.asarray(self._slot_array(self.xs, 'X', i)).reshape(-1)
            if xnew is None:
                xnew = np.zeros((xi.size,), xi.dtype)
            xnew += xi * ci
            xi = None
        return xnew
