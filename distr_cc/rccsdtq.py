#!/usr/bin/env python
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
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#         Huanchen Zhai <hczhai.ok@gmail.com>
#

import sys
import numpy as np
from collections import Counter
import functools
from itertools import permutations
import warnings
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import rccsdt
from pyscf.cc.rccsdt import (_einsum, t3_spin_summation_inplace_,
                            update_t1_fock_eris, intermediates_t1t2, compute_r1r2, r1r2_divide_e_,
                            intermediates_t3, _PhysicistsERIs)
from pyscf.cc.rccsdt_highm import (t3_spin_summation, t3_perm_symmetrize_inplace_, purify_tamps_, r1r2_add_t3_,
                                    intermediates_t3_add_t3, compute_r3, r3_divide_e_)
from pyscf import __config__

from mpi4py import MPI

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distr_cc import _rccsdtq_core as rccsdtq
from distr_cc._rccsdtq_core import t4_project_1_minus_p4_p31_inplace_, t4_add_, _IMDS
from distr_cc.rccsdt import _update_procs_mf, RCCSDT
from distr_cc._c_rccsdtq import (r4_local_tri_divide_e_, t4_single_spin_summation_inplace_,
                                t4_spin_summation_quadruple_sym_, t4_transpose_add_,)
from distr_cc._mpi import punctuate_mpi_progress, start_mpi_progress_thread
from distr_cc._runtime import (contraction_logger, contraction_message, log_memory, make_mpi_diis, make_standard_diis,
    memory_logger, norm_sq_from_norms, pack_unique_replicated_tamps, unpack_unique_replicated_tamps,
    vector_delta_norm_sq, resolve_diis_incore_space, resolve_diis_scratch, resolve_nvir_diis, warn_non_pytblis_backend)
from distr_cc.distribute_t4 import DistributedT4IJKL

_MPI_INT_MAX = np.iinfo(np.intc).max
_MPI_ALLREDUCE_MAX_BYTES = int(getattr(
    __config__, 'cc_mpi_rccsdtq_allreduce_max_bytes', 1 << 30))


def configure_t4_runtime_logging(mycc, tamps):
    if tamps is None or len(tamps) < 4:
        return
    t4 = tamps[3]
    if not isinstance(t4, (tuple, list)) or len(t4) < 1:
        return
    dt4 = t4[0]
    enabled = getattr(mycc, 'log_highest_t_communication', False)
    log_dir = getattr(mycc, 'communication_log_dir', 'comm_logs')
    if hasattr(dt4, 'configure_communication_logging'):
        dt4.configure_communication_logging(enabled=enabled, log_dir=log_dir)
    else:
        dt4.log_t4_communication = bool(enabled)
        dt4.communication_log_dir = log_dir


def close_t4_runtime_logging(tamps):
    if tamps is None or len(tamps) < 4:
        return
    t4 = tamps[3]
    if not isinstance(t4, (tuple, list)) or len(t4) < 1:
        return
    dt4 = t4[0]
    if hasattr(dt4, 'close_communication_log'):
        dt4.close_communication_log()


def _report_allreduce_timing(comm, log, label, arr, elapsed, nchunks, max_count):
    if log is None:
        return

    elapsed_max = comm.allreduce(elapsed, op=MPI.MAX)
    if comm.Get_rank() != 0:
        return

    label = label or 'buffer'
    log.info('allreduce %s: %.4f sec max over %d ranks, buffer %s, chunks %d, chunk <= %s',
             label, elapsed_max, comm.Get_size(), rccsdt.format_size(arr.nbytes),
             nchunks, rccsdt.format_size(max_count * arr.dtype.itemsize))


def _allreduce_inplace_large(comm, buf, op=MPI.SUM, max_count=None, log=None, label=None, log_timing=False):
    '''In-place Allreduce for NumPy buffers larger than MPI's int count.'''
    arr = np.asarray(buf)
    if arr.size == 0 or comm.Get_size() == 1:
        return buf

    if max_count is None:
        max_count = _MPI_ALLREDUCE_MAX_BYTES // arr.dtype.itemsize
    max_count = max(1, min(int(max_count), _MPI_INT_MAX))

    if not arr.flags['C_CONTIGUOUS']:
        time0 = MPI.Wtime() if log_timing else None
        tmp = np.ascontiguousarray(arr)
        _allreduce_inplace_large(comm, tmp, op=op, max_count=max_count)
        arr[...] = tmp.reshape(arr.shape)
        if log_timing:
            nchunks = (tmp.size + max_count - 1) // max_count
            _report_allreduce_timing(comm, log, label, arr, MPI.Wtime() - time0, nchunks, max_count)
        return buf

    flat = arr.reshape(-1)
    nchunks = (flat.size + max_count - 1) // max_count
    time0 = MPI.Wtime() if log_timing else None
    if flat.size <= max_count:
        comm.Allreduce(MPI.IN_PLACE, arr, op=op)
    else:
        for p0 in range(0, flat.size, max_count):
            comm.Allreduce(MPI.IN_PLACE, flat[p0:p0 + max_count], op=op)
    if log_timing:
        _report_allreduce_timing(comm, log, label, arr, MPI.Wtime() - time0, nchunks, max_count)
    return buf


def _allreduce_inplace_mpi_rccsdtq(mycc, buf, label, op=MPI.SUM):
    log_timing = getattr(mycc, 'log_allreduce_timing', False)
    log = None
    if log_timing:
        log = logger.Logger(mycc.stdout, logger.INFO if mycc.rank == 0 else 0)
    return _allreduce_inplace_large(mycc.comm, buf, op=op, log=log, label=label, log_timing=log_timing)


_t4_axis_labels = "abcd"
_t4_output_axes = tuple(range(4))
_t4_transpose_axes = tuple(''.join(p) for p in permutations(_t4_axis_labels))
_t4_transpose_slots = {axes: slot for slot, axes in enumerate(_t4_transpose_axes)}
_t4_transpose_einsums = tuple("pq,pabcd->q%s" % axes for axes in _t4_transpose_axes)
_woooo_target_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

def _oo_pair_index(i, j, nocc):
    if i > j:
        i, j = j, i
    return i * nocc - i * (i - 1) // 2 + (j - i)

def _w_oovvvv_pair(W_oovvvv, i, j, nocc):
    if i <= j:
        return W_oovvvv[_oo_pair_index(i, j, nocc)]
    else:
        raise IndexError("Invalid occupied pair index: i=%d, j=%d" % (i, j))

@functools.lru_cache(maxsize=None)
def _canonical_perm4(i, j, k, l):
    values = (int(i), int(j), int(k), int(l))
    order = tuple(sorted(range(4), key=lambda p: (values[p], p)))
    canonical = tuple(values[p] for p in order)
    perm = tuple(order.index(p) for p in range(4))
    return canonical, perm

@functools.lru_cache(maxsize=None)
def _woooo_t4_transpose_terms(o0, o1, o2, o3, i, j, k, l):
    """Generate all W_oooo[m,n,p,q] * T4[r,s,m,n] transpose terms."""
    canonical = (int(o0), int(o1), int(o2), int(o3))
    target = (int(i), int(j), int(k), int(l))
    terms = []

    for p, q in _woooo_target_pairs:
        complement = tuple(axis for axis in _t4_output_axes if axis not in (p, q))
        requested_head = (target[complement[0]], target[complement[1]])
        requested_to_output = complement + (p, q)
        transpose_to_output = tuple(requested_to_output.index(axis) for axis in _t4_output_axes)

        seen = set()
        for requested in permutations(canonical):
            if requested in seen:
                continue
            seen.add(requested)

            if requested[:2] != requested_head:
                continue

            canonical_check, perm = _canonical_perm4(*requested)
            if canonical_check != canonical:
                continue

            transpose_axes = ''.join(_t4_axis_labels[perm[axis]] for axis in transpose_to_output)
            terms.append((p, q, requested[2], requested[3], "abcd->" + transpose_axes))

    return tuple(terms)

@functools.lru_cache(maxsize=None)
def _foo_t4_transpose_terms(o0, o1, o2, o3, i, j, k, l):
    """Generate all F_oo[m,p] * T4[m,...] transpose terms."""
    canonical = (int(o0), int(o1), int(o2), int(o3))
    target = (int(i), int(j), int(k), int(l))
    terms = []

    for p in _t4_output_axes:
        complement = tuple(axis for axis in _t4_output_axes if axis != p)
        requested_tail = tuple(target[axis] for axis in complement)
        requested_to_output = (p,) + complement
        transpose_to_output = tuple(requested_to_output.index(axis) for axis in _t4_output_axes)

        seen = set()
        for requested in permutations(canonical):
            if requested in seen:
                continue
            seen.add(requested)

            if requested[1:] != requested_tail:
                continue

            canonical_check, perm = _canonical_perm4(*requested)
            if canonical_check != canonical:
                continue

            transpose_axes = ''.join(_t4_axis_labels[perm[axis]] for axis in transpose_to_output)
            terms.append((p, requested[0], "abcd->" + transpose_axes))

    return tuple(terms)

def _accumulate_oo_t4_prefactors_(pfs, W_oooo, F_oo, batch_ijkl_reordered, local_ijkl):
    for idx, (o00, o10, o20, o30) in enumerate(batch_ijkl_reordered):
        for i, j, k, l, local_idx in local_ijkl:
            target_occ = (i, j, k, l)
            for p, m, transpose_spec in _foo_t4_transpose_terms(o00, o10, o20, o30, i, j, k, l):
                slot = _t4_transpose_slots[transpose_spec[6:]]
                pfs[slot, idx, local_idx] -= F_oo[m, target_occ[p]]
            for p, q, m, n, transpose_spec in _woooo_t4_transpose_terms(o00, o10, o20, o30, i, j, k, l):
                slot = _t4_transpose_slots[transpose_spec[6:]]
                pfs[slot, idx, local_idx] += W_oooo[m, n, target_occ[p], target_occ[q]]

def init_amps_rhf(mycc, eris=None):
    '''Initialize CC T-amplitudes for an RHF reference.'''
    if mycc.rank == 0:
        time0 = logger.process_clock(), logger.perf_counter()
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    e_hf = mycc.e_hf
    if e_hf is None: e_hf = mycc.get_e_hf(mo_coeff=mycc.mo_coeff)

    mo_e = eris.mo_energy
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    t1 = eris.fock[:nocc, nocc:] / eia
    t2 = eris.pppp[:nocc, :nocc, nocc:, nocc:] / eijab

    tau = t2 + einsum("ia,jb->ijab", t1, t1)
    e_corr = 2.0 * einsum("ijab,ijab->", eris.pppp[:nocc, :nocc, nocc:, nocc:], tau)
    e_corr -= einsum("ijba,ijab->", eris.pppp[:nocc, :nocc, nocc:, nocc:], tau)
    e_corr += 2.0 * einsum("ai,ia->", eris.fock[nocc:, :nocc], t1)

    if mycc.rank == 0:
        logger.info(mycc, "Init t2, MP2 energy = % .12f  E_corr(MP2) % .12f" % (e_hf + e_corr, e_corr))

    if mycc.cc_order > 4:
        raise NotImplementedError("Only CCSDTQ (cc_order=4) is implemented in mpi_rccsdtq.py")

    t3 = np.zeros((nocc,) * (3) + (nvir,) * (3), dtype=t1.dtype)
    tamps = [t1, t2, t3]

    if mycc.do_tri_max_t:
        # t4 amplitude is distributed across MPI ranks
        dt4 = DistributedT4IJKL(nocc, nvir, comm=mycc.comm, batch_size=mycc.batch_size, dtype=t1.dtype,
                              allow_python_fallback=mycc.allow_python_fallback)
        dt4.log = logger.new_logger(mycc)
        dt4.configure_communication_logging(enabled=mycc.log_highest_t_communication,
                                            log_dir=mycc.communication_log_dir)
        dt4.print_distribution_info()
        t4_local = dt4.allocate_local()
        t4 = (dt4, t4_local)
    else:
        raise NotImplementedError("Only tri-stored T4 amplitudes are implemented in mpi_rccsdtq.py")
    tamps.append(t4)

    if mycc.rank == 0:
        logger.timer(mycc, 'init mp2', *time0)

    return e_corr, tamps

def memory_estimate_log_mpi_rccsdtq(mycc):
    '''Estimate per-rank memory for the distributed RCCSDTQ implementation.'''
    if mycc.rank != 0:
        return mycc

    log = logger.Logger(mycc.stdout, mycc.verbose if mycc.rank == 0 else 0)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    itemsize = np.dtype(np.float64).itemsize
    nvir4 = nvir**4
    size = mycc.size

    nocc4_full = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24
    nocc4_zero = nocc * nocc
    nocc4_stored = nocc4_full - nocc4_zero
    local_max = (nocc4_stored + size - 1) // size

    local_t4_memory = local_max * nvir4 * itemsize
    local_r4_memory = local_t4_memory
    global_t4_footprint = nocc4_stored * nvir4 * itemsize

    if nocc4_stored > 0:
        batch_size = nocc4_stored if mycc.batch_size is None else int(mycc.batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        batch = min(batch_size, nocc4_stored)
    else:
        batch = 0
    n_batches = (nocc4_stored + batch - 1) // batch if batch > 0 else 0
    t4_block_memory = nvir4 * itemsize
    if size > 1 and batch > 0:
        prefetch_slots = 2 if n_batches > 1 else 1
        t4_batch_recv_memory = prefetch_slots * batch * t4_block_memory
        t4_batch_send_memory = prefetch_slots * min(batch, local_max) * t4_block_memory
        t4_batch_collect_memory = 0
    else:
        prefetch_slots = 0
        t4_batch_recv_memory = 0
        t4_batch_send_memory = 0
        t4_batch_collect_memory = batch * t4_block_memory
    t4_batch_memory = (
        t4_batch_recv_memory + t4_batch_send_memory + t4_batch_collect_memory)
    prefactor_memory = len(_t4_transpose_axes) * batch * local_max * itemsize
    distribution_setup_memory = nocc4_stored * n_batches * itemsize

    eris_memory = nmo**4 * itemsize
    eris_runtime_memory = 3 * eris_memory
    nocc_pair = nocc * (nocc + 1) // 2
    r2_loc_memory = nocc**2 * nvir**2 * itemsize
    r3_loc_memory = nocc**3 * nvir**3 * itemsize
    w_oovvvo_memory = nocc**3 * nvir**3 * itemsize
    w_ovovvo_memory = nocc**3 * nvir**3 * itemsize
    w_ooooov_memory = nocc**5 * nvir * itemsize
    w_oooovv_memory = nocc**4 * nvir**2 * itemsize
    w_oovvvv_memory = nocc_pair * nvir4 * itemsize
    tmp_ovvo_memory = nocc**2 * nvir**2 * itemsize
    w_voov_memory = nocc**2 * nvir**2 * itemsize
    t3_work_memory = r3_loc_memory
    t4_oo_temp_memory = 5 * t4_block_memory

    r2_t4_work_memory = r2_loc_memory + t4_block_memory
    r3_t4_work_memory = r3_loc_memory + t4_block_memory
    r4_oovvvo_work_memory = w_oovvvo_memory + t3_work_memory
    r4_ovovvo_work_memory = w_ovovvo_memory + w_ooooov_memory + t3_work_memory
    r4_oooovv_oovvvv_work_memory = max(w_oooovv_memory + w_oovvvv_memory,
                    w_oooovv_memory + w_oovvvv_memory + t4_block_memory,
                    w_oooovv_memory + w_oovvvv_memory + tmp_ovvo_memory)
    r4_oo_work_memory = (w_voov_memory + t4_batch_memory + prefactor_memory + t4_oo_temp_memory)
    intermediates_work_memory = max(r2_t4_work_memory, r3_t4_work_memory, r4_oovvvo_work_memory, r4_ovovvo_work_memory,
                                    r4_oooovv_oovvvv_work_memory, t4_block_memory, r4_oo_work_memory,)

    diis_scratch = resolve_diis_scratch(mycc)
    diis_incore_space = resolve_diis_incore_space(mycc)
    diis_memory = 0.0
    diis_history_memory = 0.0
    diis_xprev_memory = 0.0
    diis_resident_memory = 0.0
    diis_live_memory = 0.0
    diis_scratch_memory = 0.0
    diis_vector_memory = 0.0
    if mycc.diis and mycc.do_diis_max_t:
        nvir_t4_diis = resolve_nvir_diis(mycc, nvir)
        nocc2_unique = nocc * (nocc + 1) // 2
        nocc3_unique = nocc * (nocc + 1) * (nocc + 2) // 6
        lower_t_memory = (nocc * nvir + nocc2_unique * nvir**2 + nocc3_unique * nvir**3) * itemsize
        local_t4_diis_memory = local_max * nvir_t4_diis**4 * itemsize
        diis_vector_memory = lower_t_memory + local_t4_diis_memory
        diis_history_memory = diis_vector_memory * diis_incore_space * 2
        diis_xprev_memory = diis_vector_memory
        diis_resident_memory = diis_history_memory + diis_xprev_memory
        # The vector passed to DIIS, the current error, extrapolated vector,
        # and previous-vector copy are live transiently even for out-of-core DIIS.
        diis_live_memory = 4 * diis_vector_memory
        diis_memory = diis_resident_memory + diis_live_memory
        diis_scratch_memory = (diis_vector_memory * (mycc.diis_space - diis_incore_space) * 2)
    elif mycc.diis and mycc.incore_complete:
        nocc2_unique = nocc * (nocc + 1) // 2
        nocc3_unique = nocc * (nocc + 1) * (nocc + 2) // 6
        diis_vector_memory = (nocc * nvir + nocc2_unique * nvir**2 + nocc3_unique * nvir**3) * itemsize
        diis_memory = diis_vector_memory * mycc.diis_space * 2

    update_work_peak_memory = local_t4_memory + local_r4_memory + eris_runtime_memory + intermediates_work_memory
    update_peak_memory = update_work_peak_memory + diis_resident_memory
    diis_peak_memory = local_t4_memory + eris_memory + diis_memory
    total_memory = max(update_peak_memory, diis_peak_memory)
    current_memory = int(lib.current_memory()[0] * 1024**2)
    projected_peak_memory = current_memory + total_memory

    fmt = rccsdt.format_size
    log.info('Approximate per-rank memory usage estimate')
    log.info('    Current rank-0 memory   %8s', fmt(current_memory))
    log.info('    Local T4 memory (max)   %8s', fmt(local_t4_memory))
    log.info('    Local R4 memory (max)   %8s', fmt(local_r4_memory))
    log.info('    Global T4 footprint     %8s', fmt(global_t4_footprint))
    log.info('    T4 prefetch recv bufs   %8s', fmt(t4_batch_recv_memory))
    log.info('    T4 prefetch send bufs   %8s', fmt(t4_batch_send_memory))
    if prefetch_slots:
        log.info('    T4 prefetch slots       %8d', prefetch_slots)
    if t4_batch_collect_memory:
        log.info('    T4 local batch data     %8s', fmt(t4_batch_collect_memory))
    log.info('    OO prefactor buffer     %8s', fmt(prefactor_memory))
    log.info('    OO T4 temp buffers      %8s', fmt(t4_oo_temp_memory))
    log.info('    ERIs memory             %8s', fmt(eris_memory))
    log.info('    T1-ERIs memory          %8s', fmt(eris_memory))
    log.info('    ERIs work buffer        %8s', fmt(eris_memory))
    log.info('    Intermediates/work peak %8s', fmt(intermediates_work_memory))
    log.info('    DIIS memory             %8s', fmt(diis_memory))
    if mycc.do_diis_max_t and mycc.diis:
        log.info('    DIIS resident memory    %8s', fmt(diis_resident_memory))
        log.info('    DIIS transient memory   %8s', fmt(diis_live_memory))
        log.info('   nvir_diis          %8d of %d', nvir_t4_diis, nvir)
        log.info('    DIIS vector size        %8s', fmt(diis_vector_memory))
    if mycc.do_diis_max_t and mycc.diis and diis_scratch is not None:
        log.info('    DIIS in-core slots      %8d of %d', diis_incore_space, mycc.diis_space)
        log.info('    DIIS scratch footprint  %8s', fmt(diis_scratch_memory))
        log.info('    DIIS scratch dir        %s', diis_scratch)
    log.info('Update work peak            %8s', fmt(update_work_peak_memory))
    log.info('Update estimated peak       %8s', fmt(update_peak_memory))
    if mycc.diis:
        log.info('DIIS estimated peak         %8s', fmt(diis_peak_memory))
    log.info('Additional estimated memory %8s', fmt(total_memory))
    log.info('Projected rank-0 peak       %8s', fmt(projected_peak_memory))
    log.info('Rank-0 T4 balance setup     %8s transient', fmt(distribution_setup_memory))

    max_memory = mycc.max_memory - lib.current_memory()[0]
    if (total_memory / 1024**2) > max_memory:
        logger.warn(mycc, 'Estimated additional per-rank memory %s exceeds available %s '
                    '(projected peak %s vs max_memory %s)',
                    fmt(total_memory), fmt(max_memory * 1024**2),
                    fmt(projected_peak_memory), fmt(mycc.max_memory * 1024**2))
    return mycc

def r2_add_t4_tri_(mycc, imds, r2, t4):
    '''Add the T4 contributions to r2. T4 amplitudes are stored in triangular form.'''
    time_total = logger.process_clock(), logger.perf_counter()
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    dt4, t4_local = t4

    t1_eris = imds.t1_eris

    r2_loc = np.zeros_like(r2)
    t4_tmp = np.empty((nvir,) * 4, dtype=np.float64)
    log_memory(mycc, memlog, 'r2 T4 buffers allocated')
    time1 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l in dt4.local_quadruples:
        if i < j < k < l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, j, nocc:, nocc:], t4_tmp, out=r2_loc[k, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, k, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, k, nocc:, nocc:], t4_tmp, out=r2_loc[j, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, l, j, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, l, nocc:, nocc:], t4_tmp, out=r2_loc[j, k, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, k, i, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[j, k, nocc:, nocc:], t4_tmp, out=r2_loc[i, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, l, i, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[j, l, nocc:, nocc:], t4_tmp, out=r2_loc[i, k, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, l, i, j)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[k, l, nocc:, nocc:], t4_tmp, out=r2_loc[i, j, :, :], alpha=1.0, beta=1.0)
        elif i == j < k < l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, j, nocc:, nocc:], t4_tmp, out=r2_loc[k, l, :, :], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, k, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, k, nocc:, nocc:], t4_tmp, out=r2_loc[j, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, l, j, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, l, nocc:, nocc:], t4_tmp, out=r2_loc[j, k, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, l, i, j)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[k, l, nocc:, nocc:], t4_tmp, out=r2_loc[i, j, :, :], alpha=0.5, beta=1.0)
        elif i < j == k < l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, j, nocc:, nocc:], t4_tmp, out=r2_loc[k, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, l, j, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, l, nocc:, nocc:], t4_tmp, out=r2_loc[j, k, :, :], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, k, i, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[j, k, nocc:, nocc:], t4_tmp, out=r2_loc[i, l, :, :], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, l, i, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[j, l, nocc:, nocc:], t4_tmp, out=r2_loc[i, k, :, :], alpha=1.0, beta=1.0)
        elif i < j < k == l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, j, nocc:, nocc:], t4_tmp, out=r2_loc[k, l, :, :], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, k, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, k, nocc:, nocc:], t4_tmp, out=r2_loc[j, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, k, i, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[j, k, nocc:, nocc:], t4_tmp, out=r2_loc[i, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, l, i, j)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[k, l, nocc:, nocc:], t4_tmp, out=r2_loc[i, j, :, :], alpha=0.5, beta=1.0)
        elif i == j < k == l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, j, nocc:, nocc:], t4_tmp, out=r2_loc[k, l, :, :], alpha=0.25, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, k, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[i, k, nocc:, nocc:], t4_tmp, out=r2_loc[j, l, :, :], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, l, i, j)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_442", 1.0, 0.0)
            einsum('ef,efab->ab', t1_eris[k, l, nocc:, nocc:], t4_tmp, out=r2_loc[i, j, :, :], alpha=0.25, beta=1.0)
        time1 = log.timer_debug1(contraction_message(mycc, 'r2: iter: T1-ERIs * T4 [%2d, %2d, %2d, %2d]:' %
                                                            (i, j, k, l)), *time1)

    time1 = log.timer_debug1(contraction_message(mycc, 'r2: local T4 contractions'), *time1)
    log_memory(mycc, memlog, 'r2 local T4 contractions')
    r2_loc = np.ascontiguousarray(r2_loc)
    _allreduce_inplace_mpi_rccsdtq(mycc, r2_loc, 'r2_loc', op=MPI.SUM)
    r2 += r2_loc
    time1 = log.timer_debug1(contraction_message(mycc, 'r2: reduce/add T4 contribution'), *time1)
    log_memory(mycc, memlog, 'r2 reduce/add T4 contribution')

    t4_tmp = None
    r2_loc = None
    log_memory(mycc, memlog, 'r2 T4 buffers released')

    log.timer_debug1(contraction_message(mycc, 'r2: add T4 contribution total'), *time_total)
    return r2

def r3_add_t4_tri_(mycc, imds, r3, t4):
    '''Add the T4 contributions to r3. T4 amplitudes are stored in triangular form.'''
    time_total = logger.process_clock(), logger.perf_counter()
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    dt4, t4_local = t4

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    r3_loc = np.zeros_like(r3)
    t4_tmp = np.empty((nvir,) * 4, dtype=np.float64)
    log_memory(mycc, memlog, 'r3 T4 buffers allocated')
    time1 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l in dt4.local_quadruples:
        if i < j < k < l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[i, nocc:], t4_tmp, out=r3_loc[j, k, l], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, i, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[j, nocc:], t4_tmp, out=r3_loc[i, k, l], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, i, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[k, nocc:], t4_tmp, out=r3_loc[i, j, l], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, l, i, j, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[l, nocc:], t4_tmp, out=r3_loc[i, j, k], alpha=1.0, beta=1.0)
        elif i == j < k < l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[i, nocc:], t4_tmp, out=r3_loc[j, k, l], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, i, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[k, nocc:], t4_tmp, out=r3_loc[i, j, l], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, l, i, j, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[l, nocc:], t4_tmp, out=r3_loc[i, j, k], alpha=0.5, beta=1.0)
        elif i < j == k < l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[i, nocc:], t4_tmp, out=r3_loc[j, k, l], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, i, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[j, nocc:], t4_tmp, out=r3_loc[i, k, l], alpha=1.0, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, l, i, j, k)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[l, nocc:], t4_tmp, out=r3_loc[i, j, k], alpha=0.5, beta=1.0)
        elif i < j < k == l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[i, nocc:], t4_tmp, out=r3_loc[j, k, l], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, j, i, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[j, nocc:], t4_tmp, out=r3_loc[i, k, l], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, i, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[k, nocc:], t4_tmp, out=r3_loc[i, j, l], alpha=1.0, beta=1.0)
        elif i == j < k == l:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[i, nocc:], t4_tmp, out=r3_loc[j, k, l], alpha=0.5, beta=1.0)
            dt4.unpack_t4_single_local(t4_local, t4_tmp, k, i, j, l)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('e,eabc->abc', t1_fock[k, nocc:], t4_tmp, out=r3_loc[i, j, l], alpha=0.5, beta=1.0)

        time1 = log.timer_debug1(contraction_message(mycc, 'r3: iter: T1-Fock * T4 [%2d, %2d, %2d, %2d]:' %
                                (i, j, k, l)), *time1)

    time1 = log.timer_debug1(contraction_message(mycc, 'r3: local T1-Fock * T4'), *time1)
    log_memory(mycc, memlog, 'r3 local T1-Fock * T4')
    for i, j, k, l in dt4.local_quadruples:
        inds = (i, j, k, l)
        if max(Counter(inds).values()) >= 3:
            perms = []
        else:
            perms = list(dict.fromkeys(permutations(inds)))
        # NOTE: This part can be further optimized by the symmetry of t4_tmp
        for (pi, pj, pk, pl) in perms:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, pi, pj, pk, pl)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('aef,febc->abc', t1_eris[nocc:, pi, nocc:, nocc:], t4_tmp, out=r3_loc[pj, pk, pl],
                   alpha=0.5, beta=1.0)
            einsum('en,eabc->nabc', t1_eris[pi, pk, nocc:, :nocc], t4_tmp, out=r3_loc[pj, :, pl], alpha=-0.5, beta=1.0)

        time1 = log.timer_debug1(contraction_message(mycc, 'r3: iter: T1-ERIs * T4 [%2d, %2d, %2d, %2d]:' %
                                (i, j, k, l)), *time1)

    time1 = log.timer_debug1(contraction_message(mycc, 'r3: local T1-ERIs * T4'), *time1)
    log_memory(mycc, memlog, 'r3 local T1-ERIs * T4')
    r3_loc = np.ascontiguousarray(r3_loc)
    _allreduce_inplace_mpi_rccsdtq(mycc, r3_loc, 'r3_loc', op=MPI.SUM)
    r3 += r3_loc
    time1 = log.timer_debug1(contraction_message(mycc, 'r3: reduce/add T4 contribution'), *time1)
    log_memory(mycc, memlog, 'r3 reduce/add T4 contribution')

    t4_tmp = None
    r3_loc = None
    log_memory(mycc, memlog, 'r3 T4 buffers released')

    log.timer_debug1(contraction_message(mycc, 'r3: add T4 contribution total'), *time_total)
    return r3

def intermediates_t4_tri(mycc, imds, t2, t3, t4):
    '''Intermediates for the T4 residual equation, with T4 amplitudes stored in triangular form.
    In place modification of W_vvvo.
    Heavy T4 intermediates are built in compute_r4_tri immediately before their contraction phase.
    '''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc = mycc.nocc

    t1_fock = imds.t1_fock
    W_vvvo = imds.W_vvvo

    einsum('me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=W_vvvo, alpha=-1.0, beta=1.0)
    return imds

def _build_w_oovvvo(mycc, imds, t2, c_t3):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_eris = imds.t1_eris
    W_oovvvo = np.empty((nocc,) * 2 + (nvir,) * 3 + (nocc,))
    einsum('maef,jibf->ijeabm', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_oovvvo, alpha=2.0, beta=0.0)
    einsum('mafe,jibf->ijeabm', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_oovvvo, alpha=-1.0, beta=1.0)
    einsum('mnei,njab->ijeabm', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_oovvvo, alpha=-2.0, beta=1.0)
    einsum('nmei,njab->ijeabm', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_oovvvo, alpha=1.0, beta=1.0)
    einsum('nmfe,nijfab->ijeabm', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_oovvvo, alpha=0.5, beta=1.0)
    einsum('mnfe,nijfab->ijeabm', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_oovvvo, alpha=-0.25, beta=1.0)
    W_oovvvo += W_oovvvo.transpose(1, 0, 2, 4, 3, 5)
    return W_oovvvo

def _build_w_ovovvo(mycc, imds, t2, t3):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_eris = imds.t1_eris
    W_ovovvo = np.empty((nocc, nvir, nocc, nvir, nvir, nocc))
    einsum('mafe,jibf->iejabm', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovovvo, alpha=1.0, beta=0.0)
    einsum('mnie,njab->iejabm', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovovvo, alpha=-1.0, beta=1.0)
    einsum('nmef,injfab->iejabm', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovovvo, alpha=-0.5, beta=1.0)
    return W_ovovvo

def _build_w_ooooov(mycc, imds, t2, t3):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1_eris = imds.t1_eris
    W_ooooov = np.empty((nocc,) * 5 + (nvir,))
    einsum('mnek,ijae->kjinma', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ooooov, alpha=1.0, beta=0.0)
    einsum('mnef,ijkaef->kjinma', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ooooov, alpha=0.5, beta=1.0)
    W_ooooov += W_ooooov.transpose(1, 0, 2, 4, 3, 5)
    return W_ooooov

def _build_w_oooovv_oovvvv(mycc, imds, t2, t3, t4):
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    dt4, t4_local = t4

    t1_eris = imds.t1_eris
    W_oooo, W_ovov, W_vvvv = imds.W_oooo, imds.W_ovov, imds.W_vvvv

    time_part = logger.process_clock(), logger.perf_counter()
    nocc_pair = nocc * (nocc + 1) // 2
    W_oooovv = np.zeros((nocc,) * 4 + (nvir,) * 2, dtype=t4_local.dtype)
    W_oovvvv = np.zeros((nocc_pair,) + (nvir,) * 4, dtype=t4_local.dtype)
    t4_tmp = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    log_memory(mycc, memlog, 'W_oooovv/W_oovvvv T4 buffers allocated')
    time_part = log.timer_debug1(contraction_message(mycc, 'r4: build W_oooovv/W_oovvvv: T4 buffer init'), *time_part)

    for i, j, k, l in dt4.local_quadruples:
        inds = (i, j, k, l)
        if max(Counter(inds).values()) >= 3:
            perms = []
        else:
            perms = list(dict.fromkeys(permutations(inds)))
        # NOTE: This part can be further optimized by the symmetry of t4_tmp
        for (pi, pj, pk, pl) in perms:
            dt4.unpack_t4_single_local(t4_local, t4_tmp, pi, pj, pk, pl)
            t4_single_spin_summation_inplace_(t4_tmp, nvir, "P4_201", 1.0, 0.0)
            einsum('mef,fabe->mab', t1_eris[:nocc, pi, nocc:, nocc:], t4_tmp, out=W_oooovv[pj, pk, pl],
                   alpha=0.5, beta=1.0)
            pair_idx = _oo_pair_index(pk, pl, nocc)
            if pk < pl:
                einsum('ef,fabc->abce', t1_eris[pj, pi, nocc:, nocc:], t4_tmp, out=W_oovvvv[pair_idx],
                       alpha=-0.5, beta=1.0)
            elif pk > pl:
                einsum('ef,fabc->acbe', t1_eris[pj, pi, nocc:, nocc:], t4_tmp, out=W_oovvvv[pair_idx],
                       alpha=-0.5, beta=1.0)
            else:
                einsum('ef,fabc->abce', t1_eris[pj, pi, nocc:, nocc:], t4_tmp, out=W_oovvvv[pair_idx],
                       alpha=-0.5, beta=1.0)
                einsum('ef,fabc->acbe', t1_eris[pj, pi, nocc:, nocc:], t4_tmp, out=W_oovvvv[pair_idx],
                       alpha=-0.5, beta=1.0)
    time_part = log.timer_debug1(contraction_message(
        mycc, 'r4: build W_oooovv/W_oovvvv: local T4 contractions'), *time_part)

    _allreduce_inplace_mpi_rccsdtq(mycc, W_oooovv, 'W_oooovv_t4', op=MPI.SUM)
    time_part = log.timer_debug1(contraction_message(
        mycc, 'r4: build W_oooovv/W_oovvvv: allreduce W_oooovv T4'), *time_part)

    _allreduce_inplace_mpi_rccsdtq(mycc, W_oovvvv, 'W_oovvvv_t4', op=MPI.SUM)
    time_part = log.timer_debug1(contraction_message(
        mycc, 'r4: build W_oooovv/W_oovvvv: allreduce W_oovvvv T4'), *time_part)

    t4_tmp = None
    log_memory(mycc, memlog, 'W_oooovv/W_oovvvv T4 contribution reduced')
    time_part = log.timer_debug1(contraction_message(
        mycc, 'r4: build W_oooovv/W_oovvvv: release T4 buffer'), *time_part)

    einsum('amef,ijkebf->ijkmab', t1_eris[nocc:, :nocc, nocc:, nocc:], t3, out=W_oooovv, alpha=1.0, beta=1.0)
    log_memory(mycc, memlog, 'W_oooovv T3 base contribution added')
    tmp_ovvo = t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    einsum('nmfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=tmp_ovvo, alpha=1.0, beta=1.0)
    einsum('mnfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=tmp_ovvo, alpha=-0.5, beta=1.0)
    c_t2 = None
    einsum('nmef,infa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tmp_ovvo, alpha=-0.5, beta=1.0)
    einsum('maei,jkbe->ijkmab', tmp_ovvo, t2, out=W_oooovv, alpha=1.0, beta=1.0)
    tmp_ovvo = None
    einsum('make,jibe->ijkmab', W_ovov, t2, out=W_oooovv, alpha=1.0, beta=1.0)
    einsum('mnki,njab->ijkmab', W_oooo, t2, out=W_oooovv, alpha=-0.5, beta=1.0)
    time_part = log.timer_debug1(contraction_message(
        mycc, 'r4: build W_oooovv/W_oovvvv: replicated W_oooovv'), *time_part)

    for j in range(nocc):
        for k in range(j, nocc):
            pair_idx = _oo_pair_index(j, k, nocc)
            W_oovvvv_jk = W_oovvvv[pair_idx]
            einsum('abef,fc->abce', W_vvvv, t2[j, k], out=W_oovvvv_jk, alpha=0.5, beta=1.0)
            einsum('acef,fb->abce', W_vvvv, t2[k, j], out=W_oovvvv_jk, alpha=0.5, beta=1.0)
    log_memory(mycc, memlog, 'W_oovvvv replicated contribution added')
    time_part = log.timer_debug1(contraction_message(
        mycc, 'r4: build W_oooovv/W_oovvvv: replicated W_oovvvv'), *time_part)

    W_oooovv += W_oooovv.transpose(1, 0, 2, 3, 5, 4)
    log_memory(mycc, memlog, 'W_oooovv symmetrized')
    log.timer_debug1(contraction_message(mycc, 'r4: build W_oooovv/W_oovvvv: symmetrize W_oooovv'), *time_part)
    return W_oooovv, W_oovvvv

def compute_r4_tri(mycc, imds, t2, t3, t4):
    '''Compute r4 with triangular-stored T4 amplitudes; r4 is returned in triangular form as well.
    r4 will require a symmetry restoration step afterward.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    dt4, t4_local = t4

    F_vv = imds.F_vv
    W_vvvo, W_vooo, W_vvvv = imds.W_vvvo, imds.W_vooo, imds.W_vvvv

    t3_tmp = np.empty_like(t3)
    t3_spin_summation(t3, t3_tmp, nocc**3, nvir, "P3_201", 1.0, 0.0)
    time_build = logger.process_clock(), logger.perf_counter()
    W_oovvvo = _build_w_oovvvo(mycc, imds, t2, t3_tmp)
    time_build = log.timer_debug1(contraction_message(mycc, 'r4: build W_oovvvo'), *time_build)
    log_memory(mycc, memlog, 'r4 build W_oovvvo')
    r4_local = np.empty_like(t4_local)
    log_memory(mycc, memlog, 'r4_local allocated')

    time2 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l, local_idx in dt4.iter_local_ijkl():
        r4_tmp = r4_local[local_idx]

        einsum("abe,ecd->abcd", W_vvvo[..., j], t3[i, k, l], out=r4_tmp, alpha=1.0, beta=0.0)
        einsum("ace,ebd->abcd", W_vvvo[..., k], t3[i, j, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("ade,ebc->abcd", W_vvvo[..., l], t3[i, j, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bae,ecd->abcd", W_vvvo[..., i], t3[j, k, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cae,ebd->abcd", W_vvvo[..., i], t3[k, j, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("dae,ebc->abcd", W_vvvo[..., i], t3[l, j, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bce,ead->abcd", W_vvvo[..., k], t3[j, i, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bde,eac->abcd", W_vvvo[..., l], t3[j, i, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cbe,ead->abcd", W_vvvo[..., j], t3[k, i, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("dbe,eac->abcd", W_vvvo[..., j], t3[l, i, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cde,eab->abcd", W_vvvo[..., l], t3[k, i, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("dce,eab->abcd", W_vvvo[..., k], t3[l, i, j], out=r4_tmp, alpha=1.0, beta=1.0)

        einsum("am,mbcd->abcd", W_vooo[:, :, i, j], t3[:, k, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("am,mcbd->abcd", W_vooo[:, :, i, k], t3[:, j, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("am,mdbc->abcd", W_vooo[:, :, i, l], t3[:, j, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("bm,macd->abcd", W_vooo[:, :, j, i], t3[:, k, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("cm,mabd->abcd", W_vooo[:, :, k, i], t3[:, j, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("dm,mabc->abcd", W_vooo[:, :, l, i], t3[:, j, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("bm,mcad->abcd", W_vooo[:, :, j, k], t3[:, i, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("bm,mdac->abcd", W_vooo[:, :, j, l], t3[:, i, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("cm,mbad->abcd", W_vooo[:, :, k, j], t3[:, i, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("dm,mbac->abcd", W_vooo[:, :, l, j], t3[:, i, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("cm,mdab->abcd", W_vooo[:, :, k, l], t3[:, i, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("dm,mcab->abcd", W_vooo[:, :, l, k], t3[:, i, j], out=r4_tmp, alpha=-1.0, beta=1.0)

        einsum("eabm,mecd->abcd", W_oovvvo[i, j], t3_tmp[:, k, l], out=r4_tmp, alpha=0.5, beta=1.0)
        einsum("eacm,mebd->abcd", W_oovvvo[i, k], t3_tmp[:, j, l], out=r4_tmp, alpha=0.5, beta=1.0)
        einsum("eadm,mebc->abcd", W_oovvvo[i, l], t3_tmp[:, j, k], out=r4_tmp, alpha=0.5, beta=1.0)
        einsum("ebcm,mead->abcd", W_oovvvo[j, k], t3_tmp[:, i, l], out=r4_tmp, alpha=0.5, beta=1.0)
        einsum("ebdm,meac->abcd", W_oovvvo[j, l], t3_tmp[:, i, k], out=r4_tmp, alpha=0.5, beta=1.0)
        einsum("ecdm,meab->abcd", W_oovvvo[k, l], t3_tmp[:, i, j], out=r4_tmp, alpha=0.5, beta=1.0)

        time2 = log.timer_debug1(contraction_message(mycc, "r4: iter: W_vvvo * t3, W_vooo * t3, "
                                "W_oovvvo * t3 [%2d, %2d, %2d, %2d]:" % (i, j, k, l)), *time2)
    W_vvvo = imds.W_vvvo = None
    W_vooo = imds.W_vooo = None
    W_oovvvo = imds.W_oovvvo = None
    t3_tmp = None
    time1 = log.timer_debug1(contraction_message(mycc, 'r4: W_vvvo * t3, W_vooo * t3, W_oovvvo * t3'), *time1)
    log_memory(mycc, memlog, 'r4 W_vvvo/W_vooo/W_oovvvo contractions')

    # NOTE: now compute W_ovovvo and W_ooooov
    time_build = logger.process_clock(), logger.perf_counter()
    W_ovovvo = _build_w_ovovvo(mycc, imds, t2, t3)
    time_build = log.timer_debug1(contraction_message(mycc, 'r4: build W_ovovvo'), *time_build)
    log_memory(mycc, memlog, 'r4 build W_ovovvo')
    time2 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l, local_idx in dt4.iter_local_ijkl():
        r4_tmp = r4_local[local_idx]
        einsum("ecbm,maed->abcd", W_ovovvo[i, :, j], t3[:, k, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("edbm,maec->abcd", W_ovovvo[i, :, j], t3[:, l, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ebcm,maed->abcd", W_ovovvo[i, :, k], t3[:, j, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ebdm,maec->abcd", W_ovovvo[i, :, l], t3[:, j, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("edcm,maeb->abcd", W_ovovvo[i, :, k], t3[:, l, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ecdm,maeb->abcd", W_ovovvo[i, :, l], t3[:, k, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ecam,mbed->abcd", W_ovovvo[j, :, i], t3[:, k, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("edam,mbec->abcd", W_ovovvo[j, :, i], t3[:, l, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ebam,mced->abcd", W_ovovvo[k, :, i], t3[:, j, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ebam,mdec->abcd", W_ovovvo[l, :, i], t3[:, j, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("edam,mceb->abcd", W_ovovvo[k, :, i], t3[:, l, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ecam,mdeb->abcd", W_ovovvo[l, :, i], t3[:, k, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("eacm,mbed->abcd", W_ovovvo[j, :, k], t3[:, i, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("eadm,mbec->abcd", W_ovovvo[j, :, l], t3[:, i, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("eabm,mced->abcd", W_ovovvo[k, :, j], t3[:, i, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("eabm,mdec->abcd", W_ovovvo[l, :, j], t3[:, i, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("eadm,mceb->abcd", W_ovovvo[k, :, l], t3[:, i, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("eacm,mdeb->abcd", W_ovovvo[l, :, k], t3[:, i, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("edcm,mbea->abcd", W_ovovvo[j, :, k], t3[:, l, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ecdm,mbea->abcd", W_ovovvo[j, :, l], t3[:, k, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("edbm,mcea->abcd", W_ovovvo[k, :, j], t3[:, l, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ecbm,mdea->abcd", W_ovovvo[l, :, j], t3[:, k, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ebdm,mcea->abcd", W_ovovvo[k, :, l], t3[:, j, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("ebcm,mdea->abcd", W_ovovvo[l, :, k], t3[:, j, i], out=r4_tmp, alpha=-1.0, beta=1.0)

        time2 = log.timer_debug1(contraction_message(mycc, "r4: iter: W_ovovvo * t3 "
                                "[%2d, %2d, %2d, %2d]:" % (i, j, k, l)), *time2)

    time1 = log.timer_debug1(contraction_message(mycc, 'r4: W_ovovvo * t3'), *time1)
    log_memory(mycc, memlog, 'r4 W_ovovvo * t3')

    t3_tmp = t3 + t3.transpose(0, 1, 2, 4, 5, 3)
    W_ovovvo += W_ovovvo.transpose(2, 1, 0, 4, 3, 5)
    time_build = logger.process_clock(), logger.perf_counter()
    W_ooooov = _build_w_ooooov(mycc, imds, t2, t3)
    time_build = log.timer_debug1(contraction_message(mycc, 'r4: build W_ooooov'), *time_build)
    log_memory(mycc, memlog, 'r4 build W_ooooov')
    time2 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l, local_idx in dt4.iter_local_ijkl():
        r4_tmp = r4_local[local_idx]
        einsum("eabm,mced->abcd", W_ovovvo[i, :, j], t3_tmp[:, k, l], out=r4_tmp, alpha=-0.5, beta=1.0)
        einsum("eacm,mbed->abcd", W_ovovvo[i, :, k], t3_tmp[:, j, l], out=r4_tmp, alpha=-0.5, beta=1.0)
        einsum("eadm,mbec->abcd", W_ovovvo[i, :, l], t3_tmp[:, j, k], out=r4_tmp, alpha=-0.5, beta=1.0)
        einsum("ebcm,maed->abcd", W_ovovvo[j, :, k], t3_tmp[:, i, l], out=r4_tmp, alpha=-0.5, beta=1.0)
        einsum("ebdm,maec->abcd", W_ovovvo[j, :, l], t3_tmp[:, i, k], out=r4_tmp, alpha=-0.5, beta=1.0)
        einsum("ecdm,maeb->abcd", W_ovovvo[k, :, l], t3_tmp[:, i, j], out=r4_tmp, alpha=-0.5, beta=1.0)

        einsum("nma,mnbcd->abcd", W_ooooov[k, j, i], t3[:, :, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nma,mnbdc->abcd", W_ooooov[l, j, i], t3[:, :, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nma,mncdb->abcd", W_ooooov[l, k, i], t3[:, :, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmb,mnacd->abcd", W_ooooov[k, i, j], t3[:, :, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmb,mnadc->abcd", W_ooooov[l, i, j], t3[:, :, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmb,mncda->abcd", W_ooooov[l, k, j], t3[:, :, i], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmc,mnabd->abcd", W_ooooov[j, i, k], t3[:, :, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmc,mnadb->abcd", W_ooooov[l, i, k], t3[:, :, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmc,mnbda->abcd", W_ooooov[l, j, k], t3[:, :, i], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmd,mnabc->abcd", W_ooooov[j, i, l], t3[:, :, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmd,mnacb->abcd", W_ooooov[k, i, l], t3[:, :, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("nmd,mnbca->abcd", W_ooooov[k, j, l], t3[:, :, i], out=r4_tmp, alpha=1.0, beta=1.0)

        time2 = log.timer_debug1(contraction_message(mycc, "r4: iter: sym W_ovovvo * t3, W_ooooov * t3 "
                                                    "[%2d, %2d, %2d, %2d]:" % (i, j, k, l)), *time2)
    t3_tmp = None
    W_ovovvo = imds.W_ovovvo = None
    W_ooooov = imds.W_ooooov = None
    time1 = log.timer_debug1(contraction_message(mycc, 'r4: sym W_ovovvo * t3, W_ooooov * t3'), *time1)
    log_memory(mycc, memlog, 'r4 sym W_ovovvo/W_ooooov contractions')

    # NOTE: now compute W_oooovv and W_oovvvv
    time_build = logger.process_clock(), logger.perf_counter()
    W_oooovv, W_oovvvv = _build_w_oooovv_oovvvv(mycc, imds, t2, t3, t4)
    time_build = log.timer_debug1(contraction_message(mycc, 'r4: build W_oooovv, W_oovvvv'), *time_build)
    log_memory(mycc, memlog, 'r4 build W_oooovv/W_oovvvv')
    imds.t1_eris = None
    time2 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l, local_idx in dt4.iter_local_ijkl():
        r4_tmp = r4_local[local_idx]
        einsum("mab,mcd->abcd", W_oooovv[i, j, k], t2[:, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mab,mdc->abcd", W_oooovv[i, j, l], t2[:, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mac,mbd->abcd", W_oooovv[i, k, j], t2[:, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mad,mbc->abcd", W_oooovv[i, l, j], t2[:, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mac,mdb->abcd", W_oooovv[i, k, l], t2[:, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mad,mcb->abcd", W_oooovv[i, l, k], t2[:, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mbc,mad->abcd", W_oooovv[j, k, i], t2[:, l], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mbd,mac->abcd", W_oooovv[j, l, i], t2[:, k], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mcd,mab->abcd", W_oooovv[k, l, i], t2[:, j], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mbc,mda->abcd", W_oooovv[j, k, l], t2[:, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mbd,mca->abcd", W_oooovv[j, l, k], t2[:, i], out=r4_tmp, alpha=-1.0, beta=1.0)
        einsum("mcd,mba->abcd", W_oooovv[k, l, j], t2[:, i], out=r4_tmp, alpha=-1.0, beta=1.0)

        W_oovvvv_ij = _w_oovvvv_pair(W_oovvvv, i, j, nocc)
        W_oovvvv_ik = _w_oovvvv_pair(W_oovvvv, i, k, nocc)
        W_oovvvv_il = _w_oovvvv_pair(W_oovvvv, i, l, nocc)
        W_oovvvv_jk = _w_oovvvv_pair(W_oovvvv, j, k, nocc)
        W_oovvvv_jl = _w_oovvvv_pair(W_oovvvv, j, l, nocc)
        W_oovvvv_kl = _w_oovvvv_pair(W_oovvvv, k, l, nocc)

        einsum("abce,ed->abcd", W_oovvvv_jk, t2[i, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("abde,ec->abcd", W_oovvvv_jl, t2[i, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("acde,eb->abcd", W_oovvvv_kl, t2[i, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bace,ed->abcd", W_oovvvv_ik, t2[j, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bade,ec->abcd", W_oovvvv_il, t2[j, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cabe,ed->abcd", W_oovvvv_ij, t2[k, l], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("dabe,ec->abcd", W_oovvvv_ij, t2[l, k], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cade,eb->abcd", W_oovvvv_il, t2[k, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("dace,eb->abcd", W_oovvvv_ik, t2[l, j], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bcde,ea->abcd", W_oovvvv_kl, t2[j, i], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cbde,ea->abcd", W_oovvvv_jl, t2[k, i], out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("dbce,ea->abcd", W_oovvvv_jk, t2[l, i], out=r4_tmp, alpha=1.0, beta=1.0)

        time2 = log.timer_debug1(contraction_message(mycc, "r4: iter: W_oooovv * t2, W_oovvvv * t2 "
                                                    "[%2d, %2d, %2d, %2d]:" % (i, j, k, l)), *time2)
    W_oooovv = imds.W_oooovv = None
    W_oovvvv = imds.W_oovvvv = None
    time1 = log.timer_debug1(contraction_message(mycc, 'r4: W_oooovv * t2, W_oovvvv * t2'), *time1)
    log_memory(mycc, memlog, 'r4 W_oooovv/W_oovvvv contractions')

    t4_tmp = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    log_memory(mycc, memlog, 'r4 F_vv/W_vvvv T4 buffer allocated')
    time2 = logger.process_clock(), logger.perf_counter()
    for i, j, k, l, local_idx in dt4.iter_local_ijkl():
        r4_tmp = r4_local[local_idx]
        dt4.unpack_t4_single_local(t4_local, t4_tmp, i, j, k, l)
        einsum("ae,ebcd->abcd", F_vv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("be,aecd->abcd", F_vv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("ce,abed->abcd", F_vv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("de,abce->abcd", F_vv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)

        einsum("abef,efcd->abcd", W_vvvv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("acef,ebfd->abcd", W_vvvv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("adef,ebcf->abcd", W_vvvv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bcef,aefd->abcd", W_vvvv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("bdef,aecf->abcd", W_vvvv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)
        einsum("cdef,abef->abcd", W_vvvv, t4_tmp, out=r4_tmp, alpha=1.0, beta=1.0)

        time2 = log.timer_debug1(contraction_message(mycc, "r4: iter: F_vv * t4, W_vvvv * t4"
                                                    " [%2d, %2d, %2d, %2d]:" % (i, j, k, l)), *time2)
    F_vv = imds.F_vv = None
    W_vvvv = imds.W_vvvv = None
    t4_tmp = None
    time1 = log.timer_debug1(contraction_message(mycc, 'r4: F_vv * t4, W_vvvv * t4'), *time1)
    log_memory(mycc, memlog, 'r4 F_vv/W_vvvv contractions')

    compute_oooo_oovv_contraction_(mycc, imds, t4, r4_local)
    log_memory(mycc, memlog, 'r4 OO contractions')
    return r4_local

def compute_oooo_oovv_contraction_(mycc, imds, t4, r4_local):
    time1 = logger.process_clock(), logger.perf_counter()
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    dt4, t4_local = t4

    nocc4 = dt4.nocc4
    if nocc4 == 0:
        return r4_local

    F_oo, W_oooo, W_ovvo, W_ovov = imds.F_oo, imds.W_oooo, imds.W_ovvo, imds.W_ovov
    W_voov = np.ascontiguousarray(W_ovvo.transpose(1, 0, 3, 2))
    log_memory(mycc, memlog, 'r4 OO W_voov allocated')

    batch_size = nocc4 if mycc.batch_size is None else int(mycc.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch_size = min(batch_size, nocc4)

    nocc = mycc.nocc
    ijkl_list = np.array(dt4._enumerate_ijkl_quadruples(), dtype=np.int64)
    local_ijkl = tuple(dt4.iter_local_ijkl())

    t4_tmp_p3_v0 = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    t4_tmp_p3_v1 = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    t4_tmp_p3_v2 = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    t4_tmp_p3_v3 = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    t4_tmp_3 = np.empty((nvir,) * 4, dtype=t4_local.dtype)
    log_memory(mycc, memlog, 'r4 OO temp buffers allocated')

    time2 = logger.process_clock(), logger.perf_counter()

    # Build list of batch ranges
    batches = []
    for batch_start in range(0, len(ijkl_list), batch_size):
        batch_end = min(batch_start + batch_size, len(ijkl_list))
        batches.append((batch_start, batch_end))
    n_batches = len(batches)

    handle = None
    t4_data = None
    batch_ijkl_reordered = None
    pfs = None
    progress_thread = start_mpi_progress_thread(mycc) if n_batches > 0 else None
    try:
        if n_batches > 0 and mycc.size > 1:
            batch_start, batch_end = batches[0]
            batch_ijkl = ijkl_list[batch_start:batch_end]
            handle = dt4.prefetch_t4_quadruples_allgather(
                t4_local, batch_ijkl, batch_index=1,
                batch_start=batch_start, batch_end=batch_end)
            if progress_thread is not None and handle is not None:
                progress_thread.add_requests(handle.get('reqs', ()))

        for i_batch in range(n_batches):
            batch_start, batch_end = batches[i_batch]
            batch_ijkl = ijkl_list[batch_start:batch_end]

            # Prefetch next batch while we finalize current
            handle_next = None
            if i_batch + 1 < n_batches and mycc.size > 1:
                next_start, next_end = batches[i_batch + 1]
                next_batch_ijkl = ijkl_list[next_start:next_end]
                if mycc.size > 1:
                    handle_next = dt4.prefetch_t4_quadruples_allgather(
                        t4_local, next_batch_ijkl, batch_index=i_batch + 2,
                        batch_start=next_start, batch_end=next_end)
                    if progress_thread is not None and handle_next is not None:
                        progress_thread.add_requests(handle_next.get('reqs', ()))

            # Finalize current batch communication
            if mycc.size > 1 and handle is not None:
                if progress_thread is not None:
                    progress_thread.pause()
                try:
                    t4_data, batch_ijkl_reordered = dt4.finalize_prefetch_t4_quadruples(handle, t4_local, batch_ijkl)
                finally:
                    if progress_thread is not None:
                        if handle_next is not None:
                            progress_thread.set_requests(handle_next.get('reqs', ()))
                        else:
                            progress_thread.clear_requests()
                        progress_thread.resume()
            else:
                # Single-rank or handle is None: blocking collection.
                t4_data = t4_local[batch_start:batch_end]
                batch_ijkl_reordered = batch_ijkl

            # Compute F_oo[m, p] * T4[m, ...] and
            # W_oooo[m, n, p, q] * T4[r, s, m, n].
            pfs = np.zeros((len(_t4_transpose_axes), len(batch_ijkl_reordered), r4_local.shape[0]), dtype=t4_data.dtype)
            _accumulate_oo_t4_prefactors_(pfs, W_oooo, F_oo, batch_ijkl_reordered, local_ijkl)
            for slot, script in enumerate(_t4_transpose_einsums):
                if np.any(pfs[slot]):
                    einsum(script, pfs[slot], t4_data, out=r4_local, alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            gil_punctuate_interval = max(1, int(getattr(mycc, 'gil_punctuate_interval', 10)))
            for idx, (o00, o10, o20, o30) in enumerate(batch_ijkl_reordered):
                t4_tmp = t4_data[idx]
                t4_transpose_add_(t4_tmp, t4_tmp_3, nvir)
                t4_spin_summation_quadruple_sym_(t4_tmp, t4_tmp_p3_v0, t4_tmp_p3_v1, t4_tmp_p3_v2, t4_tmp_p3_v3, nvir)

                if idx % gil_punctuate_interval == 0:
                    punctuate_mpi_progress(mycc, progress_thread)

                for i, j, k, l, local_idx in dt4.iter_local_ijkl():
                    r4_tmp = r4_local[local_idx]

                    # j k l
                    # case 0: m <= j <= k <= l: m=o00, j=o10, k=o20, l=o30
                    if j == o10 and k == o20 and l == o30:
                        einsum("be,aecd->abcd", W_ovov[o00, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,abed->abcd", W_ovov[o00, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,abce->abcd", W_ovov[o00, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ae,ebcd->abcd", W_voov[:, o00, i, :], t4_tmp_p3_v3, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ae,becd->abcd", W_ovov[o00, :, i, :], t4_tmp_3, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 1: j < m <= k <= l: m=o10, j=o00, k=o20, l=o30
                    if o00 != o10 and j == o00 and k == o20 and l == o30:
                        einsum("be,eacd->abcd", W_ovov[o10, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,baed->abcd", W_ovov[o10, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,bace->abcd", W_ovov[o10, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ae,becd->abcd", W_voov[:, o10, i, :], t4_tmp_p3_v2, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ae,ebcd->abcd", W_ovov[o10, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ae,bced->abcd", W_ovov[o10, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ae,bdce->abcd", W_ovov[o10, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 2: j <= k < m <= l: m=o20, j=o00, k=o10, l=o30
                    if o10 != o20 and j == o00 and k == o10 and l == o30:
                        einsum("be,ecad->abcd", W_ovov[o20, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,bead->abcd", W_ovov[o20, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,bcae->abcd", W_ovov[o20, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ae,bced->abcd", W_voov[:, o20, i, :], t4_tmp_p3_v1, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ae,ecbd->abcd", W_ovov[o20, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ae,becd->abcd", W_ovov[o20, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ae,bcde->abcd", W_ovov[o20, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 3: j <= k <= l < m: m=o30, j=o00, k=o10, l=o20
                    if o20 != o30 and j == o00 and k == o10 and l == o20:
                        einsum("be,ecda->abcd", W_ovov[o30, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,beda->abcd", W_ovov[o30, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,bcea->abcd", W_ovov[o30, :, i, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ae,bcde->abcd", W_voov[:, o30, i, :], t4_tmp_p3_v0, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ae,ecdb->abcd", W_ovov[o30, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ae,bedc->abcd", W_ovov[o30, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ae,bced->abcd", W_ovov[o30, :, i, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)

                    # i k l
                    # case 0: m <= i <= k <= l: m=o00, i=o10, k=o20, l=o30
                    if i == o10 and k == o20 and l == o30:
                        einsum("ae,becd->abcd", W_ovov[o00, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,baed->abcd", W_ovov[o00, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,bace->abcd", W_ovov[o00, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,eacd->abcd", W_voov[:, o00, j, :], t4_tmp_p3_v3, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("be,aecd->abcd", W_ovov[o00, :, j, :], t4_tmp_3, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 1: i < m <= k <= l: m=o10, i=o00, k=o20, l=o30
                    if o00 != o10 and i == o00 and k == o20 and l == o30:
                        einsum("ae,ebcd->abcd", W_ovov[o10, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,abed->abcd", W_ovov[o10, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,abce->abcd", W_ovov[o10, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aecd->abcd", W_voov[:, o10, j, :], t4_tmp_p3_v2, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("be,eacd->abcd", W_ovov[o10, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("be,aced->abcd", W_ovov[o10, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("be,adce->abcd", W_ovov[o10, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 2: i <= k < m <= l: m=o20, i=o00, k=o10, l=o30
                    if o10 != o20 and i == o00 and k == o10 and l == o30:
                        einsum("ae,ecbd->abcd", W_ovov[o20, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,aebd->abcd", W_ovov[o20, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,acbe->abcd", W_ovov[o20, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aced->abcd", W_voov[:, o20, j, :], t4_tmp_p3_v1, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("be,ecad->abcd", W_ovov[o20, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("be,aecd->abcd", W_ovov[o20, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("be,acde->abcd", W_ovov[o20, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 3: i <= k <= l < m: m=o30, i=o00, k=o10, l=o20
                    if o20 != o30 and i == o00 and k == o10 and l == o20:
                        einsum("ae,ecdb->abcd", W_ovov[o30, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,aedb->abcd", W_ovov[o30, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,aceb->abcd", W_ovov[o30, :, j, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,acde->abcd", W_voov[:, o30, j, :], t4_tmp_p3_v0, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("be,ecda->abcd", W_ovov[o30, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("be,aedc->abcd", W_ovov[o30, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("be,aced->abcd", W_ovov[o30, :, j, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)

                    # i j l
                    # case 0: m <= i <= j <= l: m=o00, i=o10, j=o20, l=o30
                    if i == o10 and j == o20 and l == o30:
                        einsum("ae,cebd->abcd", W_ovov[o00, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,caed->abcd", W_ovov[o00, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,cabe->abcd", W_ovov[o00, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,eabd->abcd", W_voov[:, o00, k, :], t4_tmp_p3_v3, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ce,aebd->abcd", W_ovov[o00, :, k, :], t4_tmp_3, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 1: i < m <= j <= l: m=o10, i=o00, j=o20, l=o30
                    if o00 != o10 and i == o00 and j == o20 and l == o30:
                        einsum("ae,ecbd->abcd", W_ovov[o10, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aced->abcd", W_ovov[o10, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,acbe->abcd", W_ovov[o10, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,aebd->abcd", W_voov[:, o10, k, :], t4_tmp_p3_v2, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ce,eabd->abcd", W_ovov[o10, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ce,abed->abcd", W_ovov[o10, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ce,adbe->abcd", W_ovov[o10, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 2: i <= j < m <= l: m=o20, i=o00, j=o10, l=o30
                    if o10 != o20 and i == o00 and j == o10 and l == o30:
                        einsum("ae,ebcd->abcd", W_ovov[o20, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aecd->abcd", W_ovov[o20, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,abce->abcd", W_ovov[o20, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,abed->abcd", W_voov[:, o20, k, :], t4_tmp_p3_v1, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ce,ebad->abcd", W_ovov[o20, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ce,aebd->abcd", W_ovov[o20, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ce,abde->abcd", W_ovov[o20, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 3: i <= j <= l < m: m=o30, i=o00, j=o10, l=o20
                    if o20 != o30 and i == o00 and j == o10 and l == o20:
                        einsum("ae,ebdc->abcd", W_ovov[o30, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aedc->abcd", W_ovov[o30, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,abec->abcd", W_ovov[o30, :, k, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,abde->abcd", W_voov[:, o30, k, :], t4_tmp_p3_v0, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("ce,ebda->abcd", W_ovov[o30, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ce,aedb->abcd", W_ovov[o30, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("ce,abed->abcd", W_ovov[o30, :, k, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)

                    # i j k
                    # case 0: m <= i <= j <= k: m=o00, i=o10, j=o20, k=o30
                    if i == o10 and j == o20 and k == o30:
                        einsum("ae,debc->abcd", W_ovov[o00, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,daec->abcd", W_ovov[o00, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,dabe->abcd", W_ovov[o00, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,eabc->abcd", W_voov[:, o00, l, :], t4_tmp_p3_v3, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("de,aebc->abcd", W_ovov[o00, :, l, :], t4_tmp_3, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 1: i < m <= j <= k: m=o10, i=o00, j=o20, k=o30
                    if o00 != o10 and i == o00 and j == o20 and k == o30:
                        einsum("ae,edbc->abcd", W_ovov[o10, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,adec->abcd", W_ovov[o10, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,adbe->abcd", W_ovov[o10, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,aebc->abcd", W_voov[:, o10, l, :], t4_tmp_p3_v2, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("de,eabc->abcd", W_ovov[o10, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("de,abec->abcd", W_ovov[o10, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("de,acbe->abcd", W_ovov[o10, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 2: i <= j < m <= k: m=o20, i=o00, j=o10, k=o30
                    if o10 != o20 and i == o00 and j == o10 and k == o30:
                        einsum("ae,ebdc->abcd", W_ovov[o20, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aedc->abcd", W_ovov[o20, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,abde->abcd", W_ovov[o20, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,abec->abcd", W_voov[:, o20, l, :], t4_tmp_p3_v1, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("de,ebac->abcd", W_ovov[o20, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("de,aebc->abcd", W_ovov[o20, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("de,abce->abcd", W_ovov[o20, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                    # case 3: i <= j <= k < m: m=o30, i=o00, j=o10, k=o20
                    if o20 != o30 and i == o00 and j == o10 and k == o20:
                        einsum("ae,ebcd->abcd", W_ovov[o30, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("be,aecd->abcd", W_ovov[o30, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("ce,abed->abcd", W_ovov[o30, :, l, :], t4_tmp, out=r4_tmp, alpha=-1.0, beta=1.0)
                        einsum("de,abce->abcd", W_voov[:, o30, l, :], t4_tmp_p3_v0, out=r4_tmp, alpha=0.5, beta=1.0)
                        einsum("de,ebca->abcd", W_ovov[o30, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("de,aecb->abcd", W_ovov[o30, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)
                        einsum("de,abec->abcd", W_ovov[o30, :, l, :], t4_tmp, out=r4_tmp, alpha=-0.5, beta=1.0)

            time2 = log.timer_debug1(contraction_message(mycc, "r4: iter: F_oo * t4, W_oooo * t4, " \
                                        "W_ovov/W_voov * t4 batch %4d:%4d:" % (batch_start, batch_end)), *time2)
            log_memory(mycc, memlog, 'r4 OO batch %4d:%4d' % (batch_start, batch_end), per_iter=True)

            handle = handle_next
    finally:
        handle = None
        if progress_thread is not None:
            progress_thread.stop()

    F_oo = imds.F_oo = None
    W_oooo = imds.W_oooo = None
    W_ovvo = imds.W_ovvo = None
    W_ovov = imds.W_ovov = None
    W_voov = None
    pfs = None
    t4_data = None
    batch_ijkl_reordered = None
    t4_tmp, t4_tmp_p3_v0, t4_tmp_p3_v1, t4_tmp_p3_v2, t4_tmp_p3_v3, t4_tmp_3 = None, None, None, None, None, None
    r4_tmp = None

    ijkl_list = None
    local_ijkl = None
    batches = None
    log_memory(mycc, memlog, 'r4 OO buffers released')

    time1 = log.timer_debug1(contraction_message(mycc, 'r4: F_oo * t4, W_oooo * t4, W_ovov/W_voov * t4'), *time1)
    return r4_local

def update_amps_rccsdtq_tri_(mycc, tamps, eris):
    '''Update RCCSDTQ amplitudes in place, with T4 amplitudes stored in triangular form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose if mycc.rank == 0 else 0)
    memlog = memory_logger(mycc)
    log_memory(mycc, memlog, 'update_amps start')

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3, t4 = tamps
    mo_energy = eris.mo_energy

    imds = _IMDS()

    # t1, t2
    update_t1_fock_eris(mycc, imds, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    log_memory(mycc, memlog, 'update fock and eris')
    intermediates_t1t2(mycc, imds, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    log_memory(mycc, memlog, 't1t2 update intermediates')
    r1, r2 = compute_r1r2(mycc, imds, t2)
    r1r2_add_t3_(mycc, imds, r1, r2, t3)
    r2_add_t4_tri_(mycc, imds, r2, t4)

    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    log_memory(mycc, memlog, 't1t2 compute r1 and r2')
    # symmetrization
    r2 += r2.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t1t2: symmetrize r2', *time1)
    log_memory(mycc, memlog, 't1t2 symmetrize r2')
    # divide by eijkabc
    r1r2_divide_e_(mycc, r1, r2, mo_energy)
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)
    log_memory(mycc, memlog, 't1t2 divide r1 and r2')

    res_norm = [np.linalg.norm(r1), np.linalg.norm(r2)]

    t1 += r1
    t2 += r2
    r1 = None
    r2 = None
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)
    log_memory(mycc, memlog, 't1t2 update t1 and t2')

    # t3
    intermediates_t3(mycc, imds, t2)
    intermediates_t3_add_t3(mycc, imds, t3)
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    log_memory(mycc, memlog, 't3 update intermediates')
    r3 = compute_r3(mycc, imds, t2, t3)
    r3_add_t4_tri_(mycc, imds, r3, t4)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    log_memory(mycc, memlog, 't3 compute r3')
    # symmetrization
    t3_perm_symmetrize_inplace_(r3, nocc, nvir, 1.0, 0.0)
    t3_spin_summation_inplace_(r3, nocc**3, nvir, "P3_full", -1.0 / 6.0, 1.0)
    purify_tamps_(r3)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    log_memory(mycc, memlog, 't3 symmetrize r3')
    # divide by eijkabc
    r3_divide_e_(mycc, r3, mo_energy)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)
    log_memory(mycc, memlog, 't3 divide r3')

    res_norm.append(np.linalg.norm(r3))

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)
    log_memory(mycc, memlog, 't3 update t3')

    # t4
    intermediates_t4_tri(mycc, imds, t2, t3, t4)
    imds.t1_fock = None
    time1 = log.timer_debug1('t4: update intermediates', *time0)
    log_memory(mycc, memlog, 't4 update intermediates')
    r4_local = compute_r4_tri(mycc, imds, t2, t3, t4)
    imds.t1_eris = None
    imds = None
    time1 = log.timer_debug1('t4: compute r4', *time1)
    log_memory(mycc, memlog, 't4 compute r4')
    # symmetrization
    # symmetrize_tamps_tri_(r4, nocc)
    # orthogonalization
    t4_project_1_minus_p4_p31_inplace_(r4_local, r4_local.shape[0], nvir)
    # purification
    # purify_tamps_tri_(r4, nocc)
    time1 = log.timer_debug1('t4: symmetrize r4', *time1)
    log_memory(mycc, memlog, 't4 symmetrize r4')
    # divide by eijkabc
    r4_local_tri_divide_e_(mycc, t4[0], r4_local, mo_energy)
    time1 = log.timer_debug1('t4: divide r4 by eijklabcd', *time1)
    log_memory(mycc, memlog, 't4 divide r4')

    r4_norm_sq_local = np.linalg.norm(r4_local)**2
    nvir_t4_diis = resolve_nvir_diis(mycc, nvir)
    if nvir_t4_diis < nvir:
        r4_active = r4_local[:, :nvir_t4_diis, :nvir_t4_diis, :nvir_t4_diis, :nvir_t4_diis]
        r4_active_norm_sq_local = np.linalg.norm(r4_active)**2
    else:
        r4_active_norm_sq_local = r4_norm_sq_local
    r4_norms = np.array([r4_norm_sq_local, r4_active_norm_sq_local], dtype=np.float64)
    mycc.comm.Allreduce(MPI.IN_PLACE, r4_norms, op=MPI.SUM)
    r4_norm_sq = float(r4_norms[0])
    mycc._last_t4_residual_norm_sq = r4_norm_sq
    mycc._last_t4_active_residual_norm_sq = float(r4_norms[1])
    res_norm.append(np.sqrt(r4_norm_sq))

    t4_add_(t4[1], r4_local, r4_local.shape[0], nvir)
    r4_local = None
    log_memory(mycc, memlog, 't4 update t4')

    if mycc.rank == 0:
        time1 = log.timer_debug1('t4: update t4', *time1)
        time0 = log.timer_debug1('t4 total', *time0)

    return res_norm

def _safe_update_norm(norm_sq):
    return np.sqrt(max(float(norm_sq), 0.0))

def _standard_diis_update_norm(res_norm, lower_count, lower_delta_norm_sq, normt):
    if lower_delta_norm_sq is None:
        if res_norm is None:
            return float(normt)
        lower_delta_norm_sq = norm_sq_from_norms(res_norm[:lower_count])
    inactive_norm_sq = 0.0 if res_norm is None else norm_sq_from_norms(res_norm[lower_count:])
    return _safe_update_norm(lower_delta_norm_sq + inactive_norm_sq)

def _max_t_diis_update_norm(mycc, res_norm, normt, adiis, nvir_diis, nvir, total_attr, active_attr):
    delta_norm_sq = getattr(adiis, 'last_delta_norm_sq', None)
    if delta_norm_sq is None:
        return float(normt) if res_norm is None else _safe_update_norm(norm_sq_from_norms(res_norm))
    delta_norm_sq = float(delta_norm_sq)
    if nvir_diis < nvir:
        total_norm_sq = float(getattr(mycc, total_attr, res_norm[-1] * res_norm[-1]))
        active_norm_sq = float(getattr(mycc, active_attr, total_norm_sq))
        delta_norm_sq += max(total_norm_sq - active_norm_sq, 0.0)
    return _safe_update_norm(delta_norm_sq)

def _call_standard_diis(mycc, tamps, istep, normt, de, adiis, res_norm):
    try:
        return mycc.run_diis(tamps, istep, normt, de, adiis, res_norm=res_norm, return_update_norm=True)
    except TypeError as err:
        try:
            old_style_tamps = mycc.run_diis(tamps, istep, normt, de, adiis)
        except TypeError:
            raise err
        return old_style_tamps, _safe_update_norm(norm_sq_from_norms(res_norm))

def run_diis(mycc, tamps, istep, normt, de, adiis, res_norm=None, return_update_norm=False):
    lower_delta_norm_sq = None
    if (adiis and istep >= mycc.diis_start_cycle and abs(de) < mycc.diis_start_energy_diff):
        vector = mycc.amplitudes_to_vector(tamps)
        old_vector = getattr(adiis, '_xprev', None)
        new_vector = adiis.update(vector)
        if old_vector is not None:
            lower_delta_norm_sq = vector_delta_norm_sq(new_vector, old_vector)
        tamps = mycc.vector_to_amplitudes(new_vector)
        if mycc.rank == 0:
            logger.debug1(mycc, 'DIIS for step %d', istep)
    if return_update_norm:
        return tamps, _standard_diis_update_norm(res_norm, len(tamps), lower_delta_norm_sq, normt)
    return tamps

def run_diis_t4(mycc, tamps, istep, normt, de, adiis_t4, res_norm=None, return_update_norm=False):
    if not (adiis_t4 and istep >= mycc.diis_start_cycle and abs(de) < mycc.diis_start_energy_diff):
        if return_update_norm:
            update_norm = float(normt) if res_norm is None else _safe_update_norm(norm_sq_from_norms(res_norm))
            return tamps, update_norm
        return tamps

    t1, t2, t3, t4 = tamps[:4]
    dt4, t4_local = t4
    # T1/T2/T3 are replicated on every rank; scale them so the MPI-reduced
    # DIIS dot product counts each replicated block once.
    scale = 1.0 / np.sqrt(mycc.size)
    nvir = t4_local.shape[1]
    nvir_t4_diis = resolve_nvir_diis(mycc, nvir)
    if nvir_t4_diis < nvir:
        t4_diis = t4_local[:, :nvir_t4_diis, :nvir_t4_diis, :nvir_t4_diis, :nvir_t4_diis]
    else:
        t4_diis = t4_local
    t4_diis_shape = t4_diis.shape
    lower_vector = pack_unique_replicated_tamps(mycc, (t1, t2, t3), scale=scale)
    vector = np.concatenate((lower_vector, t4_diis.ravel()))

    new_vector = adiis_t4.update(vector)

    (t1_new, t2_new, t3_new), offset = unpack_unique_replicated_tamps(mycc, new_vector, (t1, t2, t3), scale=scale)
    tamps[0] = t1_new
    tamps[1] = t2_new
    tamps[2] = t3_new

    if nvir_t4_diis < nvir:
        t4_local[:, :nvir_t4_diis, :nvir_t4_diis, :nvir_t4_diis, :nvir_t4_diis] = (
            new_vector[offset:].reshape(t4_diis_shape))
    else:
        t4_local[...] = new_vector[offset:].reshape(t4_diis_shape)
    tamps[3] = (dt4, t4_local)

    update_norm = None
    if return_update_norm:
        update_norm = _max_t_diis_update_norm(mycc, res_norm, normt, adiis_t4, nvir_t4_diis, nvir,
                                            '_last_t4_residual_norm_sq', '_last_t4_active_residual_norm_sq')

    if mycc.rank == 0:
        logger.debug1(mycc, 'Unified DIIS for T1/T2/T3/T4 amplitudes at step %d', istep)

    if return_update_norm:
        return tamps, update_norm
    return tamps

def kernel(mycc, eris=None, tamps=None, tol=1e-8, tolnormt=1e-6, max_cycle=50, verbose=5, callback=None):

    log = logger.Logger(mycc.stdout, verbose if mycc.rank == 0 else 0)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    if tamps is None:
        _, tamps = mycc.init_amps(eris)
    else:
        if len(tamps) < mycc.cc_order:
            _, init_tamps = mycc.init_amps(eris)
            tamps = list(tamps) + list(init_tamps[len(tamps):])
        else:
            tamps = list(tamps)
            if isinstance(tamps[3], tuple):
                dt4, t4_local = tamps[3]
                t4_local = np.ascontiguousarray(t4_local)
            else:
                dt4 = DistributedT4IJKL(mycc.nocc, mycc.nmo - mycc.nocc, comm=mycc.comm, batch_size=mycc.batch_size,
                                        dtype=tamps[0].dtype, allow_python_fallback=mycc.allow_python_fallback)
                t4_local = np.ascontiguousarray(tamps[3])
            dt4.log = logger.new_logger(mycc)
            tamps[3] = (dt4, t4_local)
    configure_t4_runtime_logging(mycc, tamps)

    name = mycc.__class__.__name__

    if mycc.rank == 0:
        cput1 = cput0 = (logger.process_clock(), logger.perf_counter())

    e_corr_old = 0.0
    e_corr = mycc.energy(tamps, eris)

    if mycc.rank == 0:
        log.info('Init E_corr(%s) = %.15g', name, e_corr)

    adiis = None
    adiis_t4 = None
    if mycc.do_diis_max_t:
        adiis_t4 = make_mpi_diis(mycc, log=log)
    else:
        adiis = make_standard_diis(mycc)

    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        res_norm = mycc.update_amps_(tamps, eris)

        if callback is not None:
            callback(locals())

        normt = np.linalg.norm(res_norm)

        if mycc.iterative_damping < 1.0:
            raise NotImplementedError("Damping is not implemented")

        if mycc.do_diis_max_t:
            tamps, update_norm = mycc.run_diis_t4(tamps, istep, normt, e_corr - e_corr_old, adiis_t4,
                                                res_norm=res_norm, return_update_norm=True)
        else:
            lower_tamps, update_norm = _call_standard_diis(
                mycc, tamps[:mycc.cc_order - 1], istep, normt, e_corr - e_corr_old, adiis, res_norm)
            tamps[:mycc.cc_order - 1] = lower_tamps

        e_corr_old, e_corr = e_corr, mycc.energy(tamps, eris)
        mycc.e_corr_ss = getattr(e_corr, 'e_corr_ss', 0)
        mycc.e_corr_os = getattr(e_corr, 'e_corr_os', 0)

        mycc.cycles = istep + 1

        # NOTE: for consistency, broadcast e_corr and norms to all processes
        e_corr = mycc.comm.bcast(e_corr, root=0)
        normt = mycc.comm.bcast(normt, root=0)
        update_norm = mycc.comm.bcast(update_norm, root=0)

        if mycc.rank == 0:
            log.info("cycle = %2d  E_corr(%s) = % .12f  dE = % .12e  norm(res) = %.8e  norm(d tamps) = %.8e" % (
                istep + 1, mycc.__class__.__name__, e_corr, e_corr - e_corr_old, normt, update_norm))
            cput1 = log.timer(f'{name} iter', *cput1)

        if abs(e_corr - e_corr_old) < tol and normt < tolnormt:
            converged = True
            break

    if mycc.rank == 0:
        log.timer(name, *cput0)

    if adiis_t4 is not None and getattr(adiis_t4, 'cleanup', False):
        adiis_t4.clean_scratch()

    close_t4_runtime_logging(tamps)
    return converged, e_corr, tamps


class RCCSDTQ(RCCSDT):

    cc_order = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_cc_order', 4)

    @property
    def t4(self):
        return self.tamps[3]

    @t4.setter
    def t4(self, val):
        self.tamps[3] = val

    def __init__(self, mf, comm=None, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, comm=comm, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)

    memory_estimate_log = memory_estimate_log_mpi_rccsdtq
    update_amps_ = update_amps_rccsdtq_tri_
    run_diis = run_diis
    run_diis_t4 = run_diis_t4
    init_amps = init_amps_rhf

    def kernel(self, tamps=None, eris=None):
        return self.ccsdtq(tamps, eris)

    def ccsdtq(self, tamps=None, eris=None):
        if self.rank != 0:
            self.verbose = 0
        log = logger.Logger(self.stdout, self.verbose if self.rank == 0 else 0)

        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        assert self.mo_coeff.dtype == np.float64, "`mo_coeff` must be float64"

        if self.rank == 0 and self.verbose >= logger.WARN:
            self.check_sanity()
            self.dump_flags()
        warn_non_pytblis_backend(self, self.__class__.__name__)

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        if self.rank == 0:
            self.memory_estimate_log()
        self.unique_tamps_map = self.build_unique_tamps_map()

        self.converged, self.e_corr, self.tamps = kernel(self, eris, tamps, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt, verbose=self.verbose, callback=self.callback)

        if self.rank == 0:
            self._finalize()

        return self.e_corr, self.tamps


if __name__ == "__main__":

    from pyscf import gto, scf
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", verbose=3 if rank == 0 else 0)
    mf = scf.RHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()

    ref_ecorr = -0.157579406507473 # 321g
    # ref_ecorr = -0.2375471949129775 # 631g
    # ref_ecorr = -0.3271440688237135 # ccpvdz
    frozen = 0
    mycc = RCCSDTQ(mf, comm=comm, frozen=frozen)
    # mycc.set_einsum_backend('pytblis')
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 100
    mycc.verbose = 8

    mycc.batch_size = 171

    mycc.diis = True
    mycc.do_diis_max_t = True
    mycc.nvir_diis = 4
    mycc.incore_complete = True
    mycc.diis_scratch = None
    mycc.diis_scratch_start = 0
    mycc.diis_scratch_cleanup = True
    mycc.diis_scratch_mmap = False

    mycc.log_highest_t_communication = True
    mycc.log_highest_t_contractions = True
    mycc.log_highest_t_contractions_all_ranks = False
    mycc.log_memory = True
    mycc.log_memory_all_ranks = False
    mycc.log_memory_per_iter = False

    mycc.use_mpi_progress_thread = False
    mycc.mpi_progress_poll_interval = 0.001
    mycc.gil_punctuate_duration = 0.0001
    mycc.gil_punctuate_interval = 10

    mycc.kernel()
    if mycc.rank == 0:
        print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_ecorr, mycc.e_corr - ref_ecorr))
        print()
