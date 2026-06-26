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
import functools
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf
from pyscf.cc import ccsd
from pyscf.cc.rccsdt import (_einsum, t3_spin_summation_inplace_, update_t1_fock_eris, energy_rhf, intermediates_t1t2,
                            compute_r1r2, r1r2_divide_e_, intermediates_t3, amplitudes_to_vector_rhf,
                            vector_to_amplitudes_rhf, build_unique_tamps_map_rhf, _ao2mo_rcc,
                            format_size, dump_flags as _pyscf_rccsdt_dump_flags, _finalize,
                            _IMDS, _PhysicistsERIs)
from pyscf import __config__
from mpi4py import MPI

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distr_cc._c_rccsdt import t3_single_spin_summation_inplace_, t3_spin_summation_triple_sym_, t3_transpose_add_
from distr_cc._mpi import punctuate_mpi_progress, start_mpi_progress_thread
from distr_cc._runtime import (contraction_logger, contraction_message, log_memory, make_mpi_diis, make_standard_diis,
                            memory_logger, norm_sq_from_norms, pack_unique_replicated_tamps,
                            unpack_unique_replicated_tamps, vector_delta_norm_sq, resolve_diis_incore_space,
                            resolve_diis_scratch, resolve_nvir_diis, warn_non_pytblis_backend)
from distr_cc.distribute_t3 import DistributedT3IJK

_MPI_INT_MAX = np.iinfo(np.intc).max
_MPI_ALLREDUCE_MAX_BYTES = int(getattr(
    __config__, 'cc_mpi_rccsdt_allreduce_max_bytes', 1 << 30))


def _report_allreduce_timing(comm, log, label, arr, elapsed, nchunks, max_count):
    if log is None:
        return

    elapsed_max = comm.allreduce(elapsed, op=MPI.MAX)
    if comm.Get_rank() != 0:
        return

    label = label or 'buffer'
    log.info('allreduce %s: %.4f sec max over %d ranks, buffer %s, chunks %d, chunk <= %s',
             label, elapsed_max, comm.Get_size(), format_size(arr.nbytes),
             nchunks, format_size(max_count * arr.dtype.itemsize))


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


def _allreduce_inplace_mpi_rccsdt(mycc, buf, label, op=MPI.SUM):
    log_timing = getattr(mycc, 'log_allreduce_timing', False)
    log = None
    if log_timing:
        log = logger.Logger(mycc.stdout, logger.INFO if mycc.rank == 0 else 0)
    return _allreduce_inplace_large(mycc.comm, buf, op=op, log=log, label=label, log_timing=log_timing)


def configure_t3_runtime_logging(mycc, tamps):
    if tamps is None or len(tamps) < 3:
        return
    t3 = tamps[2]
    if not isinstance(t3, (tuple, list)) or len(t3) < 1:
        return
    dt3 = t3[0]
    enabled = getattr(mycc, 'log_highest_t_communication', False)
    log_dir = getattr(mycc, 'communication_log_dir', 'comm_logs')
    if hasattr(dt3, 'configure_communication_logging'):
        dt3.configure_communication_logging(enabled=enabled, log_dir=log_dir)
    else:
        dt3.log_t3_communication = bool(enabled)
        dt3.communication_log_dir = log_dir

def close_t3_runtime_logging(tamps):
    if tamps is None or len(tamps) < 3:
        return
    t3 = tamps[2]
    if not isinstance(t3, (tuple, list)) or len(t3) < 1:
        return
    dt3 = t3[0]
    if hasattr(dt3, 'close_communication_log'):
        dt3.close_communication_log()


def _format_dump_value(value):
    if isinstance(value, str):
        return value
    return repr(value)


def _dump_flag_group(log, title, mycc, names):
    log.info('%s', title)
    for name in names:
        if hasattr(mycc, name):
            log.info('    %-36s = %s', name, _format_dump_value(getattr(mycc, name)))


def dump_flags(mycc, verbose=None):
    '''Print PySCF RCCSDT flags plus Distributed-CC-specific options.'''
    if getattr(mycc, 'rank', 0) != 0:
        return mycc

    _pyscf_rccsdt_dump_flags(mycc, verbose)
    log = logger.new_logger(mycc, verbose)
    log.info('')
    log.info('Distributed-CC options')
    log.info('    %-36s = %s', 'MPI ranks', getattr(mycc, 'size', 1))
    _dump_flag_group(log, '    Work distribution', mycc, (
        'batch_size',
        'allow_python_fallback',
    ))
    _dump_flag_group(log, '    DIIS', mycc, (
        'nvir_diis',
        'diis_scratch',
        'diis_scratch_start',
        'diis_scratch_cleanup',
        'diis_scratch_mmap',
        'incore_complete',
    ))
    _dump_flag_group(log, '    Diagnostics', mycc, (
        'log_memory',
        'log_memory_per_iter',
        'log_memory_all_ranks',
        'log_highest_t_contractions',
        'log_highest_t_contractions_all_ranks',
        'contraction_log_dir',
        'log_highest_t_communication',
        'communication_log_dir',
        'log_allreduce_timing',
    ))
    _dump_flag_group(log, '    MPI progress', mycc, (
        'use_mpi_progress_thread',
        'mpi_progress_poll_interval',
        'gil_punctuate_duration',
        'gil_punctuate_interval',
    ))
    return mycc


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
    eia = mo_e[: nocc, None] - mo_e[None, nocc :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    t1 = eris.fock[:nocc, nocc:] / eia
    t2 = eris.pppp[:nocc, :nocc, nocc:, nocc:] / eijab

    tau = t2 + einsum("ia,jb->ijab", t1, t1)
    e_corr = 2.0 * einsum("ijab,ijab->", eris.pppp[:nocc, :nocc, nocc:, nocc:], tau)
    e_corr -= einsum("ijba,ijab->", eris.pppp[:nocc, :nocc, nocc:, nocc:], tau)
    e_corr += 2.0 * einsum("ai,ia->", eris.fock[nocc:, :nocc], t1)

    if mycc.rank == 0:
        logger.info(mycc, "Init t2, MP2 energy = % .12f  E_corr(MP2) % .12f" % (e_hf + e_corr, e_corr))

    cc_order = mycc.cc_order
    if mycc.cc_order > 3:
        raise NotImplementedError("Only CCSDT (cc_order=3) is implemented in mpi_rccsdt.py")

    tamps = [t1, t2]
    for order in range(2, cc_order - 1):
        t = np.zeros((nocc,) * (order + 1) + (nvir,) * (order + 1), dtype=t1.dtype)
        tamps.append(t)
    if mycc.do_tri_max_t:
        # t3 amplitude is distributed across MPI ranks
        dt3 = DistributedT3IJK(nocc, nvir, comm=mycc.comm, batch_size=mycc.batch_size, dtype=t1.dtype,
                              allow_python_fallback=mycc.allow_python_fallback)
        dt3.log = logger.new_logger(mycc)
        dt3.configure_communication_logging(enabled=mycc.log_highest_t_communication,
                                            log_dir=mycc.communication_log_dir)
        dt3.print_distribution_info()
        t3_local = dt3.allocate_local()
        t3 = (dt3, t3_local)
    else:
        raise NotImplementedError("Only tri-stored T3 amplitudes are implemented in mpi_rccsdt.py")
    tamps.append(t3)

    if mycc.rank == 0:
        logger.timer(mycc, 'init mp2', *time0)

    return e_corr, tamps

def r1r2_add_t3_tri_(mycc, imds, r1, r2, t3):
    '''Add the T3 contributions to r1 and r2. T3 amplitudes are stored in triangular form.
    MPI communication is performed to gather the contributions from distributed T3 amplitudes.
    '''
    einsum = functools.partial(_einsum, mycc.einsum_backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    dt3, t3_local = t3

    r1_loc = np.zeros_like(r1)
    r2_loc = np.zeros_like(r2)

    t3_tmp = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    for i, j, k, _ in dt3.iter_local_ijk():
        dt3.take_t3_single_local(t3_local, t3_tmp, i, j, k)

        t3_single_spin_summation_inplace_(t3_tmp, nvir, "P3_422", 1.0, 0.0)
        if i < j and j < k:
            einsum('bc,abc->a', t1_eris[j, k, nocc:, nocc:], t3_tmp, out=r1_loc[i, :], alpha=0.5, beta=1.0)
            einsum('cb,acb->a', t1_eris[j, k, nocc:, nocc:], t3_tmp, out=r1_loc[i, :], alpha=0.5, beta=1.0)
            einsum('bc,bac->a', t1_eris[i, k, nocc:, nocc:], t3_tmp, out=r1_loc[j, :], alpha=0.5, beta=1.0)
            einsum('cb,cab->a', t1_eris[i, k, nocc:, nocc:], t3_tmp, out=r1_loc[j, :], alpha=0.5, beta=1.0)
            einsum('bc,bca->a', t1_eris[i, j, nocc:, nocc:], t3_tmp, out=r1_loc[k, :], alpha=0.5, beta=1.0)
            einsum('cb,cba->a', t1_eris[i, j, nocc:, nocc:], t3_tmp, out=r1_loc[k, :], alpha=0.5, beta=1.0)
        elif i == j and j < k:
            einsum('bc,abc->a', t1_eris[j, k, nocc:, nocc:], t3_tmp, out=r1_loc[i, :], alpha=0.5, beta=1.0)
            einsum('cb,acb->a', t1_eris[j, k, nocc:, nocc:], t3_tmp, out=r1_loc[i, :], alpha=0.5, beta=1.0)
            einsum('bc,bca->a', t1_eris[i, j, nocc:, nocc:], t3_tmp, out=r1_loc[k, :], alpha=0.5, beta=1.0)
        elif i < j and j == k:
            einsum('bc,abc->a', t1_eris[j, k, nocc:, nocc:], t3_tmp, out=r1_loc[i, :], alpha=0.5, beta=1.0)
            einsum('bc,bac->a', t1_eris[i, k, nocc:, nocc:], t3_tmp, out=r1_loc[j, :], alpha=0.5, beta=1.0)
            einsum('cb,cab->a', t1_eris[i, k, nocc:, nocc:], t3_tmp, out=r1_loc[j, :], alpha=0.5, beta=1.0)
        else:
            continue
    t3_tmp = None

    r1_loc = np.ascontiguousarray(r1_loc)
    _allreduce_inplace_mpi_rccsdt(mycc, r1_loc, 'r1_loc_t3', op=MPI.SUM)
    r1 += r1_loc

    t3_tmp = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    for i, j, k, _ in dt3.iter_local_ijk():
        if i < j < k:
            perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
        elif i == j < k:
            perms = [(i, j, k), (i, k, j), (k, i, j)]
        elif i < j == k:
            perms = [(i, j, k), (j, i, k), (j, k, i)]
        elif i == j == k:
            perms = [(i, j, k)]

        # NOTE: This part can be further optimized by the symmetry of t3_tmp
        for (pi, pj, pk) in perms:
            dt3.take_t3_single_local(t3_local, t3_tmp, pi, pj, pk)
            t3_single_spin_summation_inplace_(t3_tmp, nvir, "P3_201", 1.0, 0.0)
            einsum("a,abc->bc", t1_fock[pi, nocc:], t3_tmp, out=r2_loc[pj, pk], alpha=0.5, beta=1.0)
            einsum("cad,dba->bc", t1_eris[nocc:, pi, nocc:, nocc:], t3_tmp, out=r2_loc[pj, pk], alpha=1.0, beta=1.0)
            einsum("la,abc->lbc", t1_eris[pk, pi, :nocc, nocc:], t3_tmp, out=r2_loc[pj], alpha=-1.0, beta=1.0)

    r2_loc = np.ascontiguousarray(r2_loc)
    _allreduce_inplace_mpi_rccsdt(mycc, r2_loc, 'r2_loc_t3', op=MPI.SUM)
    r2 += r2_loc

    r1_loc = None
    r2_loc = None

    return r1, r2

def intermediates_t3_add_t3_tri(mycc, imds, t3):
    '''Add the T3-dependent contributions to the T3 intermediates, with T3 stored in triangular form.
    MPI communication is performed to gather the contributions from distributed T3 amplitudes.
    '''
    einsum = functools.partial(_einsum, mycc.einsum_backend)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_eris = imds.t1_eris
    W_vooo, W_vvvo = imds.W_vooo, imds.W_vvvo
    dt3, t3_local = t3

    W_vooo_loc = np.zeros_like(W_vooo)
    W_vvvo_loc = np.zeros_like(W_vvvo)

    t3_tmp = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    for i, j, k, _ in dt3.iter_local_ijk():
        if i < j < k:
            perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
        elif i == j < k:
            perms = [(i, j, k), (i, k, j), (k, i, j)]
        elif i < j == k:
            perms = [(i, j, k), (j, i, k), (j, k, i)]
        elif i == j == k:
            perms = [(i, j, k)]

        for (pi, pj, pk) in perms:
            dt3.take_t3_single_local(t3_local, t3_tmp, pi, pj, pk)
            t3_single_spin_summation_inplace_(t3_tmp, nvir, "P3_201", 1.0, 0.0)
            einsum('mde,ead->am', t1_eris[:nocc, pi, nocc:, nocc:],
                t3_tmp, out=W_vooo_loc[:, :, pj, pk], alpha=1.0, beta=1.0)
            einsum('de,eba->abd', t1_eris[pk, pi, nocc:, nocc:],
                t3_tmp, out=W_vvvo_loc[:, :, :, pj], alpha=-1.0, beta=1.0)
    t3_tmp = None

    _allreduce_inplace_mpi_rccsdt(mycc, W_vooo_loc, 'W_vooo_t3', op=MPI.SUM)
    _allreduce_inplace_mpi_rccsdt(mycc, W_vvvo_loc, 'W_vvvo_t3', op=MPI.SUM)

    W_vooo += W_vooo_loc
    W_vvvo += W_vvvo_loc
    W_vooo_loc = None
    W_vvvo_loc = None
    return imds

def compute_r3_tri(mycc, imds, t2, t3):
    '''Compute r3 with triangular-stored T3 amplitudes; r3 is returned in triangular form as well.
    r3 will require a symmetry restoration step afterward.
    MPI communication is performed to gather the contributions from distributed T3 amplitudes.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)
    log_memory(mycc, memlog, 'compute r3 start')

    einsum = functools.partial(_einsum, mycc.einsum_backend)
    dt3, t3_local = t3
    F_vv, W_vooo, W_vvvo, W_vvvv = imds.F_vv, imds.W_vooo, imds.W_vvvo, imds.W_vvvv

    r3_local = np.empty_like(t3_local)
    time2 = logger.process_clock(), logger.perf_counter()
    log_memory(mycc, memlog, 'r3 buffers allocated')
    for i, j, k, local_idx in dt3.iter_local_ijk():
        r3_tmp = r3_local[local_idx]
        t3_tmp = t3_local[local_idx]
        einsum('abd,dc->abc', W_vvvo[..., j], t2[i, k], out=r3_tmp, alpha=1.0, beta=0.0)
        einsum('acd,db->abc', W_vvvo[..., k], t2[i, j], out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('bad,dc->abc', W_vvvo[..., i], t2[j, k], out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('bcd,da->abc', W_vvvo[..., k], t2[j, i], out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('cad,db->abc', W_vvvo[..., i], t2[k, j], out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('cbd,da->abc', W_vvvo[..., j], t2[k, i], out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('al,lbc->abc', W_vooo[:, :, i, j], t2[:, k], out=r3_tmp, alpha=-1.0, beta=1.0)
        einsum('al,lcb->abc', W_vooo[:, :, i, k], t2[:, j], out=r3_tmp, alpha=-1.0, beta=1.0)
        einsum('bl,lac->abc', W_vooo[:, :, j, i], t2[:, k], out=r3_tmp, alpha=-1.0, beta=1.0)
        einsum('bl,lca->abc', W_vooo[:, :, j, k], t2[:, i], out=r3_tmp, alpha=-1.0, beta=1.0)
        einsum('cl,lab->abc', W_vooo[:, :, k, i], t2[:, j], out=r3_tmp, alpha=-1.0, beta=1.0)
        einsum('cl,lba->abc', W_vooo[:, :, k, j], t2[:, i], out=r3_tmp, alpha=-1.0, beta=1.0)
        einsum('ad,dbc->abc', F_vv, t3_tmp, out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('bd,adc->abc', F_vv, t3_tmp, out=r3_tmp, alpha=1.0, beta=1.0)
        einsum('cd,abd->abc', F_vv, t3_tmp, out=r3_tmp, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1(
            contraction_message(mycc, 't3: iter: W_vvvo, W_vooo, F_vv %3d:' % (local_idx)),
            *time2)
    F_vv = imds.F_vv = None
    W_vooo = imds.W_vooo = None
    W_vvvo = imds.W_vvvo = None

    log_memory(mycc, memlog, 'r3 W_vvvo/W_vooo/F_vv contractions')
    time1 = log.timer_debug1(contraction_message(mycc, 't3: W_vvvo * t2, W_vooo * t2, F_vv * t3'), *time1)
    mycc.comm.Barrier()

    compute_oooo_oovv_contraction_(mycc, imds, t3, r3_local)

    log_memory(mycc, memlog, 'r3 W_ovov/W_oooo contractions')
    time1 = log.timer_debug1(contraction_message(mycc, 't3: W_ovov * t3, W_oooo * t3'), *time1)
    mycc.comm.Barrier()

    time2 = logger.process_clock(), logger.perf_counter()
    for i, j, k, local_idx in dt3.iter_local_ijk():
        t3_tmp_s = t3_local[local_idx]
        r3_tmp_s = r3_local[local_idx]
        einsum('abde,dec->abc', W_vvvv, t3_tmp_s, out=r3_tmp_s, alpha=1.0, beta=1.0)
        einsum('acde,dbe->abc', W_vvvv, t3_tmp_s, out=r3_tmp_s, alpha=1.0, beta=1.0)
        einsum('bcde,ade->abc', W_vvvv, t3_tmp_s, out=r3_tmp_s, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1(contraction_message(mycc, 't3: iter: W_vvvv %3d:' % local_idx), *time2)
    W_vvvv = imds.W_vvvv = None

    log_memory(mycc, memlog, 'r3 W_vvvv contractions')
    time1 = log.timer_debug1(contraction_message(mycc, 't3: W_vvvv * t3'), *time1)
    mycc.comm.Barrier()
    return r3_local

def compute_oooo_oovv_contraction_(mycc, imds, t3, r3_local):
    '''r3_local is updated in place.
    '''
    log = contraction_logger(mycc)
    memlog = memory_logger(mycc)
    log_memory(mycc, memlog, 'OO T3 contraction start')

    einsum = functools.partial(_einsum,  mycc.einsum_backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    W_ovvo, W_ovov = imds.W_ovvo, imds.W_ovov
    W_ovvo = np.ascontiguousarray(W_ovvo)
    dt3, t3_local = t3
    nocc3 = nocc * (nocc + 1) * (nocc + 2) // 6
    batch_size = mycc.batch_size
    batch_size = min(batch_size, nocc3)

    local_ijk = []
    for local_idx, triple in enumerate(dt3.local_triples):
        local_ijk.append((triple[0], triple[1], triple[2]))
    local_ijk = np.asarray(local_ijk, dtype=np.int64).reshape(-1, 3)

    ijk_list = []
    for i in range(nocc):
        for j in range(i, nocc):
            for k in range(j, nocc):
                ijk_list.append((i, j, k))
    ijk_list = np.array(ijk_list, dtype=np.int32)

    t3_tmp_p3_v0 = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    t3_tmp_p3_v1 = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    t3_tmp_p3_v2 = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    t3_tmp_2 = np.empty((nvir,) * 3, dtype=t3_local.dtype)
    time2 = logger.process_clock(), logger.perf_counter()

    # Build list of batch ranges
    batches = []
    for batch_start in range(0, len(ijk_list), batch_size):
        batch_end = min(batch_start + batch_size, len(ijk_list))
        batches.append((batch_start, batch_end))

    n_batches = len(batches)
    handle = None

    progress_thread = start_mpi_progress_thread(mycc) if n_batches > 0 else None
    try:
        # Prefetch first batch
        if n_batches > 0 and mycc.size > 1:
            batch_start, batch_end = batches[0]
            batch_ijk = ijk_list[batch_start:batch_end]
            if mycc.size > 1:
                handle = dt3.prefetch_t3_triples_iallgather(
                    t3_local, batch_ijk, batch_index=1,
                    batch_start=batch_start, batch_end=batch_end)
                if progress_thread is not None and handle:
                    progress_thread.add_requests(handle.get('reqs', ()))

        for i_batch in range(n_batches):
            batch_start, batch_end = batches[i_batch]
            batch_ijk = ijk_list[batch_start:batch_end]

            # Prefetch next batch while we finalize current
            handle_next = None
            if i_batch + 1 < n_batches and mycc.size > 1:
                next_start, next_end = batches[i_batch + 1]
                next_batch_ijk = ijk_list[next_start:next_end]
                if mycc.size > 1:
                    handle_next = dt3.prefetch_t3_triples_iallgather(
                        t3_local, next_batch_ijk, batch_index=i_batch + 2,
                        batch_start=next_start, batch_end=next_end)
                    if progress_thread is not None and handle_next:
                        progress_thread.add_requests(handle_next.get('reqs', ()))

            # Finalize current batch communication
            if mycc.size > 1:
                if progress_thread is not None:
                    progress_thread.pause()
                try:
                    t3_data, batch_ijk_reordered = dt3.finalize_prefetch_t3_triples(handle, t3_local, batch_ijk)
                finally:
                    if progress_thread is not None:
                        if handle_next is not None:
                            progress_thread.set_requests(handle_next.get('reqs', ()))
                        else:
                            progress_thread.clear_requests()
                        progress_thread.resume()
            else:
                # Single-rank
                t3_data = t3_local[batch_start:batch_end]
                batch_ijk_reordered = batch_ijk

            # Compute on current batch (overlaps with next batch prefetch)
            compute_r3_Foo_Woooo_contribution_batch_(mycc, r3_local, dt3, t3_data, batch_ijk_reordered, imds)
            punctuate_mpi_progress(mycc, progress_thread)

            for idx, (o00, o10, o20) in enumerate(batch_ijk_reordered):
                t3_tmp = t3_data[idx]
                t3_transpose_add_(t3_tmp, t3_tmp_2, nvir)
                t3_spin_summation_triple_sym_(t3_tmp, t3_tmp_p3_v0, t3_tmp_p3_v1, t3_tmp_p3_v2, nvir)
                if idx % 10 == 0:
                    punctuate_mpi_progress(mycc, progress_thread)
                for i, j, k, local_idx in dt3.iter_local_ijk():
                    r3_tmp = r3_local[local_idx]
                    # part 3
                    # case 0: j <= k <= l: j=o00, k=o10, l=o20
                    if j == o00 and k == o10:
                        einsum('bd,dca->abc', W_ovov[o20, :, i, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,dcb->abc', W_ovov[o20, :, i, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('cd,bda->abc', W_ovov[o20, :, i, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,bcd->abc', W_ovvo[o20, :, :, i], t3_tmp_p3_v0, out=r3_tmp, alpha=0.5, beta=1.0)
                    # case 1: j <= l < k: j=o00, l=o10, k=o20
                    if o10 != o20 and j == o00 and k == o20:
                        einsum('bd,dac->abc', W_ovov[o10, :, i, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('cd,bad->abc', W_ovov[o10, :, i, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,bcd->abc', W_ovov[o10, :, i, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('ad,bdc->abc', W_ovvo[o10, :, :, i], t3_tmp_p3_v1, out=r3_tmp, alpha=0.5, beta=1.0)
                    # case 2: l < j <= k: l=o00, j=o10, k=o20
                    if o00 != o10 and j == o10 and k == o20:
                        einsum('bd,adc->abc', W_ovov[o00, :, i, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,bdc->abc', W_ovov[o00, :, i, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('cd,abd->abc', W_ovov[o00, :, i, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,dbc->abc', W_ovvo[o00, :, :, i], t3_tmp_p3_v2, out=r3_tmp, alpha=0.5, beta=1.0)
                    # part 4
                    # case 0: i <= k <= l: i=o00, k=o10, l=o20
                    if i == o00 and k == o10:
                        einsum('ad,dcb->abc', W_ovov[o20, :, j, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('bd,dca->abc', W_ovov[o20, :, j, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('cd,adb->abc', W_ovov[o20, :, j, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('bd,acd->abc', W_ovvo[o20, :, :, j], t3_tmp_p3_v0, out=r3_tmp, alpha=0.5, beta=1.0)
                    # case 1: i <= l < k: i=o00, l=o10, k=o20
                    if o10 != o20 and i == o00 and k == o20:
                        einsum('ad,dbc->abc', W_ovov[o10, :, j, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('cd,abd->abc', W_ovov[o10, :, j, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('bd,acd->abc', W_ovov[o10, :, j, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('bd,adc->abc', W_ovvo[o10, :, :, j], t3_tmp_p3_v1, out=r3_tmp, alpha=0.5, beta=1.0)
                    # case 2: l < i <= k: l=o00, i=o10, k=o20
                    if o00 != o10 and i == o10 and k == o20:
                        einsum('ad,bdc->abc', W_ovov[o00, :, j, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('bd,adc->abc', W_ovov[o00, :, j, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('cd,bad->abc', W_ovov[o00, :, j, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('bd,dac->abc', W_ovvo[o00, :, :, j], t3_tmp_p3_v2, out=r3_tmp, alpha=0.5, beta=1.0)
                    # part 5
                    # case 0: i <= j <= l: i=o00, j=o10, l=o20
                    if i == o00 and j == o10:
                        einsum('bd,adc->abc', W_ovov[o20, :, k, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,dbc->abc', W_ovov[o20, :, k, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('cd,dba->abc', W_ovov[o20, :, k, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('cd,abd->abc', W_ovvo[o20, :, :, k], t3_tmp_p3_v0, out=r3_tmp, alpha=0.5, beta=1.0)
                    # case 1: i <= l < j: i=o00, l=o10, j=o20
                    if o10 != o20 and i == o00 and j == o20:
                        einsum('cd,abd->abc', W_ovov[o10, :, k, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('bd,acd->abc', W_ovov[o10, :, k, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,dcb->abc', W_ovov[o10, :, k, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('cd,adb->abc', W_ovvo[o10, :, :, k], t3_tmp_p3_v1, out=r3_tmp, alpha=0.5, beta=1.0)
                    # case 2: l < i <= j: l=o00, i=o10, j=o20
                    if o00 != o10 and i == o10 and j == o20:
                        einsum('bd,cad->abc', W_ovov[o00, :, k, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('ad,cdb->abc', W_ovov[o00, :, k, :], t3_tmp, out=r3_tmp, alpha=-1.0, beta=1.0)
                        einsum('cd,adb->abc', W_ovov[o00, :, k, :], t3_tmp_2, out=r3_tmp, alpha=-0.5, beta=1.0)
                        einsum('cd,dab->abc', W_ovvo[o00, :, :, k], t3_tmp_p3_v2, out=r3_tmp, alpha=0.5, beta=1.0)
            time2 = log.timer_debug1(
                contraction_message(mycc, 't3: iter: W_ovov, W_oooo %4d, %4d:' % (batch_start, batch_end)),
                *time2)
            log_memory(mycc, memlog, 'OO T3 batch %4d:%4d' % (batch_start, batch_end), per_iter=True)
            # Advance handle
            handle = handle_next
    finally:
        if progress_thread is not None:
            progress_thread.stop()

    W_ovov = imds.W_ovov = None
    W_ovvo = imds.W_ovvo = None
    t3_tmp = t3_tmp_2 = t3_tmp_p3_v0 = t3_tmp_p3_v1 = t3_tmp_p3_v2 = None
    return r3_local

def compute_r3_Foo_Woooo_contribution_batch_(mycc, r3_local, dt3, t3_data, batch_ijk, imds):

    assert r3_local.dtype == np.float64 and r3_local.flags['C_CONTIGUOUS']
    assert t3_data.dtype == np.float64 and t3_data.flags['C_CONTIGUOUS']
    assert imds.F_oo.dtype == np.float64 and imds.F_oo.flags['C_CONTIGUOUS']
    assert imds.W_oooo.dtype == np.float64 and imds.W_oooo.flags['C_CONTIGUOUS']
    assert batch_ijk.dtype == np.int32 and batch_ijk.flags['C_CONTIGUOUS']
    assert batch_ijk.shape[1] == 3

    F_oo, W_oooo = imds.F_oo, imds.W_oooo
    batch_size = batch_ijk.shape[0]

    pfs = np.zeros((6, batch_size, r3_local.shape[0]), dtype=W_oooo.dtype)

    for idx, (o00, o10, o20) in enumerate(batch_ijk):
        for i, j, k, local_idx in dt3.iter_local_ijk():
            # part 0: np.einsum('lmij,lmkabc->ijkabc', W_oooo, t3, optimize=True)
            # case 0: l <= m <= k: l=o00, m=o10, k=o20
            if k == o20:
                pfs[0, idx, local_idx] += W_oooo[o00, o10, i, j] # r3_tmp += W_oooo[o00, o10, i, j] * t3_tmp
            # case 1: l <= k < m: l=o00, k=o10, m=o20
            if o10 != o20 and k == o10:
                pfs[1, idx, local_idx] += W_oooo[o00, o20, i, j] # r3_tmp += W_oooo[o00, o20, i, j] * t3_tmp.transpose(0, 2, 1)
            # case 2: m < l <= k: m=o00, l=o10, k=o20
            if o00 != o10 and k == o20:
                pfs[2, idx, local_idx] += W_oooo[o10, o00, i, j] # r3_tmp += W_oooo[o10, o00, i, j] * t3_tmp.transpose(1, 0, 2)
            # case 3: m <= k < l: m=o00, k=o10, l=o20
            if o10 != o20 and k == o10:
                pfs[3, idx, local_idx] += W_oooo[o20, o00, i, j] # r3_tmp += W_oooo[o20, o00, i, j] * t3_tmp.transpose(2, 0, 1)
            # case 4: k < l <= m: k=o00, l=o10, m=o20
            if o00 != o10 and k == o00:
                pfs[4, idx, local_idx] += W_oooo[o10, o20, i, j] # r3_tmp += W_oooo[o10, o20, i, j] * t3_tmp.transpose(1, 2, 0)
            # case 5: k < m < l: k=o00, m=o10, l=o20
            if o00 != o10 and o10 != o20 and k == o00:
                pfs[5, idx, local_idx] += W_oooo[o20, o10, i, j] # r3_tmp += W_oooo[o20, o10, i, j] * t3_tmp.transpose(2, 1, 0)

            # part 1: np.einsum('lmik,lmjacb->ijkabc', W_oooo, t3, optimize=True)
            # case 0: l <= m <= j: l=o00, m=o10, j=o20
            if j == o20:
                pfs[1, idx, local_idx] += W_oooo[o00, o10, i, k] # r3_tmp += W_oooo[o00, o10, i, k] * t3_tmp.transpose(0, 2, 1)
            # case 1: l <= j < m: l=o00, j=o10, m=o20
            if o10 != o20 and j == o10:
                pfs[0, idx, local_idx] += W_oooo[o00, o20, i, k] # r3_tmp += W_oooo[o00, o20, i, k] * t3_tmp
            # case 2: m < l <= j: m=o00, l=o10, j=o20
            if o00 != o10 and j == o20:
                pfs[4, idx, local_idx] += W_oooo[o10, o00, i, k] # r3_tmp += W_oooo[o10, o00, i, k] * t3_tmp.transpose(1, 2, 0)
            # case 3: m <= j < l: m=o00, j=o10, l=o20
            if o10 != o20 and j == o10:
                pfs[5, idx, local_idx] += W_oooo[o20, o00, i, k] # r3_tmp += W_oooo[o20, o00, i, k] * t3_tmp.transpose(2, 1, 0)
            # case 4: j < l <= m: j=o00, l=o10, m=o20
            if o00 != o10 and j == o00:
                pfs[2, idx, local_idx] += W_oooo[o10, o20, i, k] # r3_tmp += W_oooo[o10, o20, i, k] * t3_tmp.transpose(1, 0, 2)
            # case 5: j < m < l: j=o00, m=o10, l=o20
            if o00 != o10 and o10 != o20 and j == o00:
                pfs[3, idx, local_idx] += W_oooo[o20, o10, i, k] # r3_tmp += W_oooo[o20, o10, i, k] * t3_tmp.transpose(2, 0, 1)

            # part 2: np.einsum('lmjk,lmibca->ijkabc', W_oooo, t3, optimize=True)
            # case 0: l <= m <= i: l=o00, m=o10, i=o20
            if i == o20:
                pfs[3, idx, local_idx] += W_oooo[o00, o10, j, k] # r3_tmp += W_oooo[o00, o10, j, k] * t3_tmp.transpose(2, 0, 1)
            # case 1: l <= i < m: l=o00, i=o10, m=o20
            if o10 != o20 and i == o10:
                pfs[2, idx, local_idx] += W_oooo[o00, o20, j, k] # r3_tmp += W_oooo[o00, o20, j, k] * t3_tmp.transpose(1, 0, 2)
            # case 2: m < l <= i: m=o00, l=o10, i=o20
            if o00 != o10 and i == o20:
                pfs[5, idx, local_idx] += W_oooo[o10, o00, j, k] # r3_tmp += W_oooo[o10, o00, j, k] * t3_tmp.transpose(2, 1, 0)
            # case 3: m <= i < l: m=o00, i=o10, l=o20
            if o10 != o20 and i == o10:
                pfs[4, idx, local_idx] += W_oooo[o20, o00, j, k] # r3_tmp += W_oooo[o20, o00, j, k] * t3_tmp.transpose(1, 2, 0)
            # case 4: i < l <= m: i=o00, l=o10, m=o20
            if o00 != o10 and i == o00:
                pfs[0, idx, local_idx] += W_oooo[o10, o20, j, k] # r3_tmp += W_oooo[o10, o20, j, k] * t3_tmp.transpose(0, 1, 2)
            # case 5: i < m < l: i=o00, m=o10, l=o20
            if o00 != o10 and o10 != o20 and i == o00:
                pfs[1, idx, local_idx] += W_oooo[o20, o10, j, k] # r3_tmp += W_oooo[o20, o10, j, k] * t3_tmp.transpose(0, 2, 1)

            # part 3
            # case 0: j <= k <= l: j=o00, k=o10, l=o20
            if j == o00 and k == o10:
                pfs[3, idx, local_idx] -= F_oo[o20, i] # r3_tmp -= F_oo[o20, i] * t3_tmp.transpose(2, 0, 1)
            # case 1: j <= l < k: j=o00, l=o10, k=o20
            if o10 != o20 and j == o00 and k == o20:
                pfs[2, idx, local_idx] -= F_oo[o10, i] # r3_tmp -= F_oo[o10, i] * t3_tmp.transpose(1, 0, 2)
            # case 2: l < j <= k: l=o00, j=o10, k=o20
            if o00 != o10 and j == o10 and k == o20:
                pfs[0, idx, local_idx] -= F_oo[o00, i] # r3_tmp -= F_oo[o00, i] * t3_tmp.transpose(0, 1, 2)

            # part 4
            # case 0: i <= k <= l: i=o00, k=o10, l=o20
            if i == o00 and k == o10:
                pfs[1, idx, local_idx] -= F_oo[o20, j] # r3_tmp -= F_oo[o20, j] * t3_tmp.transpose(0, 2, 1)
            # case 1: i <= l < k: i=o00, l=o10, k=o20
            if o10 != o20 and i == o00 and k == o20:
                pfs[0, idx, local_idx] -= F_oo[o10, j] # r3_tmp -= F_oo[o10, j] * t3_tmp.transpose(0, 1, 2)
            # case 2: l < i <= k: l=o00, i=o10, k=o20
            if o00 != o10 and i == o10 and k == o20:
                pfs[2, idx, local_idx] -= F_oo[o00, j] # r3_tmp -= F_oo[o00, j] * t3_tmp.transpose(1, 0, 2)

            # part 5
            # case 0: i <= j <= l: i=o00, j=o10, l=o20
            if i == o00 and j == o10:
                pfs[0, idx, local_idx] -= F_oo[o20, k] # r3_tmp -= F_oo[o20, k] * t3_tmp.transpose(0, 1, 2)
            # case 1: i <= l < j: i=o00, l=o10, j=o20
            if o10 != o20 and i == o00 and j == o20:
                pfs[1, idx, local_idx] -= F_oo[o10, k] # r3_tmp -= F_oo[o10, k] * t3_tmp.transpose(0, 2, 1)
            # case 2: l < i <= j: l=o00, i=o10, j=o20
            if o00 != o10 and i == o10 and j == o20:
                pfs[4, idx, local_idx] -= F_oo[o00, k] # r3_tmp -= F_oo[o00, k] * t3_tmp.transpose(1, 2, 0)

    einsum = functools.partial(_einsum, mycc.einsum_backend)
    einsum('pq,pabc->qabc', pfs[0], t3_data, out=r3_local, alpha=1.0, beta=1.0)
    einsum('pq,pacb->qabc', pfs[1], t3_data, out=r3_local, alpha=1.0, beta=1.0)
    einsum('pq,pbac->qabc', pfs[2], t3_data, out=r3_local, alpha=1.0, beta=1.0)
    einsum('pq,pabc->qcab', pfs[3], t3_data, out=r3_local, alpha=1.0, beta=1.0)
    einsum('pq,pabc->qbca', pfs[4], t3_data, out=r3_local, alpha=1.0, beta=1.0)
    einsum('pq,pabc->qcba', pfs[5], t3_data, out=r3_local, alpha=1.0, beta=1.0)
    return r3_local

def r3_tri_divide_e_(mycc, t3, r3, mo_energy):
    nocc = mycc.nocc
    dt3, _ = t3
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:] - mycc.level_shift
    for i, j, k, local_index in dt3.iter_local_ijk():
        eijkabc_blk = (eia[i, :, None, None] + eia[j, None, :, None] + eia[k, None, None, :])
        r3_tmp = r3[local_index]
        r3_tmp /= eijkabc_blk
    eijkabc_blk = None
    mycc.comm.Barrier()
    return r3

def memory_estimate_log_mpi_rccsdt(mycc):
    '''Estimate per-rank memory for the distributed RCCSDT implementation.'''
    if mycc.rank != 0:
        return mycc

    log = logger.Logger(mycc.stdout, mycc.verbose if mycc.rank == 0 else 0)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    itemsize = np.dtype(np.float64).itemsize
    size = mycc.size

    nocc3_stored = nocc * (nocc + 1) * (nocc + 2) // 6
    local_max = (nocc3_stored + size - 1) // size
    nvir3 = nvir**3

    local_t3_memory = local_max * nvir3 * itemsize
    local_r3_memory = local_t3_memory
    global_t3_footprint = nocc3_stored * nvir3 * itemsize

    batch_size = nocc3_stored if mycc.batch_size is None else int(mycc.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch = min(batch_size, nocc3_stored) if nocc3_stored else 0
    n_batches = (nocc3_stored + batch - 1) // batch if batch else 0
    t3_block_memory = nvir3 * itemsize
    if size > 1 and batch > 0:
        prefetch_slots = 2 if n_batches > 1 else 1
        t3_batch_recv_memory = prefetch_slots * batch * t3_block_memory
        t3_batch_send_memory = prefetch_slots * min(batch, local_max) * t3_block_memory
        t3_batch_collect_memory = 0
    else:
        prefetch_slots = 0
        t3_batch_recv_memory = 0
        t3_batch_send_memory = 0
        t3_batch_collect_memory = batch * t3_block_memory
    t3_batch_memory = t3_batch_recv_memory + t3_batch_send_memory + t3_batch_collect_memory

    eris_memory = nmo**4 * itemsize
    eris_runtime_memory = 3 * eris_memory
    r1_memory = nocc * nvir * itemsize
    r2_memory = nocc**2 * nvir**2 * itemsize
    w_oooo_memory = nocc**4 * itemsize
    w_ovov_memory = nocc**2 * nvir**2 * itemsize
    t3_temp_memory = 5 * t3_block_memory
    t3_work_memory = max(local_r3_memory + t3_temp_memory,
                         local_r3_memory + w_oooo_memory + w_ovov_memory + t3_batch_memory)
    intermediates_work_memory = max(r1_memory + r2_memory, t3_work_memory)

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
        nvir_t3_diis = resolve_nvir_diis(mycc, nvir)
        nocc2_unique = nocc * (nocc + 1) // 2
        lower_t_memory = (nocc * nvir + nocc2_unique * nvir**2) * itemsize
        local_t3_diis_memory = local_max * nvir_t3_diis**3 * itemsize
        diis_vector_memory = lower_t_memory + local_t3_diis_memory
        diis_history_memory = diis_vector_memory * diis_incore_space * 2
        diis_xprev_memory = diis_vector_memory
        diis_resident_memory = diis_history_memory + diis_xprev_memory
        diis_live_memory = 4 * diis_vector_memory
        diis_memory = diis_resident_memory + diis_live_memory
        diis_scratch_memory = diis_vector_memory * (mycc.diis_space - diis_incore_space) * 2
    elif mycc.diis and mycc.incore_complete:
        nocc2_unique = nocc * (nocc + 1) // 2
        diis_vector_memory = (nocc * nvir + nocc2_unique * nvir**2) * itemsize
        diis_memory = diis_vector_memory * mycc.diis_space * 2

    update_work_peak_memory = local_t3_memory + local_r3_memory + eris_runtime_memory + intermediates_work_memory
    update_peak_memory = update_work_peak_memory + diis_resident_memory
    diis_peak_memory = local_t3_memory + eris_memory + diis_memory
    total_memory = max(update_peak_memory, diis_peak_memory)

    fmt = format_size
    log.info('')
    log.info('Approximate per-rank memory usage estimate')
    log.info('    Local T3 memory (max)   %8s', fmt(local_t3_memory))
    log.info('    Local R3 memory (max)   %8s', fmt(local_r3_memory))
    log.info('    Global T3 footprint     %8s', fmt(global_t3_footprint))
    log.info('    T3 prefetch recv bufs   %8s', fmt(t3_batch_recv_memory))
    log.info('    T3 prefetch send bufs   %8s', fmt(t3_batch_send_memory))
    if prefetch_slots:
        log.info('    T3 prefetch slots       %8d', prefetch_slots)
    if t3_batch_collect_memory:
        log.info('    T3 local batch data     %8s', fmt(t3_batch_collect_memory))
    log.info('    T3 temp buffers         %8s', fmt(t3_temp_memory))
    log.info('    ERIs memory             %8s', fmt(eris_memory))
    log.info('    Intermediates/work peak %8s', fmt(intermediates_work_memory))
    log.info('    DIIS memory             %8s', fmt(diis_memory))
    if mycc.do_diis_max_t and mycc.diis:
        log.info('    DIIS resident memory    %8s', fmt(diis_resident_memory))
        log.info('    DIIS transient memory   %8s', fmt(diis_live_memory))
        log.info('    nvir_diis         %8d of %d', nvir_t3_diis, nvir)
        log.info('    DIIS vector size        %8s', fmt(diis_vector_memory))
    if mycc.do_diis_max_t and mycc.diis and diis_scratch is not None:
        log.info('    DIIS in-core slots      %8d of %d', diis_incore_space, mycc.diis_space)
        log.info('    DIIS scratch footprint  %8s', fmt(diis_scratch_memory))
        log.info('    DIIS scratch dir        %s', diis_scratch)
    log.info('Update work peak            %8s', fmt(update_work_peak_memory))
    log.info('Update estimated peak       %8s', fmt(update_peak_memory))
    if mycc.diis:
        log.info('DIIS estimated peak         %8s', fmt(diis_peak_memory))
    log.info('Total estimated per-rank    %8s', fmt(total_memory))
    log.info('')

    max_memory = mycc.max_memory - lib.current_memory()[0]
    if (total_memory / 1024**2) > max_memory:
        logger.warn(mycc, 'Estimated per-rank memory %.2f MB exceeds available %.2f MB',
                    total_memory / 1024**2, max_memory)
    return mycc

def update_amps_rccsdt_tri_(mycc, tamps, eris):
    '''Update RCCSDT amplitudes in place, with T3 amplitudes stored in triangular form.
    MPI communication is performed within the called functions as needed.
    '''
    assert (isinstance(eris, _PhysicistsERIs))

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose if mycc.rank == 0 else 0)
    memlog = memory_logger(mycc)
    log_memory(mycc, memlog, 'update_amps start')

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3 = tamps
    _, t3_local = t3
    mo_energy = eris.mo_energy
    imds = _IMDS()

    # t1 t2
    update_t1_fock_eris(mycc, imds, t1, eris)
    if mycc.rank == 0:
        time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2(mycc, imds, t2)
    if mycc.rank == 0:
        time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, imds, t2)
    r1r2_add_t3_tri_(mycc, imds, r1, r2, t3)
    if mycc.rank == 0:
        time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # symmetrization
    r2 += r2.transpose(1, 0, 3, 2)
    if mycc.rank == 0:
        time1 = log.timer_debug1('t1t2: symmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_(mycc, r1, r2, mo_energy)
    if mycc.rank == 0:
        time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)
    res_norm = [np.linalg.norm(r1), np.linalg.norm(r2)]
    t1 += r1
    t2 += r2
    if mycc.rank == 0:
        time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
        time0 = log.timer_debug1('t1t2 total', *time0)
    log_memory(mycc, memlog, 't1t2 update t1 and t2')
    # t3
    intermediates_t3(mycc, imds, t2)
    intermediates_t3_add_t3_tri(mycc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    if mycc.rank == 0:
        time1 = log.timer_debug1('t3: update intermediates', *time0)
    log_memory(mycc, memlog, 't3 update intermediates')
    r3 = compute_r3_tri(mycc, imds, t2, t3)
    imds = None
    if mycc.rank == 0:
        time1 = log.timer_debug1('t3: compute r3', *time1)
    log_memory(mycc, memlog, 't3 compute r3')
    # symmetrization
    t3_spin_summation_inplace_(r3, r3.shape[0], nvir, "P3_full", -1.0 / 6.0, 1.0)
    if mycc.rank == 0:
        time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    log_memory(mycc, memlog, 't3 symmetrize r3')
    # divide by eijkabc
    r3_tri_divide_e_(mycc, t3, r3, mo_energy)
    if mycc.rank == 0:
        time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)
    log_memory(mycc, memlog, 't3 divide r3')

    r3_norm_sq_local = np.linalg.norm(r3)**2
    nvir_t3_diis = resolve_nvir_diis(mycc, nvir)
    if nvir_t3_diis < nvir:
        r3_active = r3[:, :nvir_t3_diis, :nvir_t3_diis, :nvir_t3_diis]
        r3_active_norm_sq_local = np.linalg.norm(r3_active)**2
    else:
        r3_active_norm_sq_local = r3_norm_sq_local
    r3_norms = np.array([r3_norm_sq_local, r3_active_norm_sq_local], dtype=np.float64)
    _allreduce_inplace_mpi_rccsdt(mycc, r3_norms, 'r3_norms', op=MPI.SUM)
    r3_norm_sq = float(r3_norms[0])
    mycc._last_t3_residual_norm_sq = r3_norm_sq
    mycc._last_t3_active_residual_norm_sq = float(r3_norms[1])
    res_norm.append(np.sqrt(r3_norm_sq))

    t3_local += r3
    r3 = None
    if mycc.rank == 0:
        time1 = log.timer_debug1('t3: update t3', *time1)
        time0 = log.timer_debug1('t3 total', *time0)
    log_memory(mycc, memlog, 't3 update t3')
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

def run_diis_t3(mycc, tamps, t3_local, istep, normt, de, adiis_t3, res_norm=None, return_update_norm=False):
    if not (adiis_t3 and istep >= mycc.diis_start_cycle and abs(de) < mycc.diis_start_energy_diff):
        if return_update_norm:
            update_norm = float(normt) if res_norm is None else _safe_update_norm(norm_sq_from_norms(res_norm))
            return tamps, update_norm
        return tamps

    t1, t2, t3 = tamps
    dt3, t3_local = t3
    scale = 1.0 / np.sqrt(mycc.size)
    nvir = mycc.nmo - mycc.nocc
    nvir_t3_diis = resolve_nvir_diis(mycc, nvir)
    if nvir_t3_diis < nvir:
        t3_diis = t3_local[:, :nvir_t3_diis, :nvir_t3_diis, :nvir_t3_diis]
    else:
        t3_diis = t3_local
    t3_diis_shape = t3_diis.shape
    lower_vector = pack_unique_replicated_tamps(mycc, (t1, t2), scale=scale)
    vector = np.concatenate((lower_vector, t3_diis.ravel()))

    new_vector = adiis_t3.update(vector)

    (t1_new, t2_new), offset = unpack_unique_replicated_tamps(mycc, new_vector, (t1, t2), scale=scale)
    tamps[0] = t1_new
    tamps[1] = t2_new

    if nvir_t3_diis < nvir:
        t3_local[:, :nvir_t3_diis, :nvir_t3_diis, :nvir_t3_diis] = new_vector[offset:].reshape(t3_diis_shape)
    else:
        t3_local[...] = new_vector[offset:].reshape(t3_diis_shape)
    tamps[2] = (dt3, t3_local)

    update_norm = None
    if return_update_norm:
        update_norm = _max_t_diis_update_norm(mycc, res_norm, normt, adiis_t3, nvir_t3_diis, nvir,
                                            '_last_t3_residual_norm_sq', '_last_t3_active_residual_norm_sq')

    if mycc.rank == 0:
        logger.debug1(mycc, 'Unified DIIS for T1/T2/T3 amplitudes at step %d', istep)
    if return_update_norm:
        return tamps, update_norm
    return tamps

def kernel(mycc, eris=None, tamps=None, tol=1e-8, tolnormt=1e-6, max_cycle=50, verbose=5, callback=None):

    log = logger.Logger(mycc.stdout, verbose if mycc.rank == 0 else 0)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    if tamps is None:
        tamps = mycc.init_amps(eris)[1]
    else:
        if len(tamps) < mycc.cc_order:
            tamps = list(tamps) + list(mycc.init_amps(eris)[1][len(tamps):])
    configure_t3_runtime_logging(mycc, tamps)

    name = mycc.__class__.__name__

    if mycc.rank == 0:
        cput1 = cput0 = (logger.process_clock(), logger.perf_counter())

    e_corr_old = 0.0
    e_corr = mycc.energy(tamps, eris)

    if mycc.rank == 0:
        log.info('Init E_corr(%s) = %.15g', name, e_corr)

    adiis = None
    adiis_t3 = None
    if mycc.do_diis_max_t:
        adiis_t3 = make_mpi_diis(mycc, log=log)
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
            _, _, t3 = tamps
            _, t3_local = t3
            tamps, update_norm = mycc.run_diis_t3(tamps, t3_local, istep, normt, e_corr - e_corr_old, adiis_t3,
                                                res_norm=res_norm, return_update_norm=True)
        else:
            lower_tamps, update_norm = _call_standard_diis(mycc, tamps[:mycc.cc_order - 1], istep, normt,
                                                           e_corr - e_corr_old, adiis, res_norm)
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

    close_t3_runtime_logging(tamps)
    return converged, e_corr, tamps

def _update_procs_mf(comm, mf):
    '''Update mean-field objects to be the same on all processors'''
    mf1 = mf.copy()
    mo_coeff  = comm.bcast(mf.mo_coeff, root=0)
    mo_energy = comm.bcast(mf.mo_energy, root=0)
    mo_occ    = comm.bcast(mf.mo_occ, root=0)
    mf1.mo_coeff = mo_coeff
    mf1.mo_energy = mo_energy
    mf1.mo_occ = mo_occ
    comm.Barrier()
    return mf1

class RCCSDT(ccsd.CCSDBase):

    conv_tol = getattr(__config__, 'cc_rccsdt_RCCSDT_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_rccsdt_RCCSDT_conv_tol_normt', 1e-6)
    cc_order = getattr(__config__, 'cc_rccsdt_RCCSDT_cc_order', 3)
    do_diis_max_t = getattr(__config__, 'cc_rccsdt_RCCSDT_do_diis_max_t', True)
    batch_size = getattr(__config__, 'cc_rccsdt_RCCSDT_batchsize', 100)
    einsum_backend = getattr(__config__, 'cc_rccsdt_RCCSDT_einsum_backend', 'numpy')
    nvir_diis = getattr(__config__, 'cc_rccsdt_RCCSDT_nvir_diis', None)
    diis_scratch = getattr(__config__, 'cc_rccsdt_RCCSDT_diis_scratch', None)
    diis_scratch_start = getattr(__config__, 'cc_rccsdt_RCCSDT_diis_scratch_start', 0)
    diis_scratch_cleanup = getattr(__config__, 'cc_rccsdt_RCCSDT_diis_scratch_cleanup', True)
    diis_scratch_mmap = getattr(__config__, 'cc_rccsdt_RCCSDT_diis_scratch_mmap', False)
    allow_python_fallback = getattr(__config__, 'cc_rccsdt_RCCSDT_allow_python_fallback', False)
    log_highest_t_communication = getattr(__config__, 'cc_rccsdt_RCCSDT_log_highest_t_communication', False)
    communication_log_dir = getattr(__config__, 'cc_rccsdt_RCCSDT_communication_log_dir', 'comm_logs')
    log_highest_t_contractions = getattr(__config__, 'cc_rccsdt_RCCSDT_log_highest_t_contractions', False)
    log_highest_t_contractions_all_ranks = getattr(__config__,
                                                   'cc_rccsdt_RCCSDT_log_highest_t_contractions_all_ranks', False)
    contraction_log_dir = getattr(__config__, 'cc_rccsdt_RCCSDT_contraction_log_dir', 'contraction_logs')
    log_allreduce_timing = getattr(__config__, 'cc_rccsdt_RCCSDT_log_allreduce_timing', False)
    log_memory = getattr(__config__, 'cc_rccsdt_RCCSDT_log_memory', False)
    log_memory_all_ranks = getattr(__config__, 'cc_rccsdt_RCCSDT_log_memory_all_ranks', False)
    log_memory_per_iter = getattr(__config__, 'cc_rccsdt_RCCSDT_log_memory_per_iter', False)
    # Keep MPI calls on the main thread by default.  The helper progress thread
    # can deadlock some OpenMPI nonblocking collective workloads.
    use_mpi_progress_thread = getattr(__config__, 'cc_rccsdt_RCCSDT_use_mpi_progress_thread', False)
    mpi_progress_poll_interval = getattr(__config__, 'cc_rccsdt_RCCSDT_mpi_progress_poll_interval', 0.001)
    gil_punctuate_duration = getattr(__config__, 'cc_rccsdt_RCCSDT_gil_punctuate_duration', 0.0001)
    gil_punctuate_interval = getattr(__config__, 'cc_rccsdt_RCCSDT_gil_punctuate_interval', 10)

    _keys = {
        'max_cycle', 'conv_tol', 'conv_tol_normt',
        'diis', 'diis_space', 'diis_file', 'diis_start_cycle', 'diis_start_energy_diff',
        'async_io', 'incore_complete', 'callback',
        'mol', 'verbose', 'stdout', 'frozen', 'level_shift', 'mo_coeff', 'mo_occ', 'cycles', 'emp2', 'e_hf',
        'converged', 'e_corr', 'chkfile', 'cc_order', 'do_diis_max_t',
        'einsum_backend', 'tamps', 'unique_tamps_map',
        'size', 'rank', 'comm', 'batch_size', 'nvir_diis',
        'diis_scratch', 'diis_scratch_start', 'diis_scratch_cleanup', 'diis_scratch_mmap',
        'allow_python_fallback',
        'log_highest_t_communication', 'communication_log_dir',
        'log_highest_t_contractions', 'log_highest_t_contractions_all_ranks',
        'contraction_log_dir', 'log_allreduce_timing',
        'log_memory', 'log_memory_all_ranks', 'log_memory_per_iter',
        'use_mpi_progress_thread', 'mpi_progress_poll_interval', 'gil_punctuate_duration',
        'gil_punctuate_interval'
    }

    def __init__(self, mf, comm=None, frozen=None, mo_coeff=None, mo_occ=None):
        if comm is None:
            raise ValueError("This %s implementation requires an MPI communicator." % self.__class__.__name__)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        mf = _update_procs_mf(comm, mf)

        if mo_coeff is not None:
            mo_coeff = comm.bcast(mo_coeff, root=0)

        self.tamps = [None, None, None]
        ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if self.rank != 0:
            self.verbose = 0
        self.unique_tamps_map = None

    @property
    def t1(self):
        return self.tamps[0]

    @t1.setter
    def t1(self, val):
        self.tamps[0] = val

    @property
    def t2(self):
        return self.tamps[1]

    @t2.setter
    def t2(self, val):
        self.tamps[1] = val

    @property
    def t3(self):
        return self.tamps[2]

    @t3.setter
    def t3(self, val):
        self.tamps[2] = val

    do_tri_max_t = property(lambda self: True)

    def set_einsum_backend(self, backend):
        self.einsum_backend = backend

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf
    ao2mo = _ao2mo_rcc
    init_amps = init_amps_rhf
    energy = energy_rhf
    memory_estimate_log = memory_estimate_log_mpi_rccsdt
    update_amps_ = update_amps_rccsdt_tri_
    amplitudes_to_vector = amplitudes_to_vector_rhf
    vector_to_amplitudes = vector_to_amplitudes_rhf
    build_unique_tamps_map = build_unique_tamps_map_rhf
    run_diis = run_diis
    run_diis_t3 = run_diis_t3
    _finalize = _finalize
    dump_flags = dump_flags

    def kernel(self, tamps=None, eris=None):
        return self.ccsdt(tamps, eris)

    def ccsdt(self, tamps=None, eris=None):
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
    from pyscf.data.elements import chemcore
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", verbose=3 if rank == 0 else 0)
    mf = scf.RHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()

    ref_e_corr = -0.3217858674891447
    mycc = RCCSDT(mf, frozen=chemcore(mol), comm=MPI.COMM_WORLD)
    # mycc.set_einsum_backend('pytblis')
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 100
    mycc.verbose = 8

    mycc.batch_size = 11

    mycc.diis = True
    mycc.do_diis_max_t = True
    mycc.nvir_diis = 7
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
        print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_e_corr, mycc.e_corr - ref_e_corr))
        print('\n' * 2)
