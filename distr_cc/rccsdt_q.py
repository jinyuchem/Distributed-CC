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
import gc
import functools
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.rccsdt import _einsum, format_size

from mpi4py import MPI

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distr_cc._rccsdtq_core import t4_add_
from distr_cc._c_rccsdt_q import (e_abcdijkl_division_, fill_t3_from_ptr_array, pack_t3_indices_, promote_t3_blocks_,
                                t4_spin_summation_inplace_, t4_multiply_factor_)
from distr_cc._mpi import punctuate_mpi_progress, start_mpi_progress_thread
from distr_cc._runtime import log_memory, memory_logger, warn_non_pytblis_backend

def prepare_t2_for_q(mycc, t2):
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    if t2.shape == (nvir, nvir, nocc, nocc):
        return np.ascontiguousarray(t2)
    if t2.shape == (nocc, nocc, nvir, nvir):
        return np.ascontiguousarray(t2.transpose(2, 3, 0, 1))
    raise ValueError("RCCSDT(Q) expected T2 with shape (nocc,nocc,nvir,nvir) "
                    "or prepared shape (nvir,nvir,nocc,nocc); got %s" % (t2.shape,))

def redistribute_t3_ijk_to_abc(dt3_ijk, t3_local_ijk, comm, blksize_abc=None, chunk_size=None):
    from distr_cc.distribute_t3 import redistribute_ijk_to_abc
    return redistribute_ijk_to_abc(dt3_ijk, t3_local_ijk, comm, blksize_abc=blksize_abc, chunk_size=chunk_size)

def prepare_t3_for_q(t3, comm=None, blksize=4, chunk_size=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    dt3, t3_local = t3
    if hasattr(dt3, 'abc_to_global_idx'):
        return t3
    return redistribute_t3_ijk_to_abc(dt3, t3_local, comm, blksize_abc=blksize, chunk_size=chunk_size)

def prepare_tamps_for_q(mycc, tamps=None, blksize=4, comm=None, chunk_size=None):
    if tamps is None:
        tamps = mycc.tamps
    if len(tamps) < 3:
        raise ValueError("RCCSDT(Q) requires at least T1, T2, and T3 amplitudes")
    if comm is None:
        comm = getattr(mycc, 'comm', MPI.COMM_WORLD)
    t1, t2, t3 = tamps[:3]
    t2_q = prepare_t2_for_q(mycc, t2)
    t3_q = prepare_t3_for_q(t3, comm=comm, blksize=blksize, chunk_size=chunk_size)
    return [t1, t2_q, t3_q]

def memory_estimate_log_rccsdt_q(mycc, nvir, nocc, blksize, n_tasks=None, local_tasks=None, comm=None):
    rank = getattr(mycc, 'rank', 0)
    if comm is not None:
        rank = comm.Get_rank()
    if rank != 0:
        return mycc

    log = logger.Logger(mycc.stdout, mycc.verbose)
    itemsize = np.dtype(np.float64).itemsize
    size = comm.Get_size() if comm is not None else getattr(mycc, 'size', 1)

    def mem(nelem):
        return float(nelem) * itemsize

    t4_block_memory = mem(blksize**4 * nocc**4)
    t3_block_memory = mem(6 * blksize**2 * nvir * nocc**3)
    factor_memory = mem(blksize**4)
    ptr_table_memory = 0.0
    try:
        ptr_table_memory = float(mycc.tamps[2][0].n_abc_triplets) * np.dtype(np.uint64).itemsize
    except Exception:
        pass

    eris_memory = mem(nvir**3 * nocc + nvir * nocc**3 + 3 * nvir**2 * nocc**2 + nvir**4 + nocc**4)

    # TODO: double check this
    prefetch_memory = mem(4 * 6 * blksize**2 * nvir * nocc**3)

    w_memory = mem(12 * blksize**3 * nvir * nocc**2 + 6 * blksize**2 * nocc**4)
    runtime_peak = 2 * t4_block_memory + t3_block_memory + factor_memory + ptr_table_memory
    total_memory = eris_memory + prefetch_memory + w_memory + runtime_peak

    log.info('RCCSDT(Q) approximate per-rank memory usage estimate')
    log.info('    MPI ranks              %8d', size)
    if n_tasks is not None:
        log.info('    ABCD tasks total/local %8s', '%d/%d' % (n_tasks, local_tasks))
    log.info('    ERI copies             %8s', format_size(eris_memory))
    log.info('    T4/Z4 work blocks      %8s', format_size(2 * t4_block_memory))
    log.info('    T3 staging blocks      %8s', format_size(t3_block_memory))
    log.info('    T3 prefetch buffers    %8s', format_size(prefetch_memory))
    log.info('    W intermediates peak   %8s', format_size(w_memory))
    log.info('    Factor/pointer tables  %8s', format_size(factor_memory + ptr_table_memory))
    log.info('Total estimated per-rank   %8s', format_size(total_memory))

    max_memory = mycc.max_memory - lib.current_memory()[0]
    if (total_memory / 1024**2) > max_memory:
        logger.warn(mycc, 'Estimated per-rank memory %.2f MB exceeds available %.2f MB',
                    total_memory / 1024**2, max_memory)
    return mycc


def kernel(mycc, eris=None, tamps=None, blksize=4, comm=None, job_idx=0, n_jobs=1):
    '''Compute [Q] and (Q) correction terms using the ABC-based algorithm.
    Offsets and tasks are determined by job_idx/n_jobs for job splitting.'''
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mycc.comm = comm
    mycc.rank = rank
    mycc.size = size
    log = logger.Logger(mycc.stdout, mycc.verbose if rank == 0 else 0)
    memlog = memory_logger(mycc)
    warn_non_pytblis_backend(mycc, "RCCSDT(Q)")
    progress_thread = start_mpi_progress_thread(mycc)
    comm_log = logger.Logger(
        mycc.stdout,
        logger.DEBUG if (rank == 0 and getattr(mycc, "log_highest_t_communication", False)) else 0,
    )
    detail_log = logger.Logger(
        mycc.stdout,
        logger.DEBUG1 if (
            getattr(mycc, "log_highest_t_contractions", False)
            and (rank == 0 or getattr(mycc, "log_highest_t_contractions_all_ranks", False))
        ) else 0,
    )

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    _, t2, t3 = prepare_tamps_for_q(mycc, tamps=tamps, blksize=blksize, comm=comm)
    dt3, t3_local = t3

    mo_energy = eris.mo_energy
    e_occ = mo_energy[:nocc]
    e_occ = np.ascontiguousarray(e_occ)
    e_vir = mo_energy[nocc:]
    e_vir = np.ascontiguousarray(e_vir)

    eris_vvvo = eris.pppp[nocc:, nocc:, nocc:, :nocc].copy()
    eris_vooo = eris.pppp[nocc:, :nocc, :nocc, :nocc].copy()
    eris_vvoo = eris.pppp[nocc:, nocc:, :nocc, :nocc].copy()
    eris_ovvo = eris.pppp[:nocc, nocc:, nocc:, :nocc].copy()
    eris_ovov = eris.pppp[:nocc, nocc:, :nocc, nocc:].copy()
    eris_vvvv = eris.pppp[nocc:, nocc:, nocc:, nocc:].copy()
    eris_oooo = eris.pppp[:nocc, :nocc, :nocc, :nocc].copy()

    eris.pppp = None
    eris = None
    del eris
    gc.collect()
    log_memory(mycc, memlog, 'RCCSDT(Q) ERIs copied')

    def create_full_task_list(nvir, blksize):
        """Generate the canonical full task list (same on all ranks)"""
        tasks = []
        for d0, d1 in lib.prange(0, nvir, blksize):
            bd = d1 - d0
            for c0, c1 in lib.prange(0, d1, blksize):
                bc = c1 - c0
                for b0, b1 in lib.prange(0, c1, blksize):
                    bb = b1 - b0
                    for a0, a1 in lib.prange(0, b1, blksize):
                        ba = a1 - a0
                        tasks.append((a0, a1, ba, b0, b1, bb, c0, c1, bc, d0, d1, bd))
        return tasks

    def get_task_for_rank(rank_id, task_index, full_tasks, size, job_idx, n_jobs):
        """
        Deterministically determine what Task(task_index) corresponds to for a specific Rank(rank_id)
        using Contiguous (Chunked) Distribution.
        Returns None if no such task exists for that rank.
        """
        total_tasks = len(full_tasks)
        tasks_per_job = (total_tasks + n_jobs - 1) // n_jobs
        start_job = job_idx * tasks_per_job
        end_job = min(start_job + tasks_per_job, total_tasks)

        # Slicing within the job window
        job_total_tasks = end_job - start_job

        # Contiguous distribution:
        # tasks are split evenly among ranks: [0..k], [k+1..2k], ...
        base_tasks_per_rank = job_total_tasks // size
        remainder = job_total_tasks % size

        # Calculate start/end index for rank_id within the job
        # Rank r gets tasks [r_start, r_end) relative to start_job
        if rank_id < remainder:
            r_start = rank_id * (base_tasks_per_rank + 1)
            r_end = r_start + base_tasks_per_rank + 1
        else:
            r_start = rank_id * base_tasks_per_rank + remainder
            r_end = r_start + base_tasks_per_rank

        # Global indices
        global_r_start = start_job + r_start
        global_r_end = start_job + r_end

        # Check if requested task_index is within this rank's range
        # Note: task_index is 0-indexed relative to the rank's own list.
        # i.e. Rank 0 asks "what is my 0-th task?" -> returns global_r_start + 0

        g_idx = global_r_start + task_index
        if g_idx < global_r_end:
            return full_tasks[g_idx]
        return None


    if rank == 0:
        log.info(f"Generating full task list for splitting (Job {job_idx+1}/{n_jobs})...")

    full_task_list = create_full_task_list(nvir, blksize)
    total_tasks = len(full_task_list)

    tasks_per_job = (total_tasks + n_jobs - 1) // n_jobs
    start_job = job_idx * tasks_per_job
    end_job = min(start_job + tasks_per_job, total_tasks)
    my_job_tasks = full_task_list[start_job:end_job]

    if rank == 0:
        log.info(f"Job Splitting: Processing tasks {start_job} to {end_job} (Total {len(my_job_tasks)} out of {total_tasks})")

    # Job Splitting: Contiguous Chunking
    job_total_tasks = end_job - start_job
    base, rem = divmod(job_total_tasks, size)
    if rank < rem:
        my_start = rank * (base + 1)
        my_end = my_start + base + 1
    else:
        my_start = rank * base + rem
        my_end = my_start + base

    # Slice relative to job start
    my_tasks = my_job_tasks[my_start:my_end]

    detail_log.debug(f"  Rank {rank}: {len(my_tasks)} tasks")
    memory_estimate_log_rccsdt_q(mycc, nvir, nocc, blksize, total_tasks, len(my_tasks), comm)


    def compute_W_vvvvoo(W_vvvvoo, a0, a1, b0, b1, c0, c1):
        # 2 * blksize^3 * nocc^2 * nvir^2 + 4 * blksize^3 * nocc^3 * nvir
        ba, bb, bc = a1 - a0, b1 - b0, c1 - c0
        # W = np.empty((ba, bb, bc, nvir, nocc, nocc), dtype=t2.dtype)
        einsum('abef,fcjk->abcejk', eris_vvvv[a0:a1, b0:b1], t2[:, c0:c1], out=W_vvvvoo[:ba, :bb, :bc], alpha=0.5, beta=0.0)
        einsum('acef,fbkj->abcejk', eris_vvvv[a0:a1, c0:c1], t2[:, b0:b1], out=W_vvvvoo[:ba, :bb, :bc], alpha=0.5, beta=1.0)
        einsum('mbej,acmk->abcejk', eris_ovvo[:, b0:b1], t2[a0:a1, c0:c1], out=W_vvvvoo[:ba, :bb, :bc], alpha=-1.0, beta=1.0)
        einsum('mcek,abmj->abcejk', eris_ovvo[:, c0:c1], t2[a0:a1, b0:b1], out=W_vvvvoo[:ba, :bb, :bc], alpha=-1.0, beta=1.0)
        einsum('maje,bcmk->abcejk', eris_ovov[:, a0:a1], t2[b0:b1, c0:c1], out=W_vvvvoo[:ba, :bb, :bc], alpha=-1.0, beta=1.0)
        einsum('make,cbmj->abcejk', eris_ovov[:, a0:a1], t2[c0:c1, b0:b1], out=W_vvvvoo[:ba, :bb, :bc], alpha=-1.0, beta=1.0)
        return W_vvvvoo

    def compute_W_vvoooo(W_vvoooo, a0, a1, b0, b1):
        # 2 * blksize^2 * nocc^5
        ba, bb = a1 - a0, b1 - b0
        # W = np.empty((ba, bb, nocc, nocc, nocc, nocc), dtype=t2.dtype)
        einsum('mnki,abnj->abmijk', eris_oooo, t2[a0:a1, b0:b1], out=W_vvoooo[:ba, :bb], alpha=-0.5, beta=0.0)
        einsum('mnkj,bani->abmijk', eris_oooo, t2[b0:b1, a0:a1], out=W_vvoooo[:ba, :bb], alpha=-0.5, beta=1.0)
        return W_vvoooo

    def get_factor(a0, a1, b0, b1, c0, c1, d0, d1):
        a = np.arange(a0, a1)[:, None, None, None]
        b = np.arange(b0, b1)[None, :, None, None]
        c = np.arange(c0, c1)[None, None, :, None]
        d = np.arange(d0, d1)[None, None, None, :]

        factor_blk_ = np.zeros((a1 - a0, b1 - b0, c1 - c0, d1 - d0), dtype=np.float64)
        factor_blk = np.zeros((blksize, blksize, blksize, blksize), dtype=np.float64)

        mask_24 = (a < b) & (b < c) & (c < d)
        factor_blk_[mask_24] = 24.0

        mask_12 = (((a == b) & (b < c) & (c < d)) | ((a < b) & (b == c) & (c < d)) | ((a < b) & (b < c) & (c == d)))
        factor_blk_[mask_12] = 12.0

        mask_6 = (a == b) & (b < c) & (c == d)
        factor_blk_[mask_6] = 6.0
        factor_blk[:a1 - a0, :b1 - b0, :c1 - c0, :d1 - d0] = factor_blk_[:, :, :, :]
        return factor_blk


    def build_block_cache(dt3, nvir, blksize):
        """
        Pre-compute ownership and indices for every T3 block.
        Returns: map (A, B, C) -> (owner, indices_array) where A<=B<=C are block indices.
        """
        if rank == 0:
            log.info(f"Building Block Dependency Cache (nvir={nvir}, blksize={blksize})...")

        n_blk = (nvir + blksize - 1) // blksize
        cache = {}

        local_abc_map = dt3.abc_to_global_idx

        for A in range(n_blk):
            a_start = A * blksize
            a_end = min((A + 1) * blksize, nvir)
            for B in range(A, n_blk):
                b_start = B * blksize
                b_end = min((B + 1) * blksize, nvir)
                for C in range(B, n_blk):
                    c_start = C * blksize
                    c_end = min((C + 1) * blksize, nvir)

                    a_range = np.arange(a_start, a_end)
                    b_range = np.arange(b_start, b_end)
                    c_range = np.arange(c_start, c_end)

                    aa, bb, cc = np.meshgrid(a_range, b_range, c_range, indexing='ij')
                    aa = aa.ravel()
                    bb = bb.ravel()
                    cc = cc.ravel()

                    mask = (aa <= bb) & (bb <= cc)
                    valid_a = aa[mask]
                    valid_b = bb[mask]
                    valid_c = cc[mask]

                    if len(valid_a) == 0: continue

                    first_abc = (valid_a[0], valid_b[0], valid_c[0])
                    owner = dt3.get_owner_abc(*first_abc)

                    blk_indices = [local_abc_map.get((a,b,c)) for a,b,c in zip(valid_a, valid_b, valid_c)]
                    blk_indices = [i for i in blk_indices if i is not None]

                    if blk_indices:
                        cache[(A, B, C)] = (owner, np.array(blk_indices, dtype=np.int64))

        if rank == 0:
            log.info(f"Block Cache built. {len(cache)} blocks indexed.")
        return cache

    # Build the cache once
    block_cache = build_block_cache(dt3, nvir, blksize)
    log_memory(mycc, memlog, 'RCCSDT(Q) block cache built')
    n_blk = (nvir + blksize - 1) // blksize

    def get_task_needed_blocks(task):
        """
        Identify canonical blocks needed by the task.
        Returns an ordered tuple of canonical (A, B, C) tuples.
        """
        needed_blocks = set()
        if not task:
            return tuple()

        a0, a1, _, b0, b1, _, c0, c1, _, d0, d1, _ = task

        # Convert task orbital ranges to block indices
        # Since ranges match blksize (e.g. 0-8), they correspond to single slice of blocks.

        # Ranges:
        # P1: (a, b, :) -> Blocks A_range x B_range x All_C
        # P2: (a, c, :)
        # P3: (a, d, :)
        # P4: (b, c, :)
        # P5: (b, d, :)
        # P6: (c, d, :)

        # Helper to add ranges
        def add_range(r1, r2):
            s1, e1 = r1
            s2, e2 = r2

            # Block ranges
            blk_s1, blk_e1 = s1 // blksize, (e1 + blksize - 1) // blksize
            blk_s2, blk_e2 = s2 // blksize, (e2 + blksize - 1) // blksize

            for B1 in range(blk_s1, blk_e1):
                for B2 in range(blk_s2, blk_e2):
                    for B3 in range(n_blk):
                        # Form triplet, sort to canonical
                        triplet = tuple(sorted((B1, B2, B3)))
                        needed_blocks.add(triplet)

        add_range((a0,a1), (b0,b1))
        add_range((a0,a1), (c0,c1))
        add_range((a0,a1), (d0,d1))
        add_range((b0,b1), (c0,c1))
        add_range((b0,b1), (d0,d1))
        add_range((c0,c1), (d0,d1))

        return tuple(sorted(needed_blocks))

    needed_index_cache_size = max(8, 4 * size + 8)
    empty_index_array = np.empty(0, dtype=np.int64)
    empty_needed_by_owner = tuple(np.empty(0, dtype=np.int64) for _ in range(size))

    @functools.lru_cache(maxsize=needed_index_cache_size)
    def get_task_needed_indices_by_owner(task):
        """
        Expand a task into deterministic per-owner arrays of global abc indices.
        """
        if not task:
            return empty_needed_by_owner

        needed_chunks = [[] for _ in range(size)]
        for blk in get_task_needed_blocks(task):
            if blk not in block_cache:
                continue
            owner, indices = block_cache[blk]
            needed_chunks[owner].append(indices)

        return tuple(np.concatenate(chunks) if chunks else empty_index_array for chunks in needed_chunks)

    persistent_buffers = {'recv': [None, None], 'send': [None, None], 'idx': 0}

    def prefetch_task(task_index, task, prev_state=None):
        """
        Optimized P2P prefetch using Block Dependency Cache, Persistent Buffers,
        and a stable per-index remote cache for overlapping remote fragments.
        """
        t_prefetch_total = logger.perf_counter()
        pidx = persistent_buffers['idx']
        persistent_buffers['idx'] = 1 - persistent_buffers['idx']

        block_len = nocc * nocc * nocc
        t_need_expand = 0.0
        t_need_scan = 0.0
        t_promote_copy = 0.0
        t_send_plan = 0.0
        t_send_lookup = 0.0
        t_send_prune = 0.0
        t_recv_setup = 0.0
        t_recv_post = 0.0
        t_send_setup = 0.0
        t_send_pack = 0.0
        t_send_post = 0.0
        n_local_needed = 0
        n_remote_stable_hits = 0
        n_remote_promoted_hits = 0

        # ---------------------------------------------------------
        # 1. Identify what I need, reusing stable remote buffers directly
        #    and promoting unstable remote overlaps exactly once.
        # ---------------------------------------------------------
        req_by_owner_RECV = [empty_index_array for _ in range(size)]
        current_buffers = {}
        current_stable_remote_buffers = {}
        promote_hits = []

        if prev_state is None:
            prev_buffers = {}
            prev_stable_remote_buffers = {}
        else:
            prev_buffers = prev_state['buffers']
            prev_stable_remote_buffers = prev_state['stable_remote_buffers']

        prev_buffers_get = prev_buffers.get
        prev_stable_get = prev_stable_remote_buffers.get
        t0 = logger.perf_counter()
        my_needed_by_owner = get_task_needed_indices_by_owner(task)
        t_need_expand += logger.perf_counter() - t0

        t0 = logger.perf_counter()
        for owner, indices in enumerate(my_needed_by_owner):
            if indices.size == 0:
                continue

            if owner == rank:
                n_local_needed += indices.size
                local_offsets = dt3.global_mapping_table[indices]
                for pos, idx in enumerate(indices):
                    prev_buf = prev_buffers_get(idx)
                    if prev_buf is not None:
                        current_buffers[idx] = prev_buf
                    else:
                        current_buffers[idx] = t3_local[local_offsets[pos]]
            else:
                if not prev_buffers:
                    req_by_owner_RECV[owner] = indices
                    continue

                missing = []
                for idx in indices:
                    prev_stable_buf = prev_stable_get(idx)
                    if prev_stable_buf is not None:
                        current_buffers[idx] = prev_stable_buf
                        current_stable_remote_buffers[idx] = prev_stable_buf
                        n_remote_stable_hits += 1
                    else:
                        prev_buf = prev_buffers_get(idx)
                        if prev_buf is not None:
                            promote_hits.append((idx, prev_buf))
                            n_remote_promoted_hits += 1
                        else:
                            missing.append(idx)

                if missing:
                    req_by_owner_RECV[owner] = np.asarray(missing, dtype=np.int64)
        t_need_scan += logger.perf_counter() - t0

        # Promote unstable remote overlap into dedicated owned arrays so
        # later tasks can alias these blocks directly without another copy.
        promote_buf = None
        if promote_hits:
            t0 = logger.perf_counter()
            n_promote = len(promote_hits)
            promote_buf = np.empty((n_promote, nocc, nocc, nocc), dtype=t2.dtype)
            promote_indices = np.fromiter((idx for idx, _ in promote_hits), dtype=np.int64, count=n_promote)
            promote_src_ptrs = np.fromiter((src.ctypes.data for _, src in promote_hits), dtype=np.uintp, count=n_promote)
            promote_t3_blocks_(promote_buf, promote_src_ptrs, n_promote, block_len)
            for pos, idx in enumerate(promote_indices):
                promoted = promote_buf[pos]
                current_buffers[idx] = promoted
                current_stable_remote_buffers[idx] = promoted
            t_promote_copy += logger.perf_counter() - t0

        # ---------------------------------------------------------
        # 2. Identify what others still need from me after overlap pruning
        # ---------------------------------------------------------
        req_by_owner_SEND = [empty_index_array for _ in range(size)]

        t0 = logger.perf_counter()
        for other_rank in range(size):
            if other_rank == rank:
                continue

            t_lookup = logger.perf_counter()
            other_task = get_task_for_rank(other_rank, task_index, full_task_list, size, job_idx, n_jobs)
            t_send_lookup += logger.perf_counter() - t_lookup
            if not other_task:
                continue

            send_indices = get_task_needed_indices_by_owner(other_task)[rank]
            if task_index > 0:
                t_lookup = logger.perf_counter()
                other_prev_task = get_task_for_rank(other_rank, task_index - 1, full_task_list, size, job_idx, n_jobs)
                t_send_lookup += logger.perf_counter() - t_lookup
                if other_prev_task:
                    prev_send_indices = get_task_needed_indices_by_owner(other_prev_task)[rank]
                    if prev_send_indices.size > 0 and send_indices.size > 0:
                        t_prune = logger.perf_counter()
                        keep_mask = np.isin(send_indices, prev_send_indices, assume_unique=True, invert=True)
                        t_send_prune += logger.perf_counter() - t_prune
                        if not np.all(keep_mask):
                            send_indices = send_indices[keep_mask]

            req_by_owner_SEND[other_rank] = send_indices
        t_send_plan += logger.perf_counter() - t0

        mpi_requests = []

        # ---------------------------------------------------------
        # 4. Prepare Recv Buffers (Partitioned Persistent Buffer)
        # ---------------------------------------------------------
        t0 = logger.perf_counter()
        counts_RECV = [req_by_owner_RECV[r].size for r in range(size)]
        total_recv_blocks = sum(counts_RECV)
        displs_RECV = [sum(counts_RECV[:r]) for r in range(size)]

        required_recv_size = total_recv_blocks * block_len
        if persistent_buffers['recv'][pidx] is None or persistent_buffers['recv'][pidx].size < required_recv_size:
            persistent_buffers['recv'][pidx] = np.empty((total_recv_blocks, nocc, nocc, nocc), dtype=t2.dtype)

        recv_buf = persistent_buffers['recv'][pidx][:total_recv_blocks]
        recv_slices = {}
        t_recv_setup += logger.perf_counter() - t0

        MAX_MPI_COUNT = 2**30

        t0 = logger.perf_counter()
        if total_recv_blocks > 0:
            for r in range(size):
                if counts_RECV[r] > 0:
                    start = displs_RECV[r]
                    end = start + counts_RECV[r]
                    slice_ref = recv_buf[start:end]
                    recv_slices[r] = slice_ref

                    # Post non-blocking Recv with Overflow safety chunking
                    total_elems = counts_RECV[r] * block_len
                    slice_flat = slice_ref.reshape(-1)

                    offset = 0
                    chunk_idx = 0
                    while offset < total_elems:
                        chunk_size = min(total_elems - offset, MAX_MPI_COUNT)
                        tag = 1000 * (task_index % 10) + chunk_idx
                        req = comm.Irecv(slice_flat[offset:offset + chunk_size], source=r, tag=tag)
                        mpi_requests.append(req)
                        offset += chunk_size
                        chunk_idx += 1
        t_recv_post += logger.perf_counter() - t0

        # ---------------------------------------------------------
        # 5. Prepare Send Buffers (Partitioned Persistent Buffer)
        # ---------------------------------------------------------
        t0 = logger.perf_counter()
        counts_SEND = [req_by_owner_SEND[r].size for r in range(size)]
        total_send_blocks = sum(counts_SEND)
        displs_SEND = [sum(counts_SEND[:r]) for r in range(size)]

        required_send_size = total_send_blocks * block_len
        if persistent_buffers['send'][pidx] is None or persistent_buffers['send'][pidx].size < required_send_size:
            persistent_buffers['send'][pidx] = np.empty((total_send_blocks, nocc, nocc, nocc), dtype=t2.dtype)

        send_buf = persistent_buffers['send'][pidx][:total_send_blocks]
        t_send_setup += logger.perf_counter() - t0

        if total_send_blocks > 0:
            for r in range(size):
                if counts_SEND[r] > 0:
                    start = displs_SEND[r]
                    end = start + counts_SEND[r]

                    indices = req_by_owner_SEND[r]
                    local_block_indices = dt3.global_mapping_table[indices]
                    if not local_block_indices.flags['C_CONTIGUOUS']:
                        local_block_indices = np.ascontiguousarray(local_block_indices, dtype=np.int64)

                    t_pack = logger.perf_counter()
                    pack_t3_indices_(send_buf[start:end], t3_local, local_block_indices, counts_SEND[r], block_len)
                    t_send_pack += logger.perf_counter() - t_pack

                    # Post non-blocking Send with Overflow safety chunking
                    total_elems = counts_SEND[r] * block_len
                    slice_flat = send_buf[start:end].reshape(-1)

                    offset = 0
                    chunk_idx = 0
                    t_post = logger.perf_counter()
                    while offset < total_elems:
                        chunk_size = min(total_elems - offset, MAX_MPI_COUNT)
                        tag = 1000 * (task_index % 10) + chunk_idx
                        req = comm.Isend(slice_flat[offset:offset + chunk_size], dest=r, tag=tag)
                        mpi_requests.append(req)
                        offset += chunk_size
                        chunk_idx += 1
                    t_send_post += logger.perf_counter() - t_post

        prefetch_handle = {
            'requests': mpi_requests,
            'recv_slices': recv_slices,
            'req_by_owner': req_by_owner_RECV,
            'current_buffers': current_buffers,
            'stable_remote_buffers': current_stable_remote_buffers,
            'promote_buf_master': promote_buf,
            'recv_buf_master': recv_buf,
            'send_buf_master': send_buf
        }

        total_prefetch_time = logger.perf_counter() - t_prefetch_total
        comm_log.debug(
            f"        Rank {rank} Task {task_index+1}: prefetch breakdown total={total_prefetch_time:.4f} sec "
            f"need_expand={t_need_expand:.4f} need_scan={t_need_scan:.4f} "
            f"promote_copy={t_promote_copy:.4f} "
            f"send_plan={t_send_plan:.4f} (lookup={t_send_lookup:.4f} prune={t_send_prune:.4f}) "
            f"recv_setup={t_recv_setup:.4f} recv_post={t_recv_post:.4f} "
            f"send_setup={t_send_setup:.4f} send_pack={t_send_pack:.4f} send_post={t_send_post:.4f} "
            f"local_needed={n_local_needed} stable_hits={n_remote_stable_hits} promoted_hits={n_remote_promoted_hits} "
            f"recv_blocks={total_recv_blocks} send_blocks={total_send_blocks}"
        )
        if progress_thread is not None:
            progress_thread.add_requests(mpi_requests)
        return prefetch_handle

    def finalize_prefetch_task(handle):
        if not handle:
            return {'buffers': {}, 'stable_remote_buffers': {}}
        t0_wait = logger.perf_counter()
        if progress_thread is not None:
            progress_thread.clear_requests()
        MPI.Request.Waitall(handle['requests'])
        comm_log.debug(f"        Rank {rank} Task {ti+1}: MPI.Waitall time: {logger.perf_counter()-t0_wait:.4f} sec.")
        t0_wait = logger.perf_counter()

        req_by_owner = handle['req_by_owner']
        recv_slices = handle['recv_slices']
        current_buffers = handle['current_buffers']
        current_stable_remote_buffers = handle['stable_remote_buffers']

        for r, indices in enumerate(req_by_owner):
            if indices.size > 0:
                buf = recv_slices[r]
                for i, abc_idx in enumerate(indices):
                    current_buffers[abc_idx] = buf[i]

        comm_log.debug(f"        Rank {rank} Task {ti+1}: Unpack time: {logger.perf_counter()-t0_wait:.4f} sec.")
        return {'buffers': current_buffers, 'stable_remote_buffers': current_stable_remote_buffers}

    if rank == 0:
        log.info("Starting RCCSDT(Q) correction computation with ABC-based algorithm (Single Task Deterministic)...")

    t4_blk = np.zeros((blksize,) * 4 + (nocc,) * 4, dtype=t2.dtype)
    z4_p_blk = np.zeros_like(t4_blk)
    t3_blks = [np.empty((blksize,) * 2 + (nvir,) + (nocc,) * 3, dtype=t2.dtype) for _ in range(6)]
    W_vvvvoo = np.empty((blksize,) * 3 + (nvir,) + (nocc,) * 2, dtype=t2.dtype)
    W_vvoooo = np.empty((blksize,) * 2 + (nocc,) * 4, dtype=t2.dtype)
    log_memory(mycc, memlog, 'RCCSDT(Q) task buffers allocated')

    e_q_bracket = 0.0
    e_q_paren = 0.0

    if rank == 0:
        log.info("  Beginning task processing with P2P prefetching...")

    if comm.rank == 0:
        time2 = logger.process_clock(), logger.perf_counter()

    # Process Tasks
    if rank == 0:
        log.info("  Processing tasks...")

    # Pre-allocate pointer table for C function (reused)
    ptr_table = np.zeros(dt3.n_abc_triplets, dtype=np.uint64)

    tasks_len = len(my_tasks)
    global_max_tasks = comm.allreduce(tasks_len, op=MPI.MAX)

    # Initial Prefetch (Task 0)
    next_handle = None
    first_task = my_tasks[0] if tasks_len > 0 else None
    next_handle = prefetch_task(0, first_task, None) # Prefetch task 0

    for ti in range(global_max_tasks):

        detail_log.debug(f"    Rank {rank} Processing task {ti+1}/{global_max_tasks}...")

        # 1. Finalize
        t0_wait = logger.perf_counter()
        task_state = finalize_prefetch_task(next_handle)
        task_buffers = task_state['buffers']
        comm_log.debug(f"    Rank {rank} Task {ti+1}: prefetch finalize time: {logger.perf_counter()-t0_wait:.4f} sec.")

        # 2. Prefetch Next
        future_handle = None
        if ti + 1 < global_max_tasks:
            # Task ti+1
            next_task = my_tasks[ti+1] if ti + 1 < tasks_len else None
            t0_issue = logger.perf_counter()
            # Reuse buffers from previous task if possible
            future_handle = prefetch_task(ti + 1, next_task, task_state)
            comm_log.debug(f"    Rank {rank} Task {ti+1}: prefetch issue time: {logger.perf_counter()-t0_issue:.4f} sec.")

        # 3. Compute
        if ti < tasks_len:
            task = my_tasks[ti]

            ptr_table.fill(0)
            if task_buffers:
                indices = np.fromiter(task_buffers.keys(), dtype=np.int64, count=len(task_buffers))
                addrs = np.fromiter((b.ctypes.data for b in task_buffers.values()), dtype=np.uint64,
                                    count=len(task_buffers))
                ptr_table[indices] = addrs

            # Helper to fill t3 using C
            def fill_t3_p2p(t3_blk_target, a0, a1, b0, b1, buffer_map):
                fill_t3_from_ptr_array(t3_blk_target, a0, a1, b0, b1, nvir, nocc, ptr_table,
                                        blksize, t3_blk_target.size, ptr_table.size)

            (a0, a1, ba, b0, b1, bb, c0, c1, bc, d0, d1, bd) = task
            detail_log.debug(f"    Rank {rank} show task {ti+1}: a[{a0}:{a1}] b[{b0}:{b1}] c[{c0}:{c1}] d[{d0}:{d1}]")
            detail_log.debug(f"    Rank {rank} memory used: {lib.current_memory()[0]} MB")
            detail_log.debug(f"    Rank {rank} memory available: {mycc.max_memory - lib.current_memory()[0]} MB")

            fill_t3_p2p(t3_blks[0], a0, a1, b0, b1, task_buffers)
            punctuate_mpi_progress(mycc, progress_thread)
            t3_blk = t3_blks[0]
            einsum('cdel,abeijk->abcdijkl', eris_vvvo[c0:c1, d0:d1], t3_blk[:ba, :bb], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=0.0)
            einsum('dcek,abeijl->abcdijkl', eris_vvvo[d0:d1, c0:c1], t3_blk[:ba, :bb], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('dmli,abcmjk->abcdijkl', eris_vooo[d0:d1], t3_blk[:ba, :bb, c0:c1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('dmlj,abcimk->abcdijkl', eris_vooo[d0:d1], t3_blk[:ba, :bb, c0:c1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('cmki,abdmjl->abcdijkl', eris_vooo[c0:c1], t3_blk[:ba, :bb, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('cmkj,abdiml->abcdijkl', eris_vooo[c0:c1], t3_blk[:ba, :bb, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            fill_t3_p2p(t3_blks[1], a0, a1, c0, c1, task_buffers)
            punctuate_mpi_progress(mycc, progress_thread)
            t3_blk = t3_blks[1]
            einsum('bdel,aceikj->abcdijkl', eris_vvvo[b0:b1, d0:d1], t3_blk[:ba, :bc], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('dbej,aceikl->abcdijkl', eris_vvvo[d0:d1, b0:b1], t3_blk[:ba, :bc], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('bmji,acdmkl->abcdijkl', eris_vooo[b0:b1], t3_blk[:ba, :bc, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('bmjk,acdiml->abcdijkl', eris_vooo[b0:b1], t3_blk[:ba, :bc, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('dmlk,acbimj->abcdijkl', eris_vooo[d0:d1], t3_blk[:ba, :bc, b0:b1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)

            fill_t3_p2p(t3_blks[2], a0, a1, d0, d1, task_buffers)
            punctuate_mpi_progress(mycc, progress_thread)
            t3_blk = t3_blks[2]
            einsum('bcek,adeilj->abcdijkl', eris_vvvo[b0:b1, c0:c1], t3_blk[:ba, :bd], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('cbej,adeilk->abcdijkl', eris_vvvo[c0:c1, b0:b1], t3_blk[:ba, :bd], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('bmjl,adcimk->abcdijkl', eris_vooo[b0:b1], t3_blk[:ba, :bd, c0:c1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('cmkl,adbimj->abcdijkl', eris_vooo[c0:c1], t3_blk[:ba, :bd, b0:b1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            fill_t3_p2p(t3_blks[3], b0, b1, c0, c1, task_buffers)
            punctuate_mpi_progress(mycc, progress_thread)
            t3_blk = t3_blks[3]
            einsum('adel,bcejki->abcdijkl', eris_vvvo[a0:a1, d0:d1], t3_blk[:bb, :bc], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('daei,bcejkl->abcdijkl', eris_vvvo[d0:d1, a0:a1], t3_blk[:bb, :bc], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('amij,bcdmkl->abcdijkl', eris_vooo[a0:a1], t3_blk[:bb, :bc, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('amik,bcdjml->abcdijkl', eris_vooo[a0:a1], t3_blk[:bb, :bc, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            fill_t3_p2p(t3_blks[4], b0, b1, d0, d1, task_buffers)
            punctuate_mpi_progress(mycc, progress_thread)
            t3_blk = t3_blks[4]
            einsum('acek,bdejli->abcdijkl', eris_vvvo[a0:a1, c0:c1], t3_blk[:bb, :bd], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('caei,bdejlk->abcdijkl', eris_vvvo[c0:c1, a0:a1], t3_blk[:bb, :bd], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('amil,bdcjmk->abcdijkl', eris_vooo[a0:a1], t3_blk[:bb, :bd, c0:c1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)

            fill_t3_p2p(t3_blks[5], c0, c1, d0, d1, task_buffers)
            punctuate_mpi_progress(mycc, progress_thread)
            t3_blk = t3_blks[5]
            einsum('abej,cdekli->abcdijkl', eris_vvvo[a0:a1, b0:b1], t3_blk[:bc, :bd], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('baei,cdeklj->abcdijkl', eris_vvvo[b0:b1, a0:a1], t3_blk[:bc, :bd], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            # t4
            compute_W_vvvvoo(W_vvvvoo, a0, a1, b0, b1, c0, c1)
            einsum('abcejk,edil->abcdijkl', W_vvvvoo[:ba, :bb, :bc], t2[:, d0:d1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=0.0)
            compute_W_vvvvoo(W_vvvvoo, a0, a1, b0, b1, d0, d1)
            einsum('abdejl,ecik->abcdijkl', W_vvvvoo[:ba, :bb, :bd], t2[:, c0:c1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvvvoo(W_vvvvoo, a0, a1, c0, c1, d0, d1)
            einsum('acdekl,ebij->abcdijkl', W_vvvvoo[:ba, :bc, :bd], t2[:, b0:b1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            compute_W_vvvvoo(W_vvvvoo, b0, b1, a0, a1, c0, c1)
            einsum('baceik,edjl->abcdijkl', W_vvvvoo[:bb, :ba, :bc], t2[:, d0:d1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvvvoo(W_vvvvoo, b0, b1, a0, a1, d0, d1)
            einsum('badeil,ecjk->abcdijkl', W_vvvvoo[:bb, :ba, :bd], t2[:, c0:c1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            compute_W_vvvvoo(W_vvvvoo, b0, b1, c0, c1, d0, d1)
            einsum('bcdekl,eaji->abcdijkl', W_vvvvoo[:bb, :bc, :bd], t2[:, a0:a1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvvvoo(W_vvvvoo, c0, c1, a0, a1, b0, b1)
            einsum('cabeij,edkl->abcdijkl', W_vvvvoo[:bc, :ba, :bb], t2[:, d0:d1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            compute_W_vvvvoo(W_vvvvoo, c0, c1, a0, a1, d0, d1)
            einsum('cadeil,ebkj->abcdijkl', W_vvvvoo[:bc, :ba, :bd], t2[:, b0:b1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvvvoo(W_vvvvoo, c0, c1, b0, b1, d0, d1)
            einsum('cbdejl,eaki->abcdijkl', W_vvvvoo[:bc, :bb, :bd], t2[:, a0:a1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            compute_W_vvvvoo(W_vvvvoo, d0, d1, a0, a1, b0, b1)
            einsum('dabeij,eclk->abcdijkl', W_vvvvoo[:bd, :ba, :bb], t2[:, c0:c1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvvvoo(W_vvvvoo, d0, d1, a0, a1, c0, c1)
            einsum('daceik,eblj->abcdijkl', W_vvvvoo[:bd, :ba, :bc], t2[:, b0:b1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            compute_W_vvvvoo(W_vvvvoo, d0, d1, b0, b1, c0, c1)
            einsum('dbcejk,eali->abcdijkl', W_vvvvoo[:bd, :bb, :bc], t2[:, a0:a1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            # 6 * (4 * O^4V + 2 * O^5) + 12 * O^5 = 24 * O^4V + 24 * O^5
            compute_W_vvoooo(W_vvoooo, a0, a1, b0, b1)
            einsum('abmijk,cdml->abcdijkl', W_vvoooo[:ba, :bb], t2[c0:c1, d0:d1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('abmijl,dcmk->abcdijkl', W_vvoooo[:ba, :bb], t2[d0:d1, c0:c1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvoooo(W_vvoooo, a0, a1, c0, c1)
            einsum('acmikj,bdml->abcdijkl', W_vvoooo[:ba, :bc], t2[b0:b1, d0:d1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('acmikl,dbmj->abcdijkl', W_vvoooo[:ba, :bc], t2[d0:d1, b0:b1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvoooo(W_vvoooo, a0, a1, d0, d1)
            einsum('admilj,bcmk->abcdijkl', W_vvoooo[:ba, :bd], t2[b0:b1, c0:c1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('admilk,cbmj->abcdijkl', W_vvoooo[:ba, :bd], t2[c0:c1, b0:b1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvoooo(W_vvoooo, b0, b1, c0, c1)
            einsum('bcmjki,adml->abcdijkl', W_vvoooo[:bb, :bc], t2[a0:a1, d0:d1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('bcmjkl,dami->abcdijkl', W_vvoooo[:bb, :bc], t2[d0:d1, a0:a1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvoooo(W_vvoooo, b0, b1, d0, d1)
            einsum('bdmjli,acmk->abcdijkl', W_vvoooo[:bb, :bd], t2[a0:a1, c0:c1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('bdmjlk,cami->abcdijkl', W_vvoooo[:bb, :bd], t2[c0:c1, a0:a1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            compute_W_vvoooo(W_vvoooo, c0, c1, d0, d1)
            einsum('cdmkli,abmj->abcdijkl', W_vvoooo[:bc, :bd], t2[a0:a1, b0:b1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            einsum('cdmklj,bami->abcdijkl', W_vvoooo[:bc, :bd], t2[b0:b1, a0:a1], out=t4_blk[:ba, :bb, :bc, :bd], alpha=-1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            t4_add_(t4_blk, z4_p_blk, blksize**4, nocc)
            e_abcdijkl_division_(t4_blk, e_occ, e_vir, a0, a1, b0, b1, c0, c1, d0, d1, blksize, blksize, blksize, blksize, nocc)
            t4_spin_summation_inplace_(t4_blk, blksize**4, nocc, 'P4_444', 1.0, 0.0)
            punctuate_mpi_progress(mycc, progress_thread)

            factor_blk = get_factor(a0, a1, b0, b1, c0, c1, d0, d1)
            t4_multiply_factor_(t4_blk, factor_blk, blksize, blksize, blksize, blksize, nocc)
            punctuate_mpi_progress(mycc, progress_thread)

            tmp_paren = np.dot(t4_blk.ravel(), z4_p_blk.ravel()) / 12.0

            # z4_sq: 6 * O^4
            einsum('abij,cdkl->abcdijkl', eris_vvoo[a0:a1, b0:b1], t2[c0:c1, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=0.0)
            einsum('acik,bdjl->abcdijkl', eris_vvoo[a0:a1, c0:c1], t2[b0:b1, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('adil,bcjk->abcdijkl', eris_vvoo[a0:a1, d0:d1], t2[b0:b1, c0:c1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('bcjk,adil->abcdijkl', eris_vvoo[b0:b1, c0:c1], t2[a0:a1, d0:d1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)
            einsum('bdjl,acik->abcdijkl', eris_vvoo[b0:b1, d0:d1], t2[a0:a1, c0:c1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            einsum('cdkl,abij->abcdijkl', eris_vvoo[c0:c1, d0:d1], t2[a0:a1, b0:b1], out=z4_p_blk[:ba, :bb, :bc, :bd], alpha=1.0, beta=1.0)
            punctuate_mpi_progress(mycc, progress_thread)

            tmp_bracket = np.dot(t4_blk.ravel(), z4_p_blk.ravel()) / 12.0
            tmp_paren += tmp_bracket

            detail_log.debug1(f"[Q] Task {ti+1} Rank {rank} A[{a0:3d}:{a1:3d}] B[{b0:3d}:{b1:3d}] C[{c0:3d}:{c1:3d}] D[{d0:3d}:{d1:3d}]: {tmp_bracket: .16e}")
            detail_log.debug1(f"(Q) Task {ti+1} Rank {rank} A[{a0:3d}:{a1:3d}] B[{b0:3d}:{b1:3d}] C[{c0:3d}:{c1:3d}] D[{d0:3d}:{d1:3d}]: {tmp_paren: .16e}")
            log_memory(mycc, memlog, 'RCCSDT(Q) task %d' % (ti + 1), per_iter=True)

            e_q_bracket += tmp_bracket
            e_q_paren += tmp_paren

            if comm.rank == 0:
                time2 = detail_log.timer_debug1('CCSDT(Q): iter [%3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d]:' % (
                    a0, a1, b0, b1, c0, c1, d0, d1), *time2)

        # Advance
        next_handle = future_handle

        if (ti + 1) % 50 == 0:
            comm.Barrier()

    comm.Barrier()

    e_q_bracket = np.array(e_q_bracket, dtype=np.float64)
    e_q_paren = np.array(e_q_paren, dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, e_q_bracket, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, e_q_paren, op=MPI.SUM)

    log_memory(mycc, memlog, 'RCCSDT(Q) reductions')

    if comm.rank == 0:
        log.info("[Q] correction = % .16e    (Q) correction = % .16e" % (e_q_bracket, e_q_paren))
        log.timer('RCCSDT(Q)', *time0)

    if progress_thread is not None:
        progress_thread.stop()
    return e_q_bracket, e_q_paren


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.data.elements import chemcore
    from distr_cc.rccsdt import RCCSDT

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    log = logger.Logger(sys.stdout, 6)

    atom = '''
    O  1.416468653903   0.111264435953   0.000000000000
    H  1.746241653903  -0.373945564047  -0.758561000000
    H  2.102765241     -0.898304829      1.578786622
    '''
    basis = 'cc-pvdz'

    mol = gto.M(atom=atom, basis=basis)
    mol.verbose = 1
    mol.max_memory = 10000
    frozen = chemcore(mol)

    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()

    mycc = RCCSDT(mf, frozen=frozen, comm=comm)
    # mycc.set_einsum_backend('pytblis')
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-6
    mycc.max_cycle = 100
    mycc.max_memory = 10000
    mycc.verbose = 1
    mycc.batch_size = 11
    mycc.do_diis_max_t = False
    mycc.incore_complete = True
    ecorr, tamps = mycc.kernel()

    q_bracket = -0.001462052703
    q_paren = -0.001620887567

    mycc.verbose = 8
    q_bracket, q_paren = kernel(mycc, tamps=tamps, blksize=8, comm=comm, job_idx=0, n_jobs=1)
    if rank == 0:
        print('SQ corr: % .12f    Ref: % .12f    Diff: % .12e'%(q_bracket, q_bracket, q_bracket - q_bracket))
        print('PQ corr: % .12f    Ref: % .12f    Diff: % .12e'%(q_paren, q_paren, q_paren - q_paren))
