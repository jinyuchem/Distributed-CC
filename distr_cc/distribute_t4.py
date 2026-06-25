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
#

"""
Distributed T4 amplitude storage with (i,j,k,l) quadruple distribution.

The RCCSDTQ triangular T4 amplitudes are stored as

    T4[i <= j <= k <= l, a, b, c, d]

where each canonical occupied quadruple is assigned to exactly one MPI rank.
All other tensors in the RCCSDTQ implementation are expected to be replicated.
"""

import json
from itertools import permutations
import numpy as np
from mpi4py import MPI
from pyscf.lib import logger
from distr_cc._runtime import require_native_c, warn_python_fallback

_NATIVE_IMPORT_ERROR = None

try:
    from distr_cc._c_rccsdtq import fill_local_data_ijkl_, unpack_t4_ijkl_single_
    _HAS_C_LIB = True
except (OSError, AttributeError) as err:
    _NATIVE_IMPORT_ERROR = err
    _HAS_C_LIB = False
    warn_python_fallback("distributed RCCSDTQ T4 helpers", err=err)


def _require_t4_c(feature, obj=None):
    return require_native_c(_HAS_C_LIB, feature, obj=obj, err=_NATIVE_IMPORT_ERROR)

def _comb2(n):
    return n * (n - 1) // 2 if n >= 2 else 0

def _comb3(n):
    return n * (n - 1) * (n - 2) // 6 if n >= 3 else 0

def _comb4(n):
    return n * (n - 1) * (n - 2) * (n - 3) // 24 if n >= 4 else 0

def ijkl_to_linear(i, j, k, l, nocc):
    n_before_i = _comb4(nocc + 3) - _comb4(nocc - i + 3)
    n_before_j = _comb3(nocc - i + 2) - _comb3(nocc - j + 2)
    n_before_k = _comb2(nocc - j + 1) - _comb2(nocc - k + 1)
    return n_before_i + n_before_j + n_before_k + (l - k)

def ijkl_is_zero_block(i, j, k, l):
    canonical = tuple(sorted((int(i), int(j), int(k), int(l))))
    return canonical[0] == canonical[1] == canonical[2] or canonical[1] == canonical[2] == canonical[3]

def linear_to_ijkl(idx, nocc):
    remaining = int(idx)
    i = 0
    while i < nocc:
        n_i = (nocc - i) * (nocc - i + 1) * (nocc - i + 2) // 6
        if remaining < n_i:
            break
        remaining -= n_i
        i += 1
    j = i
    while j < nocc:
        n_j = (nocc - j) * (nocc - j + 1) // 2
        if remaining < n_j:
            break
        remaining -= n_j
        j += 1
    k = j
    while k < nocc:
        n_k = nocc - k
        if remaining < n_k:
            break
        remaining -= n_k
        k += 1
    l = k + remaining
    return i, j, k, l

def enumerate_ijkl_quadruples(nocc):
    quadruples = []
    for i in range(nocc):
        for j in range(i, nocc):
            for k in range(j, nocc):
                for l in range(k, nocc):
                    quadruples.append((i, j, k, l))
    return quadruples

def enumerate_stored_ijkl_quadruples(nocc):
    return [quad for quad in enumerate_ijkl_quadruples(nocc) if not ijkl_is_zero_block(*quad)]

def _canonical_and_perm(i, j, k, l):
    values = (int(i), int(j), int(k), int(l))
    order = sorted(range(4), key=lambda p: (values[p], p))
    canonical = tuple(values[p] for p in order)
    perm = tuple(order.index(p) for p in range(4))
    return canonical, perm

def _as_canonical_quadruples(ijkl_quadruples):
    ijkl_quadruples = np.asarray(ijkl_quadruples, dtype=np.int32)
    if ijkl_quadruples.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    if ijkl_quadruples.ndim != 2 or ijkl_quadruples.shape[1] != 4:
        raise ValueError("ijkl_quadruples must have shape (n_quad, 4)")
    if np.any(ijkl_quadruples[:, 0] > ijkl_quadruples[:, 1]) or np.any(ijkl_quadruples[:, 1] > ijkl_quadruples[:, 2]) \
        or np.any(ijkl_quadruples[:, 2] > ijkl_quadruples[:, 3]):
        raise ValueError("ijkl_quadruples must be canonical: i <= j <= k <= l")
    return np.ascontiguousarray(ijkl_quadruples)

def _unique_permutations(values):
    return tuple(dict.fromkeys(permutations(tuple(int(x) for x in values))))


class DistributedT4IJKL:
    """
    Distributed triangular T4 amplitude storage with occupied quadruple ownership.

    Each canonical quadruple (i <= j <= k <= l) is assigned to one rank.  Local
    storage has shape ``(local_nocc4, nvir, nvir, nvir, nvir)``.  Blocks with
    i == j == k or j == k == l are omitted from storage because their T4
    amplitudes are identically zero.  ``nocc4_full`` is the full triangular
    count, while ``nocc4`` is the compact stored count.
    """

    def __init__(self, nocc, nvir, comm, distribution="balanced", batch_size=None, dtype=np.float64,
                 allow_python_fallback=None):
        self.nocc = int(nocc)
        self.nvir = int(nvir)
        self.dtype = np.dtype(dtype)
        self.comm = comm
        self.distribution = distribution
        self.batch_size = batch_size
        self.allow_python_fallback = allow_python_fallback

        self.nocc4_full = self.nocc * (self.nocc + 1) * (self.nocc + 2) * (self.nocc + 3) // 24
        self.nocc4_zero = self.nocc * self.nocc
        self.nocc4 = self.nocc4_full - self.nocc4_zero

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.log_t4_communication = False

        self._compute_distribution(distribution, batch_size=batch_size)
        self._build_local_mapping()

        self.log = None

    def _log_communication(self, message):
        if not self.log_t4_communication:
            return
        print(f"        Rank {self.rank} {message}", flush=True)

    def _enumerate_ijkl_quadruples(self):
        return enumerate_stored_ijkl_quadruples(self.nocc)

    def _compute_distribution(self, strategy, batch_size=None):
        all_quadruples = self._enumerate_ijkl_quadruples()
        self._assign_rank_quadruples(all_quadruples, strategy, batch_size=batch_size)
        self._finalize_distribution(all_quadruples)

    def _assign_rank_quadruples(self, all_quadruples, strategy, batch_size=None):
        """Assign canonical stored quadruples to ranks without MPI finalization."""
        if strategy == "round_robin":
            self._distribute_round_robin(all_quadruples)
        elif strategy == "block":
            self._distribute_block(all_quadruples)
        elif strategy == "balanced":
            if batch_size is not None:
                self._distribute_balanced_per_batch(all_quadruples, batch_size)
            else:
                self._distribute_round_robin(all_quadruples)
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")

    def _distribute_round_robin(self, all_quadruples):
        self.rank_quadruples = [[] for _ in range(self.size)]
        for idx, quadruple in enumerate(all_quadruples):
            owner = idx % self.size
            self.rank_quadruples[owner].append(quadruple)

    def _distribute_block(self, all_quadruples):
        n = len(all_quadruples)
        base_size = n // self.size
        remainder = n % self.size

        self.rank_quadruples = [[] for _ in range(self.size)]
        idx = 0
        for r in range(self.size):
            count = base_size + (1 if r < remainder else 0)
            for _ in range(count):
                if idx < n:
                    self.rank_quadruples[r].append(all_quadruples[idx])
                    idx += 1

    def _distribute_balanced_per_batch(self, all_quadruples, batch_size):
        n_quad = len(all_quadruples)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if n_quad == 0:
            self.rank_quadruples = [[] for _ in range(self.size)]
            self._rank_batch_costs = np.zeros((self.size, 0), dtype=np.float64)
            return

        if self.rank != 0:
            self.rank_quadruples = [[] for _ in range(self.size)]
            return

        base_count = n_quad // self.size
        remainder = n_quad % self.size
        target_counts = np.array([base_count + (1 if r < remainder else 0) for r in range(self.size)], dtype=np.int64)

        all_quadruples_arr = np.asarray(all_quadruples, dtype=np.int32).reshape(-1, 4)
        batches = [all_quadruples_arr[start:min(start + batch_size, n_quad)] for start in range(0, n_quad, batch_size)]

        quadruple_batch_costs = self._compute_cost_matrix(all_quadruples_arr, batches)
        quadruple_to_rank = self._initial_assignment_balanced(quadruple_batch_costs, target_counts)
        quadruple_to_rank = self._refine_assignment(quadruple_to_rank, quadruple_batch_costs, target_counts,
                                                    max_iterations=100)

        self.rank_quadruples = [[] for _ in range(self.size)]
        for q_idx, quadruple in enumerate(all_quadruples):
            self.rank_quadruples[int(quadruple_to_rank[q_idx])].append(quadruple)

        rank_batch_costs = np.zeros((self.size, len(batches)), dtype=np.float64)
        for q_idx, rank in enumerate(quadruple_to_rank):
            rank_batch_costs[int(rank)] += quadruple_batch_costs[q_idx]

        self._rank_batch_costs = rank_batch_costs

    def _compute_cost_matrix(self, all_quadruples_arr, batches):
        n_quad = len(all_quadruples_arr)
        n_batches = len(batches)
        quadruple_batch_costs = np.zeros((n_quad, n_batches), dtype=np.float64)

        for b_idx, batch_ijkl in enumerate(batches):
            batch_cost = np.zeros(n_quad, dtype=np.float64)
            for batch_quadruple in batch_ijkl:
                batch_cost += self._compute_batch_quadruple_cost(all_quadruples_arr, batch_quadruple)
            quadruple_batch_costs[:, b_idx] = batch_cost

        return quadruple_batch_costs

    def _compute_batch_quadruple_cost(self, local_quadruples, batch_quadruple):
        local_cols = [local_quadruples[:, axis] for axis in range(4)]
        batch_quadruple = tuple(int(x) for x in batch_quadruple)
        unique_perms = _unique_permutations(batch_quadruple)

        cost = np.zeros(len(local_quadruples), dtype=np.float64)
        cost_oo = 1.0
        cost_ovov = float(self.nvir)

        # F_oo[m,p] * T4[m,...] and W_oooo[m,n,p,q] * T4[...,m,n]
        for requested in unique_perms:
            tail = requested[1:]
            head = requested[:2]

            for p in range(4):
                comp = tuple(axis for axis in range(4) if axis != p)
                mask = ((local_cols[comp[0]] == tail[0]) & (local_cols[comp[1]] == tail[1])
                         & (local_cols[comp[2]] == tail[2]))
                cost += cost_oo * mask

            for p, q in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                comp = tuple(axis for axis in range(4) if axis not in (p, q))
                mask = ((local_cols[comp[0]] == head[0]) & (local_cols[comp[1]] == head[1]))
                cost += cost_oo * mask

        # Explicit W_ovov/W_voov contractions in compute_oooo_oovv_contraction_.
        # Removing the first batch index gives five contractions; the other
        # unique removal positions give seven contractions each.
        for omitted_axis in range(4):
            comp = tuple(axis for axis in range(4) if axis != omitted_axis)
            for remove_pos in range(4):
                if remove_pos > 0 and batch_quadruple[remove_pos] == batch_quadruple[remove_pos - 1]:
                    continue
                remaining = tuple(batch_quadruple[pos] for pos in range(4) if pos != remove_pos)
                n_terms = 5.0 if remove_pos == 0 else 7.0
                mask = ((local_cols[comp[0]] == remaining[0]) & (local_cols[comp[1]] == remaining[1])
                         & (local_cols[comp[2]] == remaining[2]))
                cost += cost_ovov * n_terms * mask
        return cost

    def _initial_assignment_balanced(self, quadruple_batch_costs, target_counts):
        n_quad, n_batches = quadruple_batch_costs.shape
        quadruple_to_rank = np.full(n_quad, -1, dtype=np.int32)
        rank_counts = np.zeros(self.size, dtype=np.int64)
        rank_batch_costs = np.zeros((self.size, n_batches), dtype=np.float64)

        max_impact = quadruple_batch_costs.max(axis=1)
        total_costs = quadruple_batch_costs.sum(axis=1)
        sort_key = max_impact * 1000.0 + total_costs
        sorted_indices = np.lexsort((np.arange(n_quad, dtype=np.int64), -sort_key))

        for q_idx in sorted_indices:
            quadruple_costs = quadruple_batch_costs[q_idx]
            valid_ranks = [r for r in range(self.size) if rank_counts[r] < target_counts[r]]
            if not valid_ranks:
                valid_ranks = list(range(self.size))

            best_rank = None
            best_score = (np.inf, np.inf)
            for r in valid_ranks:
                new_costs = rank_batch_costs[r] + quadruple_costs
                score = (new_costs.max(), new_costs.sum())
                if score < best_score:
                    best_score = score
                    best_rank = r

            quadruple_to_rank[q_idx] = best_rank
            rank_counts[best_rank] += 1
            rank_batch_costs[best_rank] += quadruple_costs

        return quadruple_to_rank

    def _refine_assignment(self, quadruple_to_rank, quadruple_batch_costs, target_counts, max_iterations=100):
        n_quad, n_batches = quadruple_batch_costs.shape

        rank_batch_costs = np.zeros((self.size, n_batches), dtype=np.float64)
        rank_quadruple_indices = [[] for _ in range(self.size)]
        for q_idx in range(n_quad):
            rank = int(quadruple_to_rank[q_idx])
            rank_batch_costs[rank] += quadruple_batch_costs[q_idx]
            rank_quadruple_indices[rank].append(q_idx)

        def objective(costs):
            mean_per_batch = costs.mean(axis=0)
            mean_per_batch[mean_per_batch == 0.0] = 1.0
            return (costs / mean_per_batch[None, :]).max()

        current_obj = objective(rank_batch_costs)
        tol = 1e-5

        for _ in range(max_iterations):
            mean_per_batch = rank_batch_costs.mean(axis=0)
            mean_per_batch[mean_per_batch == 0.0] = 1.0
            relative_load = rank_batch_costs / mean_per_batch[None, :]

            flat_idx = np.argmax(relative_load)
            overloaded_rank, worst_batch = np.unravel_index(flat_idx, relative_load.shape)
            target_ranks = np.lexsort((np.arange(self.size, dtype=np.int64), rank_batch_costs[:, worst_batch]))[:3]

            candidates_from_overloaded = sorted(rank_quadruple_indices[overloaded_rank],
                                                key=lambda q: (-quadruple_batch_costs[q, worst_batch], q))[:20]

            best_swap = None
            best_new_obj = current_obj

            for q1_idx in candidates_from_overloaded:
                q1_costs = quadruple_batch_costs[q1_idx]
                for target_rank in target_ranks:
                    target_rank = int(target_rank)
                    if target_rank == overloaded_rank:
                        continue

                    candidates_from_target = sorted(rank_quadruple_indices[target_rank],
                                                    key=lambda q: (quadruple_batch_costs[q, worst_batch], q))[:5]

                    for q2_idx in candidates_from_target:
                        q2_costs = quadruple_batch_costs[q2_idx]
                        overloaded_new = rank_batch_costs[overloaded_rank] - q1_costs + q2_costs
                        target_new = rank_batch_costs[target_rank] - q2_costs + q1_costs

                        if (overloaded_new[worst_batch] / mean_per_batch[worst_batch] >= current_obj
                            or target_new[worst_batch] / mean_per_batch[worst_batch] >= current_obj):
                            continue

                        trial_costs = rank_batch_costs.copy()
                        trial_costs[overloaded_rank] = overloaded_new
                        trial_costs[target_rank] = target_new
                        new_obj = objective(trial_costs)

                        if new_obj < best_new_obj - tol:
                            best_new_obj = new_obj
                            best_swap = (q1_idx, q2_idx, overloaded_rank, target_rank)

            if best_swap is None:
                break

            q1_idx, q2_idx, rank1, rank2 = best_swap
            q1_costs = quadruple_batch_costs[q1_idx]
            q2_costs = quadruple_batch_costs[q2_idx]

            quadruple_to_rank[q1_idx] = rank2
            quadruple_to_rank[q2_idx] = rank1

            rank_quadruple_indices[rank1].remove(q1_idx)
            rank_quadruple_indices[rank1].append(q2_idx)
            rank_quadruple_indices[rank2].remove(q2_idx)
            rank_quadruple_indices[rank2].append(q1_idx)

            rank_batch_costs[rank1] += q2_costs - q1_costs
            rank_batch_costs[rank2] += q1_costs - q2_costs
            current_obj = best_new_obj

        return quadruple_to_rank

    @classmethod
    def generate_distribution(cls, nocc, nvir, nranks, distribution="balanced", batch_size=None):
        """
        Generate the deterministic canonical (i,j,k,l) ownership map.

        The returned list-of-lists matches ``rank_quadruples`` for a runtime
        ``DistributedT4IJKL`` object initialized with the same system size,
        number of ranks, distribution strategy, and ``batch_size``.  Zero T4
        blocks are omitted, following the runtime storage rule.
        """
        nranks = int(nranks)
        if nranks <= 0:
            raise ValueError("nranks must be a positive integer")

        generator = cls.__new__(cls)
        generator.nocc = int(nocc)
        generator.nvir = int(nvir)
        generator.size = nranks
        generator.rank = 0
        generator.batch_size = batch_size

        all_quadruples = generator._enumerate_ijkl_quadruples()
        generator._assign_rank_quadruples(all_quadruples, distribution, batch_size=batch_size)
        return [[tuple(int(x) for x in quadruple) for quadruple in quadruples]
                for quadruples in generator.rank_quadruples]

    def _finalize_distribution(self, all_quadruples):
        """Broadcast and build distribution mappings consistently on all ranks."""
        if self.rank == 0:
            serialized = [np.asarray(self.rank_quadruples[r], dtype=np.int32).reshape(-1, 4) for r in range(self.size)]
        else:
            serialized = None

        serialized = self.comm.bcast(serialized, root=0)
        self.rank_quadruples = [list(map(tuple, arr)) for arr in serialized]
        self.local_quadruples = self.rank_quadruples[self.rank]

        self.ijkl_to_rank = {}
        for r in range(self.size):
            for quadruple in self.rank_quadruples[r]:
                quadruple = tuple(int(x) for x in quadruple)
                self.ijkl_to_rank[quadruple] = r

        self.ijkl_to_global_idx = {}
        self.stored_to_full_idx = np.empty(len(all_quadruples), dtype=np.int64)
        self.full_to_stored_idx = np.full(self.nocc4_full, -1, dtype=np.int64)
        for idx, quadruple in enumerate(all_quadruples):
            quadruple = tuple(int(x) for x in quadruple)
            self.ijkl_to_global_idx[quadruple] = idx
            full_idx = ijkl_to_linear(*quadruple, self.nocc)
            self.stored_to_full_idx[idx] = full_idx
            self.full_to_stored_idx[full_idx] = idx

        self.ijkl_idx_to_rank = np.zeros(self.nocc4, dtype=np.int32)
        self.full_ijkl_idx_to_rank = np.full(self.nocc4_full, -1, dtype=np.int32)
        for quadruple, rank in self.ijkl_to_rank.items():
            global_idx = self.ijkl_to_global_idx[quadruple]
            self.ijkl_idx_to_rank[global_idx] = rank
            full_idx = self.stored_to_full_idx[global_idx]
            self.full_ijkl_idx_to_rank[full_idx] = rank

    def _finalize_distribution_from_saved(self, all_quadruples):
        """Finalize distribution mappings from saved data without recomputation."""
        self.ijkl_to_rank = {}
        for r in range(self.size):
            for quadruple in self.rank_quadruples[r]:
                quadruple = tuple(int(x) for x in quadruple)
                self.ijkl_to_rank[quadruple] = r

        self.ijkl_to_global_idx = {}
        self.stored_to_full_idx = np.empty(len(all_quadruples), dtype=np.int64)
        self.full_to_stored_idx = np.full(self.nocc4_full, -1, dtype=np.int64)
        for idx, quadruple in enumerate(all_quadruples):
            quadruple = tuple(int(x) for x in quadruple)
            self.ijkl_to_global_idx[quadruple] = idx
            full_idx = ijkl_to_linear(*quadruple, self.nocc)
            self.stored_to_full_idx[idx] = full_idx
            self.full_to_stored_idx[full_idx] = idx

        self.ijkl_idx_to_rank = np.zeros(self.nocc4, dtype=np.int32)
        self.full_ijkl_idx_to_rank = np.full(self.nocc4_full, -1, dtype=np.int32)
        for quadruple, rank in self.ijkl_to_rank.items():
            global_idx = self.ijkl_to_global_idx[quadruple]
            self.ijkl_idx_to_rank[global_idx] = rank
            full_idx = self.stored_to_full_idx[global_idx]
            self.full_ijkl_idx_to_rank[full_idx] = rank

    def _build_local_mapping(self):
        self.local_ijkl_offset = {}
        self.local_ijkl_list = []

        for local_idx, quadruple in enumerate(self.local_quadruples):
            quadruple = tuple(int(x) for x in quadruple)
            self.local_ijkl_offset[quadruple] = local_idx
            self.local_ijkl_list.append((*quadruple, local_idx))

        self.local_nocc4 = len(self.local_quadruples)
        self._build_c_arrays()

    def _build_c_arrays(self):
        self.global_to_local_offset = np.full(self.nocc4, -1, dtype=np.int64)
        self.full_to_local_offset = np.full(self.nocc4_full, -1, dtype=np.int64)
        for quadruple, local_idx in self.local_ijkl_offset.items():
            global_idx = self.ijkl_to_global_idx[quadruple]
            self.global_to_local_offset[global_idx] = local_idx
            full_idx = self.stored_to_full_idx[global_idx]
            self.full_to_local_offset[full_idx] = local_idx

        self.ijkl_idx_to_rank = np.zeros(self.nocc4, dtype=np.int32)
        self.full_ijkl_idx_to_rank = np.full(self.nocc4_full, -1, dtype=np.int32)
        for quadruple, rank in self.ijkl_to_rank.items():
            global_idx = self.ijkl_to_global_idx[quadruple]
            self.ijkl_idx_to_rank[global_idx] = rank
            full_idx = self.stored_to_full_idx[global_idx]
            self.full_ijkl_idx_to_rank[full_idx] = rank

    def _get_canonical(self, i, j, k, l):
        canonical, _ = _canonical_and_perm(i, j, k, l)
        return canonical

    def _get_permutation(self, i, j, k, l):
        _, perm = _canonical_and_perm(i, j, k, l)
        return perm

    def get_owner(self, i, j, k, l):
        """Return rank that owns the canonical form of (i,j,k,l)."""
        if ijkl_is_zero_block(i, j, k, l):
            return -1
        return self.ijkl_to_rank.get(self._get_canonical(i, j, k, l), -1)

    def get_owner_ijkl_idx(self, ijkl_idx, compact=False):
        """
        Return rank that owns a canonical ijkl index.

        By default, ``ijkl_idx`` is the full triangular index returned by
        ``ijkl_to_linear``.  Pass ``compact=True`` for the stored nonzero index.
        Zero blocks return -1 because they are not owned by any rank.
        """
        ijkl_idx = int(ijkl_idx)
        if compact:
            return int(self.ijkl_idx_to_rank[ijkl_idx])
        stored_idx = self.full_to_stored_idx[ijkl_idx]
        if stored_idx < 0:
            return -1
        return int(self.ijkl_idx_to_rank[stored_idx])

    def get_local_offset(self, i, j, k, l):
        """Return local storage offset for the canonical form of (i,j,k,l)."""
        if ijkl_is_zero_block(i, j, k, l):
            return None
        return self.local_ijkl_offset.get(self._get_canonical(i, j, k, l))

    def get_local_index(self, i, j, k, l):
        """Alias for get_local_offset."""
        return self.get_local_offset(i, j, k, l)

    def allocate_local(self):
        """Allocate local T4 storage."""
        return np.zeros((self.local_nocc4, self.nvir, self.nvir, self.nvir, self.nvir), dtype=self.dtype,)

    def iter_local_ijkl(self):
        """Iterate over local canonical quadruples and their local offsets."""
        for i, j, k, l, local_idx in self.local_ijkl_list:
            yield i, j, k, l, local_idx

    def _require_c_copy_requests(self, t4_local, send_data, indices):
        _require_t4_c("fill_local_data_ijkl_", obj=self)
        if self.dtype != np.float64 or t4_local.dtype != np.float64 or send_data.dtype != np.float64:
            raise TypeError("C T4 copy helper requires float64 T4 buffers")
        if not t4_local.flags["C_CONTIGUOUS"] or not send_data.flags["C_CONTIGUOUS"]:
            raise ValueError("C T4 copy helper requires contiguous T4 buffers")
        if len(indices) == 0:
            return None
        if max(indices) > np.iinfo(np.int32).max:
            raise OverflowError("C T4 copy helper requires int32 request indices")
        return np.asarray(indices, dtype=np.int32)

    def _require_c_unpack_ready(self, t4_local, t4_blk):
        _require_t4_c("unpack_t4_ijkl_single_", obj=self)
        if self.dtype != np.float64 or t4_local.dtype != np.float64 or t4_blk.dtype != np.float64:
            raise TypeError("C T4 unpack helper requires float64 T4 buffers")
        if not t4_local.flags["C_CONTIGUOUS"] or not t4_blk.flags["C_CONTIGUOUS"]:
            raise ValueError("C T4 unpack helper requires contiguous T4 buffers")

    def _mpi_dtype(self):
        if self.dtype == np.float64:
            return MPI.DOUBLE
        if self.dtype == np.float32:
            return MPI.FLOAT
        raise TypeError(f"Unsupported MPI dtype for T4 amplitudes: {self.dtype}")

    def _fill_send_data(self, t4_local, send_data, ijkl_indices):
        requests = self._require_c_copy_requests(t4_local, send_data, ijkl_indices)
        if requests is None:
            return
        fill_local_data_ijkl_(t4_local, send_data, requests, self.global_to_local_offset, len(ijkl_indices), self.nvir)

    def _partition_collect_requests(self, ijkl_quadruples):
        """Split requested canonical quadruples into stored blocks and zero blocks."""
        quadruples_by_owner = [[] for _ in range(self.size)]
        zero_quadruples = []
        my_indices = []

        for quad in ijkl_quadruples:
            quadruple = tuple(int(x) for x in quad)
            if ijkl_is_zero_block(*quadruple):
                zero_quadruples.append(quadruple)
                continue

            stored_idx = self.ijkl_to_global_idx[quadruple]
            owner = int(self.ijkl_idx_to_rank[stored_idx])
            quadruples_by_owner[owner].append(quadruple)
            if owner == self.rank:
                my_indices.append(int(stored_idx))

        counts = np.array([len(quadruples_by_owner[r]) for r in range(self.size)], dtype=np.int32)
        displs = np.zeros(self.size, dtype=np.int32)
        displs[1:] = np.cumsum(counts)[:-1]
        return quadruples_by_owner, zero_quadruples, my_indices, counts, displs

    def _append_zero_blocks(self, recv_data, quadruples_by_owner, zero_quadruples):
        nonzero_quadruples = [quad for r in range(self.size) for quad in quadruples_by_owner[r]]

        if zero_quadruples:
            n_zero = len(zero_quadruples)
            shape = (len(recv_data) + n_zero, self.nvir, self.nvir, self.nvir, self.nvir)
            out = np.empty(shape, dtype=self.dtype)
            if len(recv_data) > 0:
                out[:len(recv_data)] = recv_data
            out[len(recv_data):] = 0.0
            ijkl_reordered = np.asarray(nonzero_quadruples + zero_quadruples, dtype=np.int32)
            return out, ijkl_reordered

        ijkl_reordered = np.asarray(nonzero_quadruples, dtype=np.int32).reshape(-1, 4)
        return recv_data, ijkl_reordered

    def unpack_t4_single_local(self, t4_local, t4_blk, i0, j0, k0, l0):
        if not ijkl_is_zero_block(i0, j0, k0, l0) and self.get_local_offset(i0, j0, k0, l0) is None:
            canonical = self._get_canonical(i0, j0, k0, l0)
            raise ValueError(f"{canonical} is not owned by rank {self.rank}")

        self._require_c_unpack_ready(t4_local, t4_blk)
        unpack_t4_ijkl_single_(t4_local, t4_blk, self.full_to_local_offset, i0, j0, k0, l0, self.nocc, self.nvir)
        return t4_blk

    def prefetch_t4_quadruples_allgather(self, t4_local, ijkl_quadruples):
        """
        Start a non-blocking allgather prefetch for canonical T4 quadruples.
        """
        if not hasattr(self, "recv_buffers"):
            self.recv_buffers = [None, None]
            self.send_buffers = [None, None]
            self.buffer_idx = 0
            self.quadruple_type = None

        nvir = self.nvir
        ijkl_quadruples = _as_canonical_quadruples(ijkl_quadruples)
        n_quad = len(ijkl_quadruples)

        if n_quad == 0 or self.size == 1:
            return None

        pidx = self.buffer_idx
        self.buffer_idx = 1 - self.buffer_idx

        quadruples_by_owner, zero_quadruples, my_indices, counts, displs = \
            self._partition_collect_requests(ijkl_quadruples)
        my_send_count = int(counts[self.rank])
        total_recv = int(counts.sum())

        required_recv_size = total_recv * nvir * nvir * nvir * nvir
        if required_recv_size == 0:
            recv_data = np.empty((0, nvir, nvir, nvir, nvir), dtype=self.dtype)
        elif self.recv_buffers[pidx] is None or self.recv_buffers[pidx].size < required_recv_size:
            self.recv_buffers[pidx] = np.empty((total_recv, nvir, nvir, nvir, nvir), dtype=self.dtype)
            recv_data = self.recv_buffers[pidx][:total_recv]
        else:
            recv_data = self.recv_buffers[pidx][:total_recv]

        required_send_size = my_send_count * nvir * nvir * nvir * nvir
        if required_send_size == 0:
            send_data = np.empty((0, nvir, nvir, nvir, nvir), dtype=self.dtype)
        elif self.send_buffers[pidx] is None or self.send_buffers[pidx].size < required_send_size:
            self.send_buffers[pidx] = np.empty((my_send_count, nvir, nvir, nvir, nvir), dtype=self.dtype)
            send_data = self.send_buffers[pidx][:my_send_count]
        else:
            send_data = self.send_buffers[pidx][:my_send_count]

        self._fill_send_data(t4_local, send_data, my_indices)

        t0_issue = logger.perf_counter()
        if total_recv == 0:
            reqs = []
        else:
            if self.quadruple_type is None:
                block_size = nvir * nvir * nvir * nvir
                self.quadruple_type = self._mpi_dtype().Create_contiguous(block_size)
                self.quadruple_type.Commit()

            req = self.comm.Iallgatherv([send_data.ravel(), my_send_count, self.quadruple_type],
                                        [recv_data.ravel(), counts, displs, self.quadruple_type])
            reqs = [req]
        t1_issue = logger.perf_counter()
        self._log_communication(f"T4 prefetch issue: nreq={n_quad}, send={my_send_count}, "
                                f"recv={total_recv}, post time {t1_issue - t0_issue:.4f} sec.")

        return {
            "reqs": reqs,
            "recv_data": recv_data,
            "send_data": send_data,
            "quadruples_by_owner": quadruples_by_owner,
            "zero_quadruples": zero_quadruples,
            "ijkl_quadruples": ijkl_quadruples,
            "t0_issue": t0_issue,
            "t1_issue": t1_issue,
        }

    def finalize_prefetch_t4_quadruples(self, handle, t4_local, ijkl_quadruples):
        """
        Finalize a T4 allgather prefetch.
        """
        nvir = self.nvir
        ijkl_quadruples = _as_canonical_quadruples(ijkl_quadruples)
        n_quad = len(ijkl_quadruples)

        if n_quad == 0:
            shape = (0, nvir, nvir, nvir, nvir)
            return np.empty(shape, dtype=self.dtype), np.empty((0, 4), dtype=np.int32)

        if handle is None:
            return self.collect_t4_quadruples(t4_local, ijkl_quadruples)

        t0_finalize = logger.perf_counter()
        if handle["reqs"]:
            MPI.Request.Waitall(handle["reqs"])
        t1_finalize = logger.perf_counter()
        t0_issue = handle.get("t0_issue", t0_finalize)
        t1_issue = handle.get("t1_issue", t0_issue)
        self._log_communication(f"T4 prefetch finalize: nreq={n_quad}, post time {t1_issue - t0_issue:.4f} sec., "
                                f"elapsed since issue {t1_finalize - t0_issue:.4f} sec., "
                                f"wait time {t1_finalize - t0_finalize:.4f} sec.")

        return self._append_zero_blocks(handle["recv_data"], handle["quadruples_by_owner"],
                                        handle.get("zero_quadruples", []))

    def print_distribution_info(self):
        """Print distribution information."""
        if self.rank != 0:
            return

        lines = [
            "",
            "DistributedT4IJKL Configuration:",
            f"  nocc = {self.nocc}, nvir = {self.nvir}",
            f"  Distribution strategy: {self.distribution}",
            f"  Full canonical quadruples: {self.nocc4_full}",
            f"  Skipped zero quadruples: {self.nocc4_zero}",
            f"  Stored canonical quadruples: {self.nocc4}",
            f"  Number of MPI ranks: {self.size}",
            f"  C library loaded: {_HAS_C_LIB}",
            "",
            "Distribution by rank:",
        ]
        for r in range(self.size):
            n_quad = len(self.rank_quadruples[r])
            pct = 100.0 * n_quad / self.nocc4 if self.nocc4 > 0 else 0.0
            lines.append(f"  Rank {r:3d}: {n_quad:6d} quadruples ({pct:.1f}%)")

        if hasattr(self, "_rank_batch_costs"):
            rank_batch_costs = self._rank_batch_costs
            mean_per_batch = rank_batch_costs.mean(axis=0)
            valid = mean_per_batch > 0.0
            if np.any(valid):
                relative = rank_batch_costs[:, valid] / mean_per_batch[None, valid]
                worst = relative.max()
                lines.extend(["", "Per-batch contraction load model:",
                    f"  Batch size: {self.batch_size}", f"  Worst rank/mean batch load ratio: {worst:.3f}"])
        if self.log:
            for line in lines:
                self.log.info("%s", line)
        else:
            print("\n".join(lines))

    def memory_usage_bytes(self):
        """Return local T4 storage size in bytes."""
        return self.local_nocc4 * self.nvir**4 * self.dtype.itemsize

    def memory_usage_gb(self):
        """Return local T4 storage size in GiB."""
        return self.memory_usage_bytes() / (1024**3)

    def save_to_disk(self, t4_local, prefix):
        """
        Save local T4 data and distribution metadata.
        """
        np.save(f"{prefix}t4_local_{self.rank}.npy", t4_local)
        np.save(f"{prefix}ijkl_quadruples_{self.rank}.npy",
                np.asarray(self.local_quadruples, dtype=np.int32).reshape(-1, 4))

        if self.rank == 0:
            metadata = {
                "kind": "DistributedT4IJKL",
                "nocc": self.nocc,
                "nvir": self.nvir,
                "distribution": self.distribution,
                "batch_size": self.batch_size,
                "dtype": self.dtype.str,
                "size": self.size,
                "zero_rule": "skip_i_eq_j_eq_k_or_j_eq_k_eq_l",
            }
            with open(f"{prefix}metadata.json", "w") as f:
                json.dump(metadata, f)

        self.comm.Barrier()

    @classmethod
    def load_from_disk(cls, prefix, comm, mmap_mode=None):
        """Load T4 data and reconstruct a DistributedT4IJKL object."""
        rank = comm.Get_rank()
        size = comm.Get_size()

        with open(f"{prefix}metadata.json", "r") as f:
            metadata = json.load(f)

        nocc = int(metadata["nocc"])
        nvir = int(metadata["nvir"])
        distribution = metadata["distribution"]
        dtype = np.dtype(metadata["dtype"])
        saved_size = int(metadata["size"])

        if size != saved_size:
            raise ValueError(f"MPI size mismatch: saved with {saved_size} ranks, "
                            f"but loading with {size} ranks. Redistribution is not implemented.")

        saved_rank_quadruples = []
        for r in range(size):
            arr = np.load(f"{prefix}ijkl_quadruples_{r}.npy")
            saved_rank_quadruples.append([tuple(int(x) for x in quad) for quad in arr])

        dist_t4 = cls.__new__(cls)
        dist_t4.log = None
        dist_t4.nocc = nocc
        dist_t4.nvir = nvir
        dist_t4.dtype = dtype
        dist_t4.comm = comm
        dist_t4.distribution = distribution
        dist_t4.batch_size = metadata.get("batch_size")
        dist_t4.nocc4_full = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24
        dist_t4.nocc4_zero = nocc * nocc
        dist_t4.nocc4 = dist_t4.nocc4_full - dist_t4.nocc4_zero
        dist_t4.rank = rank
        dist_t4.size = size
        dist_t4.log_t4_communication = False
        dist_t4.rank_quadruples = saved_rank_quadruples
        dist_t4.local_quadruples = saved_rank_quadruples[rank]

        all_quadruples = dist_t4._enumerate_ijkl_quadruples()
        dist_t4._finalize_distribution_from_saved(all_quadruples)
        dist_t4._build_local_mapping()

        t4_local = np.load(f"{prefix}t4_local_{rank}.npy", mmap_mode=mmap_mode)
        return dist_t4, t4_local

def generate_deterministic_ijkl_distribution(nocc, nvir, nranks, distribution="balanced", batch_size=None):
    return DistributedT4IJKL.generate_distribution(nocc, nvir, nranks, distribution=distribution, batch_size=batch_size)
