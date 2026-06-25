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

"""
Distributed T3 Amplitude Storage with (i,j,k) Triple Distribution

This module provides a DistributedT3IJK class for storing and manipulating triangularly-stored T3 amplitudes
across multiple MPI ranks using direct (i,j,k) triple distribution for optimal load balancing.
"""

import sys
import numpy as np
import json

from mpi4py import MPI
from pyscf.lib import logger
from distr_cc._runtime import require_native_c, warn_python_fallback

_NATIVE_IMPORT_ERROR = None

try:
    from distr_cc._c_rccsdt import (fill_local_data_ijk_, pack_interleaved, pack_redistributed_send_buffer_c,
                                    unpack_received_data_indices, take_t3_ijk_single_)
    _HAS_C_LIB = True
except (OSError, AttributeError) as err:
    _NATIVE_IMPORT_ERROR = err
    _HAS_C_LIB = False
    warn_python_fallback("distributed RCCSDT T3 helpers", err=err)

def _require_t3_c(feature, obj=None):
    return require_native_c(_HAS_C_LIB, feature, obj=obj, err=_NATIVE_IMPORT_ERROR)

def ijk_to_linear(i, j, k, nocc):
    """
    Convert canonical triple (i <= j <= k) to linear index.
    """
    n = nocc
    ni = nocc - i
    n_before_i = (n * (n + 1) * (n + 2) - ni * (ni + 1) * (ni + 2)) // 6
    n_before_j = (j - i) * (2 * nocc - j - i + 1) // 2
    n_k = k - j
    return n_before_i + n_before_j + n_k

def linear_to_ijk(idx, nocc):
    """
    Convert linear index back to canonical triple (i, j, k).
    """
    remaining = idx
    i = 0
    while i < nocc:
        ni = nocc - i
        n_i = ni * (ni + 1) // 2
        if remaining < n_i:
            break
        remaining -= n_i
        i += 1
    j = i
    while j < nocc:
        n_j = nocc - j
        if remaining < n_j:
            break
        remaining -= n_j
        j += 1
    k = j + remaining
    return i, j, k


class DistributedT3IJK:
    """
    Distributed triangular T3 amplitude storage with (i,j,k) triple distribution.

    Each canonical triple (i <= j <= k) is independently assigned to a rank,
    allowing for optimal load balancing across MPI processes.
    """

    def __init__(self, nocc, nvir, comm, distribution='balanced', batch_size=None, dtype=np.float64,
                allow_python_fallback=None):
        """
        Initialize distributed T3 storage.

        Parameters
        ----------
        nocc : int
            Number of occupied orbitals
        nvir : int
            Number of virtual orbitals
        comm : MPI.Comm
            MPI communicator
        distribution : str
            Distribution strategy: 'balanced', 'round_robin', or 'block'
        batch_size : int, optional
            Block size for weighted distribution (considers computational costs)
        dtype : np.dtype
            Data type for storage
        """
        self.nocc = nocc
        self.nvir = nvir
        self.dtype = dtype
        self.comm = comm
        self.distribution = distribution
        self.batch_size = batch_size
        self.allow_python_fallback = allow_python_fallback
        self.nocc3 = nocc * (nocc + 1) * (nocc + 2) // 6
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._compute_distribution(distribution, batch_size)
        self._build_local_mapping()
        self._setup_permutation_maps()
        self.log_t3_communication = False
        self.log = None # Logger object

    def _log_communication(self, message):
        if not getattr(self, "log_t3_communication", False):
            return
        print(f"        Rank {self.rank} {message}", flush=True)

    def _enumerate_ijk_triples(self):
        """Enumerate all canonical (i,j,k) triples with i <= j <= k."""
        nocc = self.nocc
        triples = []
        for i in range(nocc):
            for j in range(i, nocc):
                for k in range(j, nocc):
                    triples.append((i, j, k))
        return triples

    def _compute_distribution(self, strategy, batch_size=None):
        """Compute which triples each rank owns."""
        all_triples = self._enumerate_ijk_triples()
        self._assign_rank_triples(all_triples, strategy, batch_size)
        self._finalize_distribution(all_triples)

    def _assign_rank_triples(self, all_triples, strategy, batch_size=None):
        """Assign canonical triples to ranks without building MPI-local maps."""
        if strategy == 'round_robin':
            self._distribute_round_robin(all_triples)
        elif strategy == 'block':
            self._distribute_block(all_triples)
        elif strategy == 'balanced':
            if batch_size is not None:
                self._distribute_balanced_per_batch(all_triples, batch_size)
            else:
                self._distribute_round_robin(all_triples)
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")

    def _distribute_round_robin(self, all_triples):
        """Simple round-robin distribution of triples."""
        self.rank_triples = [[] for _ in range(self.size)]
        for idx, triple in enumerate(all_triples):
            owner = idx % self.size
            self.rank_triples[owner].append(triple)

    def _distribute_block(self, all_triples):
        """Block distribution of triples."""
        n = len(all_triples)
        base_size = n // self.size
        remainder = n % self.size

        self.rank_triples = [[] for _ in range(self.size)]
        idx = 0
        for r in range(self.size):
            count = base_size + (1 if r < remainder else 0)
            for _ in range(count):
                if idx < n:
                    self.rank_triples[r].append(all_triples[idx])
                    idx += 1

    def _distribute_balanced_per_batch(self, all_triples, batch_size):
        n_triples = len(all_triples)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if n_triples == 0:
            self.rank_triples = [[] for _ in range(self.size)]
            self._rank_batch_costs = np.zeros((self.size, 0), dtype=np.float64)
            self._triple_batch_costs = np.zeros((0, 0), dtype=np.float64)
            return
        nocc = self.nocc

        # Target number of triples per rank
        base_count = n_triples // self.size
        remainder = n_triples % self.size
        target_counts = np.array([base_count + (1 if r < remainder else 0) for r in range(self.size)])

        # Build batches
        all_triples_arr = np.array(all_triples, dtype=np.int32)
        n_batches = (n_triples + batch_size - 1) // batch_size

        batches = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_triples)
            batches.append(all_triples_arr[start:end])

        # Step 1: Compute cost matrix efficiently
        triple_batch_costs = self._compute_cost_matrix(all_triples, batches, n_batches)
        # Step 2: Initial assignment using multi-criteria sorting
        triple_to_rank = self._initial_assignment_balanced(triple_batch_costs, target_counts, n_batches)
        # Step 3: Iterative refinement
        triple_to_rank = self._refine_assignment(triple_to_rank, triple_batch_costs, target_counts, max_iterations=100)
        # Build rank_triples
        self.rank_triples = [[] for _ in range(self.size)]
        for t_idx, triple in enumerate(all_triples):
            r = triple_to_rank[t_idx]
            self.rank_triples[r].append(triple)
        # Compute final costs for diagnostics
        rank_batch_costs = np.zeros((self.size, n_batches), dtype=np.float64)
        for t_idx in range(n_triples):
            r = triple_to_rank[t_idx]
            rank_batch_costs[r] += triple_batch_costs[t_idx]

        self._rank_batch_costs = rank_batch_costs
        self._triple_batch_costs = triple_batch_costs

    def _compute_cost_matrix(self, all_triples, batches, n_batches):
        """
        Fully vectorized cost matrix computation with correct cost ratios.
        """
        n_triples = len(all_triples)
        nvir = self.nvir

        # Cost ratio
        COST_WOOOO = 1.0
        COST_WOVOV = 3.0 * nvir

        all_triples_arr = np.array(all_triples, dtype=np.int32)
        local_i = all_triples_arr[:, 0]
        local_j = all_triples_arr[:, 1]
        local_k = all_triples_arr[:, 2]

        triple_batch_costs = np.zeros((n_triples, n_batches), dtype=np.float64)

        for b, batch_ijk in enumerate(batches):
            batch_size_b = len(batch_ijk)

            o00 = batch_ijk[:, 0]
            o10 = batch_ijk[:, 1]
            o20 = batch_ijk[:, 2]

            d01 = (o00 != o10).astype(np.float64)
            d12 = (o10 != o20).astype(np.float64)
            ad = d01 * d12

            # Broadcast comparisons: (n_triples, batch_size_b)
            k_eq_o20 = (local_k[:, None] == o20[None, :]).astype(np.float64)
            k_eq_o10 = (local_k[:, None] == o10[None, :]).astype(np.float64)
            k_eq_o00 = (local_k[:, None] == o00[None, :]).astype(np.float64)

            j_eq_o20 = (local_j[:, None] == o20[None, :]).astype(np.float64)
            j_eq_o10 = (local_j[:, None] == o10[None, :]).astype(np.float64)
            j_eq_o00 = (local_j[:, None] == o00[None, :]).astype(np.float64)

            i_eq_o20 = (local_i[:, None] == o20[None, :]).astype(np.float64)
            i_eq_o10 = (local_i[:, None] == o10[None, :]).astype(np.float64)
            i_eq_o00 = (local_i[:, None] == o00[None, :]).astype(np.float64)

            # W_oooo contributions
            cost = np.zeros((n_triples, batch_size_b), dtype=np.float64)

            # Part 0: k matches
            cost += COST_WOOOO * k_eq_o20
            cost += COST_WOOOO * d12[None, :] * k_eq_o10
            cost += COST_WOOOO * d01[None, :] * k_eq_o20
            cost += COST_WOOOO * d12[None, :] * k_eq_o10
            cost += COST_WOOOO * d01[None, :] * k_eq_o00
            cost += COST_WOOOO * ad[None, :] * k_eq_o00

            # Part 1: j matches
            cost += COST_WOOOO * j_eq_o20
            cost += COST_WOOOO * d12[None, :] * j_eq_o10
            cost += COST_WOOOO * d01[None, :] * j_eq_o20
            cost += COST_WOOOO * d12[None, :] * j_eq_o10
            cost += COST_WOOOO * d01[None, :] * j_eq_o00
            cost += COST_WOOOO * ad[None, :] * j_eq_o00

            # Part 2: i matches
            cost += COST_WOOOO * i_eq_o20
            cost += COST_WOOOO * d12[None, :] * i_eq_o10
            cost += COST_WOOOO * d01[None, :] * i_eq_o20
            cost += COST_WOOOO * d12[None, :] * i_eq_o10
            cost += COST_WOOOO * d01[None, :] * i_eq_o00
            cost += COST_WOOOO * ad[None, :] * i_eq_o00

            # W_ovov contributions (much higher cost)
            cost += COST_WOVOV * j_eq_o00 * k_eq_o10
            cost += COST_WOVOV * d12[None, :] * j_eq_o00 * k_eq_o20
            cost += COST_WOVOV * d01[None, :] * j_eq_o10 * k_eq_o20

            cost += COST_WOVOV * i_eq_o00 * k_eq_o10
            cost += COST_WOVOV * d12[None, :] * i_eq_o00 * k_eq_o20
            cost += COST_WOVOV * d01[None, :] * i_eq_o10 * k_eq_o20

            cost += COST_WOVOV * i_eq_o00 * j_eq_o10
            cost += COST_WOVOV * d12[None, :] * i_eq_o00 * j_eq_o20
            cost += COST_WOVOV * d01[None, :] * i_eq_o10 * j_eq_o20

            triple_batch_costs[:, b] = cost.sum(axis=1)

        return triple_batch_costs

    def _initial_assignment_balanced(self, triple_batch_costs, target_counts, n_batches):
        """
        Initial assignment using LPT (Longest Processing Time) heuristic adapted for multi-batch.

        Strategy: Sort triples by their *maximum single-batch cost* (descending).
        Assign each triple to the rank that minimizes the increase in bottleneck batch cost.
        """
        n_triples = triple_batch_costs.shape[0]
        triple_to_rank = np.full(n_triples, -1, dtype=np.int32)
        rank_counts = np.zeros(self.size, dtype=np.int64)
        rank_batch_costs = np.zeros((self.size, n_batches), dtype=np.float64)

        # Sort by max single-batch impact first, then total cost
        max_impact = triple_batch_costs.max(axis=1)
        total_costs = triple_batch_costs.sum(axis=1)
        sort_key = max_impact * 1000.0 + total_costs
        sorted_indices = np.lexsort((np.arange(n_triples, dtype=np.int64), -sort_key))

        for t_idx in sorted_indices:
            triple_costs = triple_batch_costs[t_idx]

            best_rank = None
            best_score = (np.inf, np.inf)

            valid_ranks = [r for r in range(self.size) if rank_counts[r] < target_counts[r]]

            if not valid_ranks:
                 valid_ranks = [r for r in range(self.size) if rank_counts[r] <= target_counts[r]]

            for r in valid_ranks:
                new_costs = rank_batch_costs[r] + triple_costs
                ranking_metric = new_costs.max()
                total_load = new_costs.sum()
                score = (ranking_metric, total_load)

                if score < best_score:
                    best_score = score
                    best_rank = r

            if best_rank is not None:
                triple_to_rank[t_idx] = best_rank
                rank_counts[best_rank] += 1
                rank_batch_costs[best_rank] += triple_costs
            else:
                for r in range(self.size):
                     if rank_counts[r] < target_counts[r] + 1:
                        triple_to_rank[t_idx] = r
                        rank_counts[r] += 1
                        rank_batch_costs[r] += triple_costs
                        break
        return triple_to_rank

    def _refine_assignment(self, triple_to_rank, triple_batch_costs, target_counts, max_iterations=100):
        """
        Iteratively refine assignment by swapping triples between ranks.

        Improvements:
        1. Deterministic ordering and explicit tie-breakers.
        2. Steepest descent search over broad set of swaps.
        3. Objective function: minimize max(rank_cost / mean_rank_cost) over batches.
        """
        n_triples = triple_batch_costs.shape[0]
        n_batches = triple_batch_costs.shape[1]

        # Compute current rank batch costs
        rank_batch_costs = np.zeros((self.size, n_batches), dtype=np.float64)
        for t_idx in range(n_triples):
            r = triple_to_rank[t_idx]
            rank_batch_costs[r] += triple_batch_costs[t_idx]

        # Build index lists per rank for efficient lookup
        rank_triple_indices = [[] for _ in range(self.size)]
        for t_idx in range(n_triples):
            rank_triple_indices[triple_to_rank[t_idx]].append(t_idx)

        def compute_objective(rbc):
            mean_per_batch = rbc.mean(axis=0)
            mean_per_batch[mean_per_batch == 0] = 1.0
            relative_load = rbc / mean_per_batch[None, :]
            return relative_load.max()

        current_obj = compute_objective(rank_batch_costs)

        # Improvement threshold
        tol = 1e-5

        for iteration in range(max_iterations):
            improved = False
            mean_per_batch = rank_batch_costs.mean(axis=0)
            mean_per_batch[mean_per_batch == 0] = 1.0
            relative_load = rank_batch_costs / mean_per_batch[None, :]
            flat_idx = np.argmax(relative_load)
            overloaded_rank, worst_batch = np.unravel_index(flat_idx, relative_load.shape)
            target_ranks = np.lexsort((np.arange(self.size, dtype=np.int64), rank_batch_costs[:, worst_batch],))[:3]

            best_swap = None
            best_new_obj = current_obj

            candidates_from_overloaded = sorted(rank_triple_indices[overloaded_rank],
                                              key=lambda t: (-triple_batch_costs[t, worst_batch], t))
            candidates_from_overloaded = candidates_from_overloaded[:20]

            for t1_idx in candidates_from_overloaded:
                t1_costs = triple_batch_costs[t1_idx]

                for r_target in target_ranks:
                    if r_target == overloaded_rank: continue

                    candidates_from_target = sorted(rank_triple_indices[r_target],
                                                   key=lambda t: (triple_batch_costs[t, worst_batch], t))
                    candidates_from_target = candidates_from_target[:5]

                    for t2_idx in candidates_from_target:
                        t2_costs = triple_batch_costs[t2_idx]
                        overloaded_new = rank_batch_costs[overloaded_rank] - t1_costs + t2_costs
                        target_new = rank_batch_costs[r_target] - t2_costs + t1_costs
                        new_rel_overloaded = overloaded_new[worst_batch] / mean_per_batch[worst_batch]
                        new_rel_target = target_new[worst_batch] / mean_per_batch[worst_batch]

                        if new_rel_overloaded < current_obj and new_rel_target < current_obj:
                            temp_rbc = rank_batch_costs.copy()
                            temp_rbc[overloaded_rank] = overloaded_new
                            temp_rbc[r_target] = target_new

                            new_obj = compute_objective(temp_rbc)
                            if new_obj < best_new_obj - tol:
                                best_new_obj = new_obj
                                best_swap = (t1_idx, t2_idx, overloaded_rank, r_target)
            if best_swap:
                t1_idx, t2_idx, r1, r2 = best_swap

                triple_to_rank[t1_idx] = r2
                triple_to_rank[t2_idx] = r1

                rank_triple_indices[r1].remove(t1_idx)
                rank_triple_indices[r1].append(t2_idx)
                rank_triple_indices[r2].remove(t2_idx)
                rank_triple_indices[r2].append(t1_idx)

                t1_costs = triple_batch_costs[t1_idx]
                t2_costs = triple_batch_costs[t2_idx]
                rank_batch_costs[r1] -= t1_costs
                rank_batch_costs[r1] += t2_costs
                rank_batch_costs[r2] -= t2_costs
                rank_batch_costs[r2] += t1_costs

                current_obj = best_new_obj
                improved = True

            if not improved:
                break

        return triple_to_rank

    @classmethod
    def generate_distribution(cls, nocc, nvir, nranks, distribution='balanced', batch_size=None):
        """
        Generate the deterministic canonical (i,j,k) ownership map for a parameter set.

        This helper does not require an MPI communicator.  It uses the same
        assignment code as the runtime class, so the returned list-of-lists is
        identical to ``rank_triples`` for a run with the same ``nocc``, ``nvir``,
        number of ranks, distribution strategy, and ``batch_size``.
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
        generator.log_t3_communication = False

        all_triples = generator._enumerate_ijk_triples()
        generator._assign_rank_triples(all_triples, distribution, batch_size)
        return [[tuple(int(x) for x in triple) for triple in triples]
                for triples in generator.rank_triples]

    def print_batch_balance_info(self, batch_size):
        """Print per-batch load balance information."""
        if not hasattr(self, '_rank_batch_costs'):
            print("No batch cost information available. Use 'balanced' distribution with batch_size.")
            return

        rank_batch_costs = self._rank_batch_costs
        n_batches = rank_batch_costs.shape[1]

        if self.rank == 0:
            print(f"\nPer-batch load balance (batch_size={batch_size}, n_batches={n_batches}):")

            for b in range(min(n_batches, 10)):  # Show first 10 batches
                costs = rank_batch_costs[:, b]
                max_cost = costs.max()
                min_cost = costs.min()
                mean_cost = costs.mean()
                imbalance = (max_cost - min_cost) / mean_cost * 100 if mean_cost > 0 else 0

                print(f"  Batch {b:3d}: min={min_cost:8.0f}, max={max_cost:8.0f}, "
                    f"mean={mean_cost:8.1f}, imbalance={imbalance:5.1f}%")

            if n_batches > 10:
                print(f"  ... ({n_batches - 10} more batches)")

            # Overall statistics
            total_per_rank = rank_batch_costs.sum(axis=1)
            print(f"\nTotal cost per rank:")
            for r in range(self.size):
                print(f"  Rank {r}: {total_per_rank[r]:.0f} ({len(self.rank_triples[r])} triples)")

    def _finalize_distribution(self, all_triples):
        """Finalize distribution mappings - must be consistent across all ranks."""

        # Broadcast rank_triples from rank 0 to ensure consistency
        if self.rank == 0:
            # Serialize rank_triples
            serialized = []
            for r in range(self.size):
                serialized.append(np.array(self.rank_triples[r], dtype=np.int32))
        else:
            serialized = None

        # Broadcast from rank 0
        serialized = self.comm.bcast(serialized, root=0)

        # Rebuild rank_triples from broadcast data
        self.rank_triples = [list(map(tuple, arr)) for arr in serialized]
        self.local_triples = self.rank_triples[self.rank]

        # Build global triple to rank mapping (now consistent across all ranks)
        self.ijk_to_rank = {}
        for r in range(self.size):
            for triple in self.rank_triples[r]:
                triple = tuple(int(x) for x in triple)
                self.ijk_to_rank[triple] = r

        # Build ijk_idx_to_rank mapping
        self.ijk_idx_to_rank = {}
        for triple, rank in self.ijk_to_rank.items():
            i, j, k = triple
            ijk_idx = ijk_to_linear(i, j, k, self.nocc)
            self.ijk_idx_to_rank[ijk_idx] = rank

        # Build global triple to linear index mapping
        self.ijk_to_global_idx = {}
        for idx, triple in enumerate(all_triples):
            triple = tuple(int(x) for x in triple)
            self.ijk_to_global_idx[triple] = idx

        # Build local index mapping
        self.global_to_local_offset = np.full(len(all_triples), -1, dtype=np.int64)
        for local_idx, triple in enumerate(self.local_triples):
            triple = tuple(int(x) for x in triple)
            global_idx = self.ijk_to_global_idx[triple]
            ijk_idx = ijk_to_linear(triple[0], triple[1], triple[2], self.nocc)
            self.global_to_local_offset[ijk_idx] = local_idx

    def _build_local_mapping(self):
        """Build mappings for local storage."""
        self.local_ijk_offset = {}
        self.local_ijk_list = []

        for local_idx, triple in enumerate(self.local_triples):
            self.local_ijk_offset[triple] = local_idx
            self.local_ijk_list.append((triple[0], triple[1], triple[2], local_idx))

        self.local_nocc3 = len(self.local_triples)

        self._build_c_arrays()

    def _build_c_arrays(self):
        """Build arrays needed for C function calls."""
        # Global ijk_idx to local offset (-1 if not local)
        self.global_to_local_offset = np.full(self.nocc3, -1, dtype=np.int64)
        for triple, local_idx in self.local_ijk_offset.items():
            global_idx = self.ijk_to_global_idx[triple]
            self.global_to_local_offset[global_idx] = local_idx

        # Triple to rank array for fast lookup
        self.ijk_idx_to_rank = np.zeros(self.nocc3, dtype=np.int32)
        for triple, r in self.ijk_to_rank.items():
            global_idx = self.ijk_to_global_idx[triple]
            self.ijk_idx_to_rank[global_idx] = r

    def _setup_permutation_maps(self):
        """Setup mappings for unpacking triangular to cubic representation."""
        self.tril2cube_vir_perm = [
            (0, 1, 2),  # Case 0: i,j,k -> i,j,k (i <= j <= k)
            (0, 2, 1),  # Case 1: i,j,k -> i,k,j (i <= k < j)
            (1, 0, 2),  # Case 2: i,j,k -> j,i,k (j < i <= k)
            (1, 2, 0),  # Case 3: i,j,k -> k,i,j (k < i <= j)
            (2, 0, 1),  # Case 4: i,j,k -> j,k,i (j <= k < i)
            (2, 1, 0),  # Case 5: i,j,k -> k,j,i (k < j < i)
        ]

    def _get_permutation(self, i, j, k):
        """Get virtual index permutation for given (i,j,k)."""
        if i <= j <= k:
            return (0, 1, 2)
        elif i <= k < j:
            return (0, 2, 1)
        elif j < i <= k:
            return (1, 0, 2)
        elif k < i <= j:
            return (1, 2, 0)
        elif j <= k < i:
            return (2, 0, 1)
        else:  # k < j < i
            return (2, 1, 0)

    def _get_canonical(self, i, j, k):
        """Get canonical form (sorted) of (i,j,k)."""
        return tuple(sorted([i, j, k]))

    def get_owner(self, i, j, k):
        """Return rank that owns the (i,j,k) triple."""
        ci, cj, ck = self._get_canonical(i, j, k)
        return self.ijk_to_rank.get((ci, cj, ck), -1)

    def get_owner_ijk_idx(self, ijk_idx):
        """Return rank that owns the given global ijk index."""
        return int(self.ijk_idx_to_rank[ijk_idx])

    def allocate_local(self):
        """Allocate local storage array."""
        return np.zeros((self.local_nocc3, self.nvir, self.nvir, self.nvir), dtype=self.dtype)

    def iter_local_ijk(self):
        """Iterate over local (i,j,k) triples."""
        for i, j, k, local_idx in self.local_ijk_list:
            yield i, j, k, local_idx

    def get_local_offset(self, i, j, k):
        """Get local storage offset for canonical (i,j,k) triple."""
        ci, cj, ck = self._get_canonical(i, j, k)
        return self.local_ijk_offset.get((ci, cj, ck))

    def get_local_index(self, i, j, k):
        """Get local storage index for canonical triple (i,j,k)."""
        return self.get_local_offset(i, j, k)

    def take_t3_single_local(self, t3_local, t3_blk, i0, j0, k0):
        """
        Unpack a single T3[i,j,k,:,:,:] element from local storage (in-place).
        """
        if _require_t3_c("take_t3_ijk_single_", obj=self):
            take_t3_ijk_single_(t3_local, t3_blk, self.global_to_local_offset, i0, j0, k0, self.nocc, self.nvir)
        else:
            warn_python_fallback("take_t3_single_local", err=_NATIVE_IMPORT_ERROR)
            return self._take_t3_single_local_py(t3_local, t3_blk, i0, j0, k0)
        return t3_blk

    def _take_t3_single_local_py(self, t3_local, t3_blk, i0, j0, k0):
        """Python implementation of take_t3_single_local."""
        nvir = self.nvir
        ci, cj, ck = self._get_canonical(i0, j0, k0)

        local_idx = self.local_ijk_offset.get((ci, cj, ck))
        if local_idx is None:
            raise ValueError(f"({ci},{cj},{ck}) not owned by this rank")

        t3_slice = t3_local[local_idx]
        perm = self._get_permutation(i0, j0, k0)

        for a in range(nvir):
            for b in range(nvir):
                for c in range(nvir):
                    abc = [a, b, c]
                    aa, bb, cc = abc[perm[0]], abc[perm[1]], abc[perm[2]]
                    t3_blk[aa, bb, cc] = t3_slice[a, b, c]
        return t3_blk

    def prefetch_t3_triples_iallgather(self, t3_local, ijk_triples, batch_size_hint=None):
        """
        Non-blocking prefetch using MPI_Iallgatherv and double buffering.

        This replaces point-to-point Isend/Irecv with a single collective operation.
        Each rank contributes its local triples for the current batch.
        The result is a gathered buffer containing ALL triples in the batch, sorted by rank.

        V2: Uses custom MPI datatype to avoid 32-bit integer overflow for large counts.
        """
        if not hasattr(self, 'recv_buffers'):
            self.recv_buffers = [None, None]
            self.send_buffers = [None, None]
            self.buffer_idx = 0

        # Select current buffer set
        pidx = self.buffer_idx
        self.buffer_idx = 1 - self.buffer_idx

        nocc, nvir = self.nocc, self.nvir
        n_tri = len(ijk_triples)

        # 1. Determine Ownership and construct triples_by_owner (needed for finalize)
        ijk_indices = np.array([ijk_to_linear(i, j, k, nocc) for i, j, k in ijk_triples], dtype=np.int32)

        triples_by_owner = [[] for _ in range(self.size)]
        my_indices = []

        for n, ijk_idx in enumerate(ijk_indices):
            owner = self.ijk_idx_to_rank[ijk_idx]
            triples_by_owner[owner].append(ijk_triples[n])
            if owner == self.rank:
                my_indices.append(ijk_idx)

        # 2. Compute counts
        counts = np.array([len(triples_by_owner[r]) for r in range(self.size)], dtype=np.int32)
        my_send_count = counts[self.rank]
        total_recv = int(counts.sum())
        assert total_recv == n_tri

        # Displacements (element-wise offset for python buffer slicing later/ or verification)
        displs = np.zeros(self.size, dtype=np.int32)
        displs[1:] = np.cumsum(counts)[:-1]

        # 3. Manage Receive Buffer
        required_recv_size = total_recv * nvir * nvir * nvir
        if self.recv_buffers[pidx] is None or self.recv_buffers[pidx].size < required_recv_size:
            if self.log:
                self.log.debug(f"Rank {self.rank}: Allocating recv_buffer[{pidx}] "
                               f"(Iallgatherv) size {required_recv_size/1e9:.3f} GB")
            self.recv_buffers[pidx] = np.empty((total_recv, nvir, nvir, nvir), dtype=self.dtype)

        recv_data = self.recv_buffers[pidx][:total_recv] # View

        # 4. Manage Send Buffer
        required_send_size = my_send_count * nvir * nvir * nvir
        if self.send_buffers[pidx] is None or self.send_buffers[pidx].size < required_send_size:
            if self.log:
                self.log.debug(f"Rank {self.rank}: Allocating send_buffer[{pidx}] "
                               f"(Iallgatherv) size {required_send_size/1e9:.3f} GB")
            self.send_buffers[pidx] = np.empty((my_send_count, nvir, nvir, nvir), dtype=self.dtype)

        send_data = self.send_buffers[pidx][:my_send_count]

        # 5. Fill Send Data
        if my_send_count > 0 and _require_t3_c("fill_local_data_ijk_", obj=self):
            my_requests_arr = np.array(my_indices, dtype=np.int32)
            fill_local_data_ijk_(t3_local, send_data, my_requests_arr, self.global_to_local_offset, my_send_count, nvir)
        else:
            for idx, ijk_idx in enumerate(my_indices):
                local_idx = self.global_to_local_offset[ijk_idx]
                if local_idx >= 0:
                    send_data[idx] = t3_local[local_idx]

        # 6. Issue Iallgatherv
        t0_coll = logger.perf_counter()

        # MPI buffers
        send_buf = send_data.reshape(-1) # Flatten view
        recv_buf = recv_data.reshape(-1) # Flatten view

        if not hasattr(self, 'triple_type'):
            block_len = nvir * nvir * nvir
            self.triple_type = MPI.DOUBLE.Create_contiguous(block_len)
            self.triple_type.Commit()
            self._log_communication(f"Created custom MPI type for triples (block_len={block_len})")

        req = self.comm.Iallgatherv([send_buf, my_send_count, self.triple_type],
                                    [recv_buf, counts, displs, self.triple_type])
        reqs = [req]

        t1_issue = logger.perf_counter()
        self._log_communication(f"Prefetch (Iallgatherv): issue time: {t1_issue-t0_coll:.4f} sec.")

        return {
            'reqs': reqs,
            'recv_data': recv_data,
            'send_data': send_data,
            'triples_by_owner': triples_by_owner,
            'ijk_triples': ijk_triples,
            'counts': counts,
            't0_coll': t0_coll,
            'is_allgather': True
        }

    def finalize_prefetch_t3_triples(self, handle, t3_local, ijk_triples):
        """
        Finalize a non-blocking prefetch and return data in the same format as collect_t3_triples.

        Parameters
        ----------
        handle : dict or None
            Handle returned by prefetch_t3_triples. None for single-rank.
        t3_local : ndarray
            Local T3 storage (used only when handle is None, i.e. single-rank fallback)
        ijk_triples : ndarray of shape (n_tri, 3)
            The same triples passed to prefetch_t3_triples

        Returns
        -------
        t3_out : ndarray of shape (n_tri, nvir, nvir, nvir)
        ijk_reordered : ndarray of shape (n_tri, 3)
        """
        nvir = self.nvir
        n_tri = len(ijk_triples)

        if n_tri == 0:
            return (np.empty((0, nvir, nvir, nvir), dtype=self.dtype), np.empty((0, 3), dtype=np.int32))

        if handle is None:
            raise RuntimeError("Single-rank fallback not implemented. It should never be reached.")

        # Wait for all communication to finish
        requests = handle['reqs']

        t0_finalize = logger.perf_counter()
        MPI.Request.Waitall(requests)
        t1_wait = logger.perf_counter()

        # Build reordered ijk metadata matching the received-data order.
        recv_data = handle['recv_data']
        triples_by_owner = handle['triples_by_owner']
        ijk_reordered = np.array([triple for r in range(self.size) for triple in triples_by_owner[r]], dtype=np.int32)
        t1_finalize = logger.perf_counter()
        self._log_communication(f"Prefetch: comm (issue+wait) time: {t1_wait - handle['t0_coll']:.4f} sec.")
        self._log_communication(f"Prefetch: finalize time: {t1_finalize - t0_finalize:.4f} sec.")

        return recv_data, ijk_reordered

    def print_distribution_info(self):
        """Print distribution information."""
        if self.rank == 0:
            print(f"\nDistributedT3IJK Configuration:")
            print(f"  nocc = {self.nocc}, nvir = {self.nvir}")
            print(f"  Distribution strategy: {self.distribution}")
            print(f"  Total canonical triples: {self.nocc3}")
            print(f"  Number of MPI ranks: {self.size}")
            print(f"  C library loaded: {_HAS_C_LIB}")
            print(f"\nDistribution by rank:")

        self.comm.Barrier()

        for r in range(self.size):
            if self.rank == r:
                n_triples = len(self.rank_triples[r])
                pct = 100.0 * n_triples / self.nocc3 if self.nocc3 > 0 else 0
                print(f"  Rank {r:3d}: {n_triples:6d} triples ({pct:.1f}%)")
            self.comm.Barrier()

    def memory_usage_bytes(self):
        itemsize = np.dtype(self.dtype).itemsize
        return self.local_nocc3 * self.nvir**3 * itemsize

    def memory_usage_gb(self):
        return self.memory_usage_bytes() / (1024**3)

    def save_to_disk(self, t3_local, prefix):
        """
        Save T3 local data and class state to disk.

        Each rank saves its own t3_local to prefix_t3_local_{rank}.npy
        Rank 0 saves the metadata needed to reconstruct the object.

        Parameters
        ----------
        t3_local : ndarray
            The local T3 amplitude array for this rank
        prefix : str
            Path prefix for output files (e.g., '/path/to/checkpoint_')
        """
        # Each rank saves its own t3_local
        t3_filename = f"{prefix}t3_local_{self.rank}.npy"
        np.save(t3_filename, t3_local)

        # Each rank saves its own (i,j,k) triples to a separate file
        ijk_filename = f"{prefix}ijk_triples_{self.rank}.npy"
        ijk_array = np.array(self.local_triples, dtype=np.int32)
        np.save(ijk_filename, ijk_array)

        # Rank 0 saves scalar metadata as JSON
        if self.rank == 0:
            metadata = {
                'nocc': int(self.nocc),
                'nvir': int(self.nvir),
                'distribution': str(self.distribution),
                'batch_size': None if self.batch_size is None else int(self.batch_size),
                'dtype': np.dtype(self.dtype).str,
                'size': int(self.size),
            }
            meta_filename = f"{prefix}metadata.json"
            with open(meta_filename, 'w') as f:
                json.dump(metadata, f)

        self.comm.Barrier()

    @classmethod
    def load_from_disk(cls, prefix, comm, batch_size=None, mmap_mode=None):
        """
        Load T3 data and reconstruct DistributedT3IJK object from disk.
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Load metadata
        meta_filename = f"{prefix}metadata.json"
        with open(meta_filename, 'r') as f:
            metadata = json.load(f)

        nocc = metadata['nocc']
        nvir = metadata['nvir']
        distribution = metadata['distribution']
        dtype = np.dtype(metadata['dtype'])
        saved_size = metadata['size']

        if size != saved_size:
            raise ValueError(f"MPI size mismatch: saved with {saved_size} ranks, "
                            f"but loading with {size} ranks. Redistribution not yet supported.")

        # Load saved (i,j,k) triples for all ranks
        saved_rank_triples = []
        for r in range(size):
            ijk_filename = f"{prefix}ijk_triples_{r}.npy"
            ijk_array = np.load(ijk_filename)
            saved_rank_triples.append([tuple(triple) for triple in ijk_array])

        # Create object with minimal initialization, then override distribution
        dist_t3 = cls.__new__(cls)
        dist_t3.log = None
        dist_t3.log_t3_communication = False
        dist_t3.nocc = nocc
        dist_t3.nvir = nvir
        dist_t3.dtype = dtype
        dist_t3.comm = comm
        dist_t3.distribution = distribution
        dist_t3.batch_size = metadata.get('batch_size', batch_size)
        dist_t3.nocc3 = nocc * (nocc + 1) * (nocc + 2) // 6
        dist_t3.rank = rank
        dist_t3.size = size

        # Use saved distribution directly
        dist_t3.rank_triples = saved_rank_triples
        dist_t3.local_triples = saved_rank_triples[rank]

        # Build mappings from saved distribution
        all_triples = dist_t3._enumerate_ijk_triples()
        dist_t3._finalize_distribution_from_saved(all_triples)
        dist_t3._build_local_mapping()
        dist_t3._setup_permutation_maps()

        # Load t3_local
        t3_filename = f"{prefix}t3_local_{rank}.npy"
        t3_local = np.load(t3_filename, mmap_mode=mmap_mode)

        return dist_t3, t3_local

    def _finalize_distribution_from_saved(self, all_triples):
        """Finalize distribution mappings from saved data (no recomputation)."""
        # Build global triple to rank mapping
        self.ijk_to_rank = {}
        for r in range(self.size):
            for triple in self.rank_triples[r]:
                triple = tuple(int(x) for x in triple)
                self.ijk_to_rank[triple] = r

        # Build ijk_idx_to_rank mapping
        self.ijk_idx_to_rank = {}
        for triple, rank in self.ijk_to_rank.items():
            i, j, k = triple
            ijk_idx = ijk_to_linear(i, j, k, self.nocc)
            self.ijk_idx_to_rank[ijk_idx] = rank

        # Build global triple to linear index mapping
        self.ijk_to_global_idx = {}
        for idx, triple in enumerate(all_triples):
            triple = tuple(int(x) for x in triple)
            self.ijk_to_global_idx[triple] = idx

        # Build local index mapping
        self.global_to_local_offset = np.full(len(all_triples), -1, dtype=np.int64)
        for local_idx, triple in enumerate(self.local_triples):
            triple = tuple(int(x) for x in triple)
            global_idx = self.ijk_to_global_idx[triple]
            ijk_idx = ijk_to_linear(triple[0], triple[1], triple[2], self.nocc)
            self.global_to_local_offset[ijk_idx] = local_idx


def generate_deterministic_ijk_distribution(nocc, nvir, nranks, distribution='balanced', batch_size=None):
    """Return deterministic rank-owned canonical (i,j,k) triples."""
    return DistributedT3IJK.generate_distribution(
        nocc, nvir, nranks, distribution=distribution, batch_size=batch_size)


class DistributedT3ABC:
    """
    Distributed triangular T3 amplitude storage with (a,b,c) block distribution.

    Stores T[a <= b <= c, i, j, k] where (a,b,c) triplets are distributed across ranks.
    """

    def __init__(self, nocc, nvir, comm, blksize_abc=None, dtype=np.float64):
        self.nocc = nocc
        self.nvir = nvir
        self.dtype = dtype
        self.comm = comm

        self.n_abc_triplets = nvir * (nvir + 1) * (nvir + 2) // 6
        self.nvir3 = self.n_abc_triplets

        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self._compute_abc_distribution(blksize_abc)
        self._build_local_abc_mapping()

    def _enumerate_abc_triplets(self):
        """Enumerate all (a,b,c) triplets with a <= b <= c."""
        triplets = []
        for a in range(self.nvir):
            for b in range(a, self.nvir):
                for c in range(b, self.nvir):
                    triplets.append((a, b, c))
        return triplets

    def _compute_abc_distribution(self, blksize_abc):
        """Distribute (a,b,c) triplets across ranks using block-cyclic distribution."""
        if blksize_abc is None:
            blksize_abc = 1

        all_abc = self._enumerate_abc_triplets()

        self.rank_abc_triplets = [[] for _ in range(self.size)]
        self.abc_to_rank = {}
        self.abc_to_global_idx = {}

        # 1. Enumerate and map global index
        for idx, abc in enumerate(all_abc):
            self.abc_to_global_idx[abc] = idx

        # 2. Block-cyclic distribution
        n_blk = (self.nvir + blksize_abc - 1) // blksize_abc

        block_owners = {}
        block_list = []
        for A in range(n_blk):
            for B in range(A, n_blk):
                for C in range(B, n_blk):
                    block_list.append((A, B, C))

        # Distribute blocks round-robin
        for idx, blk_idx in enumerate(block_list):
            owner = idx % self.size
            block_owners[blk_idx] = owner

        # 3. Assign triplets to owners
        for abc in all_abc:
            a, b, c = abc
            blk_idx = (a // blksize_abc, b // blksize_abc, c // blksize_abc)
            owner = block_owners[blk_idx]
            self.rank_abc_triplets[owner].append(abc)
            self.abc_to_rank[abc] = owner

        self.local_abc_triplets = self.rank_abc_triplets[self.rank]

        # 4. Precompute idx -> abc map for O(1) reverse lookup
        if self.nvir < 256:
            dtype_idx = np.uint8
        elif self.nvir < 65536:
            dtype_idx = np.uint16
        else:
            dtype_idx = np.uint32

        self.idx_to_abc_map = np.empty((len(all_abc), 3), dtype=dtype_idx)
        for idx, abc in enumerate(all_abc):
            self.idx_to_abc_map[idx] = abc

    def _build_local_abc_mapping(self):
        """Build local storage mappings."""
        self.local_abc_offset = {}
        self.local_nvir3 = len(self.local_abc_triplets)

        # Build global mapping table for C function: [global_idx] -> local_offset (-1 if not local)
        self.global_mapping_table = np.full(self.n_abc_triplets, -1, dtype=np.int64)

        for idx, (a, b, c) in enumerate(self.local_abc_triplets):
            self.local_abc_offset[(a, b, c)] = idx
            global_idx = self.abc_to_global_idx[(a, b, c)]
            self.global_mapping_table[global_idx] = idx

    def get_owner_abc(self, a, b, c):
        """Get rank that owns (a,b,c) triplet."""
        abc_sorted = tuple(sorted([a, b, c]))
        return self.abc_to_rank.get(abc_sorted, -1)

    def abc_idx_to_abc(self, abc_idx):
        """Convert global abc index to (a,b,c) triplet."""
        if 0 <= abc_idx < len(self.idx_to_abc_map):
            return tuple(self.idx_to_abc_map[abc_idx])
        raise ValueError(f"Invalid abc_idx: {abc_idx}")

    def get_local_index(self, a, b, c, i, j, k):
        """Get local storage index for (a,b,c,i,j,k)."""
        abc_sorted = tuple(sorted([a, b, c]))
        offset = self.local_abc_offset.get(abc_sorted)
        if offset is None:
            return None
        ijk_idx = i * self.nocc * self.nocc + j * self.nocc + k
        return offset * (self.nocc ** 3) + ijk_idx

    def allocate_local(self):
        """Allocate local storage."""
        return np.zeros((self.local_nvir3, self.nocc, self.nocc, self.nocc), dtype=self.dtype)

    def memory_usage_gb(self):
        """Return memory usage in GB."""
        itemsize = np.dtype(self.dtype).itemsize
        return self.local_nvir3 * (self.nocc ** 3) * itemsize / (1024**3)


#############################
# IJK -> ABC redistribution #
#############################

def _precompute_abc_maps(dt3_abc):
    """
    Precompute masks and maps for all 6 permutations of (a,b,c).
    Returns: list of (mask, dest_ranks, abc_indices, perm_order)
    """
    nvir = dt3_abc.nvir
    a, b, c = np.mgrid[0:nvir, 0:nvir, 0:nvir]

    # 6 permutations of (a,b,c) and corresponding (i,j,k) permutation
    perms = [
        ((a, b, c), (0, 1, 2)),  # abc -> no swap
        ((a, c, b), (0, 2, 1)),  # acb -> swap j,k
        ((b, a, c), (1, 0, 2)),  # bac -> swap i,j
        ((b, c, a), (1, 2, 0)),  # bca -> cyclical
        ((c, a, b), (2, 0, 1)),  # cab -> cyclical
        ((c, b, a), (2, 1, 0))   # cba -> reverse
    ]

    cached = []

    for (pa, pb, pc), perm_order in perms:
        valid_mask = (pa <= pb) & (pb <= pc)

        va = pa[valid_mask]
        vb = pb[valid_mask]
        vc = pc[valid_mask]

        n_valid = len(va)
        dest_ranks = np.empty(n_valid, dtype=np.int32)
        abc_indices = np.empty(n_valid, dtype=np.int32)

        for idx in range(n_valid):
            abc_tuple = (va[idx], vb[idx], vc[idx])
            dest_ranks[idx] = dt3_abc.get_owner_abc(*abc_tuple)
            abc_indices[idx] = dt3_abc.abc_to_global_idx[abc_tuple]

        cached.append((valid_mask, dest_ranks, abc_indices, perm_order))

    dt3_abc._cached_maps = cached


def _compute_chunk_size(nocc, nvir, size):
    """
    Compute optimal chunk size based on available memory.
    Keep send buffers under ~1 GB per rank.
    """
    bytes_per_element = 8  # float64
    values_per_triple = nvir**3 * 6
    values_per_triple_per_rank = values_per_triple / size
    target_buffer_size = 1 * 1024**3  # 1 GB
    chunk_size = max(1, int(target_buffer_size / (values_per_triple_per_rank * bytes_per_element)))
    chunk_size = max(1, min(chunk_size, 1000))
    return chunk_size


def _build_send_buffers_for_chunk_ijk(dt3_ijk, dt3_abc, t3_local_ijk, chunk_triples, nocc, nvir, size):
    """
    Build send buffers for a chunk of (i,j,k) triples.
    Adapted for IJK distribution where each triple is independent.

    For each triple (i,j,k) with data block T[a,b,c], we generate 6 permutations:
    T(p(i,j,k), p(a,b,c)) where p(a,b,c) is canonical (pa <= pb <= pc).
    """
    send_data = [[] for _ in range(size)]

    if not hasattr(dt3_abc, '_cached_maps'):
        _precompute_abc_maps(dt3_abc)

    cached_maps = dt3_abc._cached_maps

    for local_idx, (i, j, k) in chunk_triples:
        # Shape: (nvir, nvir, nvir)
        block = t3_local_ijk[local_idx]

        for p_idx, (valid_mask, dest_ranks, abc_indices, perm_order) in enumerate(cached_maps):
            values = block[valid_mask]  # (n_valid,)

            n_valid = values.shape[0]
            if n_valid == 0:
                continue

            idx_i, idx_j, idx_k = perm_order
            ijk = [i, j, k]
            pi = ijk[idx_i]
            pj = ijk[idx_j]
            pk = ijk[idx_k]

            # Bucket by destination rank
            for r in range(size):
                col_mask = (dest_ranks == r)
                if not np.any(col_mask):
                    continue

                vals_r = values[col_mask]
                abc_r = abc_indices[col_mask]
                n_cols = vals_r.shape[0]

                I_r = np.full(n_cols, pi, dtype=np.int32)
                J_r = np.full(n_cols, pj, dtype=np.int32)
                K_r = np.full(n_cols, pk, dtype=np.int32)

                send_data[r].append((abc_r, I_r, J_r, K_r, vals_r))

    return send_data


def _flatten_send_data(send_data):
    """
    Flatten send data into a 1D array.
    Format: [abc_idx, i, j, k, value, abc_idx, i, j, k, value, ...]
    """
    total_entries_count = 0
    for r_data in send_data:
        for block in r_data:
            total_entries_count += len(block[0])

    send_buffer = np.empty(total_entries_count * 5, dtype=np.float64)

    idx = 0
    for r_data in send_data:
        if not r_data:
            continue

        if len(r_data) == 1:
            abc, i, j, k, val = r_data[0]
        else:
            components = list(zip(*r_data))
            abc = np.concatenate(components[0])
            i = np.concatenate(components[1])
            j = np.concatenate(components[2])
            k = np.concatenate(components[3])
            val = np.concatenate(components[4])

        n = len(abc)
        if n == 0:
            continue

        if _require_t3_c("pack_interleaved"):
            pack_interleaved(n, abc, i, j, k, val, send_buffer[idx:])
        else:
            send_buffer[idx : idx + n*5 : 5] = abc
            send_buffer[idx+1 : idx + n*5 : 5] = i
            send_buffer[idx+2 : idx + n*5 : 5] = j
            send_buffer[idx+3 : idx + n*5 : 5] = k
            send_buffer[idx+4 : idx + n*5 : 5] = val

        idx += n * 5

    return send_buffer


def _unpack_received_data(recv_buffer, recv_counts, recv_displs, t3_local_abc, dt3_abc, nocc, nvir, size, comm):
    """
    Unpack received data and accumulate into target storage.
    Each received entry is: (abc_idx, i, j, k, value)
    """
    rank = comm.Get_rank()

    if int(np.sum(recv_counts)) == 0:
        return

    if _require_t3_c("unpack_received_data_indices"):
        unpack_received_data_indices(recv_buffer, recv_counts, recv_displs, t3_local_abc,
                                    dt3_abc.global_mapping_table, size, nocc, nvir)
        return

    # Fallback to Python
    for r in range(size):
        if recv_counts[r] == 0:
            continue

        start = recv_displs[r]
        end = start + recv_counts[r]
        chunk = recv_buffer[start:end]

        n_entries = recv_counts[r] // 5
        for e in range(n_entries):
            base = e * 5
            abc_idx = int(chunk[base])
            i = int(chunk[base + 1])
            j = int(chunk[base + 2])
            k = int(chunk[base + 3])
            value = chunk[base + 4]

            a, b, c = dt3_abc.abc_idx_to_abc(abc_idx)
            abc_sorted = tuple(sorted([a, b, c]))

            offset = dt3_abc.local_abc_offset.get(abc_sorted)
            if offset is not None:
                t3_local_abc[offset, i, j, k] = value

def _precompute_abc_pack_mappings(dt3_abc):
    """
    Precompute arrays for C-level packing.
    """
    nvir = dt3_abc.nvir
    size = dt3_abc.size
    a, b, c = np.mgrid[0:nvir, 0:nvir, 0:nvir]

    perms = [
        ((a, b, c), 0),  # (0, 1, 2)
        ((a, c, b), 1),  # (0, 2, 1)
        ((b, a, c), 2),  # (1, 0, 2)
        ((b, c, a), 3),  # (1, 2, 0)
        ((c, a, b), 4),  # (2, 0, 1)
        ((c, b, a), 5)   # (2, 1, 0)
    ]

    flat_idx = np.arange(nvir**3, dtype=np.int32).reshape(nvir, nvir, nvir)

    rank_lists_idx = [[] for _ in range(size)]
    rank_lists_g_idx = [[] for _ in range(size)]
    rank_lists_perm = [[] for _ in range(size)]

    for (pa, pb, pc), p_type in perms:
        valid_mask = (pa <= pb) & (pb <= pc)

        valid_flat_idx = flat_idx[valid_mask]

        va = pa[valid_mask]
        vb = pb[valid_mask]
        vc = pc[valid_mask]

        n_valid = len(va)
        if n_valid == 0:
            continue

        dest_ranks = np.empty(n_valid, dtype=np.int32)
        g_indices = np.empty(n_valid, dtype=np.int32)

        for i in range(n_valid):
            abc_tuple = (va[i], vb[i], vc[i])
            dest_ranks[i] = dt3_abc.get_owner_abc(*abc_tuple)
            g_indices[i] = dt3_abc.abc_to_global_idx[abc_tuple]

        for r in range(size):
            r_mask = (dest_ranks == r)
            if np.any(r_mask):
                rank_lists_idx[r].append(valid_flat_idx[r_mask])
                rank_lists_g_idx[r].append(g_indices[r_mask])
                rank_lists_perm[r].append(np.full(np.count_nonzero(r_mask), p_type, dtype=np.int32))

    rank_offsets = np.zeros(size + 1, dtype=np.int32)
    all_pack_idx = []
    all_pack_g_idx = []
    all_pack_perm = []

    for r in range(size):
        if len(rank_lists_idx[r]) > 0:
            all_pack_idx.append(np.concatenate(rank_lists_idx[r]))
            all_pack_g_idx.append(np.concatenate(rank_lists_g_idx[r]))
            all_pack_perm.append(np.concatenate(rank_lists_perm[r]))
        rank_offsets[r+1] = sum(len(x) for x in rank_lists_idx[r]) + rank_offsets[r]

    dt3_abc._c_pack_idx = (np.concatenate(all_pack_idx).astype(np.int64)
                            if all_pack_idx else np.array([], dtype=np.int64))
    dt3_abc._c_pack_g_idx = (np.concatenate(all_pack_g_idx).astype(np.int64)
                            if all_pack_g_idx else np.array([], dtype=np.int64))
    dt3_abc._c_pack_perm = (np.concatenate(all_pack_perm).astype(np.int64)
                            if all_pack_perm else np.array([], dtype=np.int64))
    dt3_abc._c_rank_offsets = rank_offsets.astype(np.int64)


def _build_send_buffers_for_chunk_ijk_c(dt3_ijk, dt3_abc, t3_local_ijk, chunk_triples, nocc, nvir, size):
    """
    Build send buffers for a chunk of (i,j,k) triples natively in C.
    Bypasses dynamic python list allocations and numpy fragmentation.
    """
    if not hasattr(dt3_abc, '_c_pack_idx'):
        _precompute_abc_pack_mappings(dt3_abc)

    n_triples = len(chunk_triples)
    if n_triples == 0:
        return np.empty(0, dtype=np.float64), np.zeros(size, dtype=np.int32)

    chunk_i = np.empty(n_triples, dtype=np.int64)
    chunk_j = np.empty(n_triples, dtype=np.int64)
    chunk_k = np.empty(n_triples, dtype=np.int64)

    first_local_idx = chunk_triples[0][0]
    contiguous = True
    for idx_offset, (local_idx, _) in enumerate(chunk_triples):
        if local_idx != first_local_idx + idx_offset:
            contiguous = False
            break

    if contiguous:
        t3_chunk_view = t3_local_ijk[first_local_idx : first_local_idx + n_triples]
        if not t3_chunk_view.flags['C_CONTIGUOUS']:
            t3_chunk_view = np.ascontiguousarray(t3_chunk_view)
    else:
        t3_chunk_view = np.ascontiguousarray([t3_local_ijk[idx] for idx, _ in chunk_triples])

    for idx, (local_idx, (i, j, k)) in enumerate(chunk_triples):
        chunk_i[idx] = i
        chunk_j[idx] = j
        chunk_k[idx] = k

    # The total number of permutations mapped across all arrays
    entries_per_triple = dt3_abc._c_rank_offsets[-1]

    # The C code writes everything natively to the pre-flatted send_buffer directly.
    send_buffer = np.empty(n_triples * entries_per_triple * 5, dtype=np.float64)

    pack_redistributed_send_buffer_c(n_triples, nvir, size, chunk_i, chunk_j, chunk_k, t3_chunk_view,
                                    dt3_abc._c_rank_offsets, dt3_abc._c_pack_idx,
                                    dt3_abc._c_pack_g_idx, dt3_abc._c_pack_perm, send_buffer)

    send_counts = np.empty(size, dtype=np.int32)
    for r in range(size):
        entries_per_triple = dt3_abc._c_rank_offsets[r+1] - dt3_abc._c_rank_offsets[r]
        send_counts[r] = entries_per_triple * n_triples

    return send_buffer, send_counts


def redistribute_ijk_to_abc(dt3_ijk, t3_local_ijk, comm, blksize_abc=None, chunk_size=None):
    """
    C-accelerated redistribution of T3 amplitudes from (i <= j <= k, a,b,c) to (a <= b <= c, i,j,k).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    nocc = dt3_ijk.nocc
    nvir = dt3_ijk.nvir

    log = logger.Logger(sys.stdout, 5)
    cput0 = (logger.process_clock(), logger.perf_counter())

    if rank == 0:
        log.info(f"\n{'='*70}")
        log.info(f"Starting C-accelerated IJK -> ABC redistribution")
        log.info(f"  nocc={nocc}, nvir={nvir}, nranks={size}")
        log.info(f"  Source: {dt3_ijk.memory_usage_gb():.2f} GB/rank (IJK distribution)")
        log.info(f"{'='*70}")

    dt3_abc = DistributedT3ABC(nocc, nvir, comm, blksize_abc=blksize_abc)
    t3_local_abc = dt3_abc.allocate_local()

    if chunk_size is None:
        chunk_size = _compute_chunk_size(nocc, nvir, size)

    local_triple_list = [(local_idx, (i, j, k)) for i, j, k, local_idx in dt3_ijk.local_ijk_list]
    n_local = len(local_triple_list)
    n_chunks_local = (n_local + chunk_size - 1) // chunk_size
    n_chunks_max = comm.allreduce(n_chunks_local, op=MPI.MAX)

    for chunk_idx in range(n_chunks_max):
        if rank == 0 and chunk_idx % max(1, n_chunks_max // 10) == 0:
            log.info(f"  Processing chunk {chunk_idx+1}/{n_chunks_max}...")

        cput1 = (logger.process_clock(), logger.perf_counter())
        start_idx = chunk_idx * chunk_size

        if chunk_idx < n_chunks_local:
            end_idx = min(start_idx + chunk_size, n_local)
            chunk_triples = local_triple_list[start_idx:end_idx]
        else:
            chunk_triples = []

        # C-accelerated buffer builds
        if _HAS_C_LIB:
            send_buffer, send_counts = _build_send_buffers_for_chunk_ijk_c(dt3_ijk, dt3_abc, t3_local_ijk,
                                                                            chunk_triples, nocc, nvir, size)
        else:
            warn_python_fallback("IJK-to-ABC redistribution packing", err=_NATIVE_IMPORT_ERROR)
            send_data = _build_send_buffers_for_chunk_ijk(dt3_ijk, dt3_abc, t3_local_ijk, chunk_triples,
                                                            nocc, nvir, size)
            send_buffer = _flatten_send_data(send_data)
            send_counts = np.zeros(size, dtype=np.int32)
            for r in range(size):
                if send_data[r]:
                    send_counts[r] = sum(len(block[0]) for block in send_data[r])

        # Exchange metadata
        recv_counts = np.empty(size, dtype=np.int32)
        comm.Alltoall(send_counts, recv_counts)

        send_displs = np.zeros(size, dtype=np.int32)
        send_displs[1:] = np.cumsum(send_counts)[:-1]

        recv_displs = np.zeros(size, dtype=np.int32)
        recv_displs[1:] = np.cumsum(recv_counts)[:-1]

        c_recv_counts = recv_counts.astype(np.int64)
        c_recv_displs = recv_displs.astype(np.int64)

        total_recv_entries = int(recv_counts.sum())
        recv_buffer = np.empty(total_recv_entries * 5, dtype=np.float64)

        block_dt = MPI.DOUBLE.Create_contiguous(5)
        block_dt.Commit()
        comm.Alltoallv([send_buffer, send_counts, send_displs, block_dt],
                       [recv_buffer, recv_counts, recv_displs, block_dt])
        block_dt.Free()

        _unpack_received_data(recv_buffer, c_recv_counts * 5, c_recv_displs * 5, t3_local_abc, dt3_abc,
                              nocc, nvir, size, comm)

        send_buffer = None
        recv_buffer = None

        if rank == 0:
            t_chunk = logger.perf_counter() - cput1[1]
            log.info(f"  Chunk {chunk_idx+1} finished in {t_chunk:.2f} s")

    comm.Barrier()
    if rank == 0:
        log.info(f"Redistribution complete in {logger.perf_counter() - cput0[1]:.2f} s")

    return dt3_abc, t3_local_abc
