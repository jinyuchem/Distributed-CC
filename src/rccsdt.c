/*
 * Copyright 2025-2026 The Distributed-CC Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Permutation table for T3 amplitudes */
static const int64_t tp_t3[6][3] = {
    {0, 1, 2}, /* Case 0: i <= j <= k */
    {0, 2, 1}, /* Case 1: i <= k < j */
    {1, 0, 2}, /* Case 2: j < i <= k */
    {1, 2, 0}, /* Case 3: k < i <= j */
    {2, 0, 1}, /* Case 4: j <= k < i */
    {2, 1, 0}, /* Case 5: k < j < i */
};

static inline int get_canonical_case(int64_t i, int64_t j, int64_t k, int64_t *ci, int64_t *cj, int64_t *ck) {
    if (i <= j && j <= k) {
        *ci = i;
        *cj = j;
        *ck = k;
        return 0;
    } else if (i <= k && k < j) {
        *ci = i;
        *cj = k;
        *ck = j;
        return 1;
    } else if (j < i && i <= k) {
        *ci = j;
        *cj = i;
        *ck = k;
        return 2;
    } else if (k < i && i <= j) {
        *ci = k;
        *cj = i;
        *ck = j;
        return 3;
    } else if (j <= k && k < i) {
        *ci = j;
        *cj = k;
        *ck = i;
        return 4;
    } else {
        *ci = k;
        *cj = j;
        *ck = i;
        return 5;
    }
}

static inline int64_t ijk_to_linear(int64_t i, int64_t j, int64_t k, int64_t nocc) {
    int64_t n_before_i = 0;
    for (int64_t ii = 0; ii < i; ii++) {
        n_before_i += (nocc - ii) * (nocc - ii + 1) / 2;
    }
    int64_t n_before_j = 0;
    for (int64_t jj = i; jj < j; jj++) {
        n_before_j += (nocc - jj);
    }
    return n_before_i + n_before_j + (k - j);
}

static void apply_perm_and_copy(const double *src, double *dst, int64_t nvir, int case_num) {
    int64_t nvir2 = nvir * nvir;
    const int64_t *perm = tp_t3[case_num];

#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t a = 0; a < nvir; a++) {
        for (int64_t b = 0; b < nvir; b++) {
            for (int64_t c = 0; c < nvir; c++) {
                int64_t abc[3] = {a, b, c};
                int64_t aa = abc[perm[0]];
                int64_t bb = abc[perm[1]];
                int64_t cc = abc[perm[2]];
                dst[(aa * nvir2) + (bb * nvir) + cc] = src[(a * nvir2) + (b * nvir) + c];
            }
        }
    }
}

void fill_local_data_ijk_(const double *t3_local, double *send_data, const int32_t *requests,
                          const int64_t *ijk_to_local_idx, int64_t n_requests, int64_t nvir) {
    int64_t nvir3 = nvir * nvir * nvir;
    size_t block_bytes = nvir3 * sizeof(double);

#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n_requests; r++) {
        int64_t local_idx = ijk_to_local_idx[requests[r]];
        if (local_idx >= 0) {
            memcpy(send_data + r * nvir3, t3_local + local_idx * nvir3, block_bytes);
        }
    }
}

void take_t3_ijk_single_(const double *t3_local, double *t3_blk, const int64_t *ijk_to_local_idx, int64_t i, int64_t j,
                         int64_t k, int64_t nocc, int64_t nvir) {
    int64_t nvir3 = nvir * nvir * nvir;
    int64_t ci, cj, ck;
    int case_num = get_canonical_case(i, j, k, &ci, &cj, &ck);
    int64_t ijk_idx = ijk_to_linear(ci, cj, ck, nocc);
    int64_t local_idx = ijk_to_local_idx[ijk_idx];
    if (local_idx < 0)
        return;
    apply_perm_and_copy(t3_local + local_idx * nvir3, t3_blk, nvir, case_num);
}

void t3_single_spin_summation_inplace_(double *A, int64_t nvir, char *pattern, double alpha, double beta) {
    int64_t nvv = nvir * nvir;
    double p0, p1, p2, p3, p4;

    if (strcmp(pattern, "P3_full") == 0) {
        p0 = 1.0;
        p1 = 1.0;
        p2 = 1.0;
        p3 = 1.0;
        p4 = 1.0;
    } else if (strcmp(pattern, "P3_422") == 0) {
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 2.0;
        p4 = -1.0;
    } else if (strcmp(pattern, "P3_201") == 0) {
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 1.0;
        p4 = 0.0;
    } else {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

    int64_t nabc = nvir * (nvir + 1) * (nvir + 2) / 6;

#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t n = 0; n < nabc; n++) {
        int64_t a = (int64_t)(cbrt(6.0 * n + 0.5));
        while (a * (a + 1) * (a + 2) / 6 > n)
            a--;
        while ((a + 1) * (a + 2) * (a + 3) / 6 <= n)
            a++;

        int64_t rem = n - a * (a + 1) * (a + 2) / 6;
        int64_t b = (int64_t)(sqrt(2.0 * rem + 0.25) - 0.5);
        if (b > a)
            b = a;
        while (b * (b + 1) / 2 > rem)
            b--;
        while ((b + 1) * (b + 2) / 2 <= rem && b + 1 <= a)
            b++;

        int64_t c = rem - b * (b + 1) / 2;

        if (a > b && b > c) {
            double T_local[6];
            int64_t idx_abc = a * nvv + b * nvir + c;
            int64_t idx_cba = c * nvv + b * nvir + a;
            int64_t idx_acb = a * nvv + c * nvir + b;
            int64_t idx_bac = b * nvv + a * nvir + c;
            int64_t idx_bca = b * nvv + c * nvir + a;
            int64_t idx_cab = c * nvv + a * nvir + b;

            T_local[0] = p0 * A[idx_abc] + p1 * A[idx_cba] + p2 * A[idx_bac];
            T_local[1] = p0 * A[idx_cba] + p1 * A[idx_abc] + p2 * A[idx_bca];
            T_local[2] = p0 * A[idx_acb] + p1 * A[idx_bca] + p2 * A[idx_cab];
            T_local[3] = p0 * A[idx_bac] + p1 * A[idx_cab] + p2 * A[idx_abc];
            T_local[4] = p0 * A[idx_bca] + p1 * A[idx_acb] + p2 * A[idx_cba];
            T_local[5] = p0 * A[idx_cab] + p1 * A[idx_bac] + p2 * A[idx_acb];

            A[idx_abc] = beta * A[idx_abc] + alpha * (p3 * T_local[0] + p4 * T_local[2]);
            A[idx_cba] = beta * A[idx_cba] + alpha * (p3 * T_local[1] + p4 * T_local[5]);
            A[idx_acb] = beta * A[idx_acb] + alpha * (p3 * T_local[2] + p4 * T_local[0]);
            A[idx_bac] = beta * A[idx_bac] + alpha * (p3 * T_local[3] + p4 * T_local[4]);
            A[idx_bca] = beta * A[idx_bca] + alpha * (p3 * T_local[4] + p4 * T_local[3]);
            A[idx_cab] = beta * A[idx_cab] + alpha * (p3 * T_local[5] + p4 * T_local[1]);
        } else if (a > b && b == c) {
            double T_local[3];
            int64_t idx_abb = a * nvv + b * nvir + b;
            int64_t idx_bba = b * nvv + b * nvir + a;
            int64_t idx_bab = b * nvv + a * nvir + b;

            T_local[0] = p0 * A[idx_abb] + p1 * A[idx_bba] + p2 * A[idx_bab];
            T_local[1] = p0 * A[idx_bba] + p1 * A[idx_abb] + p2 * A[idx_bba];
            T_local[2] = p0 * A[idx_bab] + p1 * A[idx_bab] + p2 * A[idx_abb];

            A[idx_abb] = beta * A[idx_abb] + alpha * (p3 * T_local[0] + p4 * T_local[0]);
            A[idx_bba] = beta * A[idx_bba] + alpha * (p3 * T_local[1] + p4 * T_local[2]);
            A[idx_bab] = beta * A[idx_bab] + alpha * (p3 * T_local[2] + p4 * T_local[1]);
        } else if (a == b && b > c) {
            double T_local[3];
            int64_t idx_aac = a * nvv + a * nvir + c;
            int64_t idx_caa = c * nvv + a * nvir + a;
            int64_t idx_aca = a * nvv + c * nvir + a;

            T_local[0] = p0 * A[idx_aac] + p1 * A[idx_caa] + p2 * A[idx_aac];
            T_local[1] = p0 * A[idx_caa] + p1 * A[idx_aac] + p2 * A[idx_aca];
            T_local[2] = p0 * A[idx_aca] + p1 * A[idx_aca] + p2 * A[idx_caa];

            A[idx_aac] = beta * A[idx_aac] + alpha * (p3 * T_local[0] + p4 * T_local[2]);
            A[idx_caa] = beta * A[idx_caa] + alpha * (p3 * T_local[1] + p4 * T_local[1]);
            A[idx_aca] = beta * A[idx_aca] + alpha * (p3 * T_local[2] + p4 * T_local[0]);
        } else {
            int64_t idx_aaa = a * nvv + a * nvir + a;
            double T_local = (p0 + p1 + p2) * A[idx_aaa];
            A[idx_aaa] = beta * A[idx_aaa] + alpha * (p3 + p4) * T_local;
        }
    }
}

#include <stdint.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

void t3_spin_summation_triple_sym_(const double *restrict A, double *restrict B0, double *restrict B1,
                                    double *restrict B2, int64_t nvir) {
    const int64_t nvv = nvir * nvir;

#pragma omp parallel
    {
        double *col_buf = (double *)malloc(nvir * sizeof(double));

#pragma omp for schedule(static)
        for (int64_t a = 0; a < nvir; a++) {
            for (int64_t b = 0; b < nvir; b++) {
                const double *A_a = A + a * nvv;
                for (int64_t k = 0; k < nvir; k++) {
                    col_buf[k] = A_a[k * nvir + b];
                }

                const double *T_ab = A + a * nvv + b * nvir;
                const double *T_ba = A + b * nvv + a * nvir;
                const double *T_cba_base = A + b * nvir + a;

                double *d0 = B0 + a * nvv + b * nvir;
                double *d1 = B1 + a * nvv + b * nvir;
                double *d2 = B2 + a * nvv + b * nvir;

                int64_t c = 0;
#ifdef __AVX2__
                for (; c + 4 <= nvir; c += 4) {
                    __m256d v_abc = _mm256_loadu_pd(T_ab + c);
                    __m256d v_bac = _mm256_loadu_pd(T_ba + c);
                    __m256d v_acb = _mm256_loadu_pd(col_buf + c);

                    __m256d v_cba = _mm256_set_pd(T_cba_base[(c + 3) * nvv], T_cba_base[(c + 2) * nvv],
                                                  T_cba_base[(c + 1) * nvv], T_cba_base[c * nvv]);

                    __m256d two_abc = _mm256_add_pd(v_abc, v_abc);

                    // v0 = 2*T_abc - T_acb - T_cba
                    __m256d r0 = _mm256_sub_pd(two_abc, _mm256_add_pd(v_acb, v_cba));
                    _mm256_storeu_pd(d0 + c, r0);

                    // v1 = 2*T_abc - T_bac - T_acb
                    __m256d r1 = _mm256_sub_pd(two_abc, _mm256_add_pd(v_bac, v_acb));
                    _mm256_storeu_pd(d1 + c, r1);

                    // v2 = v0 + T_acb - T_bac
                    __m256d r2 = _mm256_add_pd(r0, _mm256_sub_pd(v_acb, v_bac));
                    _mm256_storeu_pd(d2 + c, r2);
                }
#endif
                for (; c < nvir; c++) {
                    double t_abc = T_ab[c];
                    double t_bac = T_ba[c];
                    double t_acb = col_buf[c];
                    double t_cba = T_cba_base[c * nvv];

                    double two_t = 2.0 * t_abc;
                    d0[c] = two_t - t_acb - t_cba;
                    d1[c] = two_t - t_bac - t_acb;
                    d2[c] = d0[c] + t_acb - t_bac;
                }
            }
        }

        free(col_buf);
    }
}

/**
 * Compute B[a,b,c] = A[a,b,c] + A[c,a,b]
 *
 * A.transpose(1,2,0)[a,b,c] = A[c,a,b]
 *
 * A[a,b,c] is at index a*nvv + b*nvir + c     (sequential in c)
 * A[c,a,b] is at index c*nvv + a*nvir + b     (stride nvv in c)
 */
void t3_transpose_add_(const double *restrict A, double *restrict B, int64_t nvir) {
    const int64_t nvv = nvir * nvir;

#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t a = 0; a < nvir; a++) {
        for (int64_t b = 0; b < nvir; b++) {
            // B[a,b,c] = A[a,b,c] + A[c,a,b]
            // A[a,b,c] at a*nvv + b*nvir + c        (sequential in c)
            // A[c,a,b] at c*nvv + a*nvir + b        (stride nvv in c)

            double *d = B + a * nvv + b * nvir;
            const double *s1 = A + a * nvv + b * nvir; // A[a,b,:]  - sequential
            const double *s2 = A + a * nvir + b;       // A[:,a,b]  - stride nvv

            int64_t c = 0;

#ifdef __AVX512F__
            // AVX-512: 8 doubles at a time with gather
            __m512i idx_nvv = _mm512_set_epi64(7 * nvv, 6 * nvv, 5 * nvv, 4 * nvv, 3 * nvv, 2 * nvv, 1 * nvv, 0 * nvv);

            for (; c + 8 <= nvir; c += 8) {
                // Load A[a,b,c:c+8] - sequential
                __m512d v1 = _mm512_loadu_pd(s1 + c);

                // Gather A[c:c+8,a,b] - stride nvv
                __m512i indices = _mm512_add_epi64(idx_nvv, _mm512_set1_epi64(c * nvv));
                __m512d v2 = _mm512_i64gather_pd(indices, s2, 8);

                // Add and store
                __m512d sum = _mm512_add_pd(v1, v2);
                _mm512_storeu_pd(d + c, sum);
            }
#endif

#ifdef __AVX2__
            // AVX2: 4 doubles at a time
            for (; c + 4 <= nvir; c += 4) {
                // Load A[a,b,c:c+4] - sequential
                __m256d v1 = _mm256_loadu_pd(s1 + c);

                // Gather A[c:c+4,a,b] - stride nvv
                __m256d v2 = _mm256_set_pd(s2[(c + 3) * nvv], s2[(c + 2) * nvv], s2[(c + 1) * nvv], s2[c * nvv]);

                // Add and store
                __m256d sum = _mm256_add_pd(v1, v2);
                _mm256_storeu_pd(d + c, sum);
            }
#endif

            // Scalar cleanup
            for (; c < nvir; c++) {
                d[c] = s1[c] + s2[c * nvv];
            }
        }
    }
}

/*
 * Pack interleaved data for MPI send buffer
 *
 * n: Number of elements
 * abc, i, j, k: Index arrays (int32)
 * val: Value array (double)
 * dest: Destination buffer (double) - size 5 * n
 *       Format: [abc0, i0, j0, k0, val0, abc1, i1, j1, k1, val1, ...]
 */
void pack_interleaved(int64_t n, int32_t *abc, int32_t *i, int32_t *j, int32_t *k, double *val, double *dest) {
    for (int64_t x = 0; x < n; x++) {
        int64_t offset = x * 5;
        dest[offset + 0] = (double)abc[x];
        dest[offset + 1] = (double)i[x];
        dest[offset + 2] = (double)j[x];
        dest[offset + 3] = (double)k[x];
        dest[offset + 4] = val[x];
    }
}

/*
 * Unpack received data from Alltoallv buffer into t3_local_abc
 *
 * recv_buffer: Input buffer containing [(abc_idx, i, j, k, value), ...]
 *              Stored as double array, so indices are cast to double.
 * recv_counts: Number of doubles for each rank
 * recv_displs: Displacements for each rank in recv_buffer
 * t3_local_ptr: Pointer to local T3 array (local_nvir3, nocc, nocc, nocc)
 * global_mapping_table: Array mapping [abc_idx] -> local_offset.
 *                       Size = n_abc_triplets.
 *                       Value = offset if local, -1 if not local.
 * size: Number of MPI ranks
 * nocc, nvir: dimensions
 */
void unpack_received_data_indices(double *recv_buffer, int64_t *recv_counts, int64_t *recv_displs, double *t3_local_ptr,
                                  int64_t *global_mapping_table, int size, int64_t nocc, int64_t nvir) {
    int64_t stride_i = nocc * nocc;
    int64_t stride_j = nocc;
    int64_t stride_k = 1;
    int64_t block_vol = nocc * nocc * nocc;

    for (int r = 0; r < size; r++) {
        int64_t count = recv_counts[r];
        if (count == 0)
            continue;

        int64_t start = recv_displs[r];
        int64_t n_entries = count / 5;

        for (int e = 0; e < n_entries; e++) {
            int64_t base = start + e * 5;
            int64_t abc_idx = (int64_t)recv_buffer[base];
            int64_t i = (int64_t)recv_buffer[base + 1];
            int64_t j = (int64_t)recv_buffer[base + 2];
            int64_t k = (int64_t)recv_buffer[base + 3];
            double value = recv_buffer[base + 4];

            int64_t offset = global_mapping_table[abc_idx];

            if (offset != -1) {
                int64_t pos = offset * block_vol + i * stride_i + j * stride_j + k * stride_k;
                t3_local_ptr[pos] = value;
            }
        }
    }
}

/*
 * Pack redistributed T3 amplitudes from IJK to ABC flat buffer in C.
 *
 * This function iterates over destination ranks and sequentially builds the
 * `send_buffer` so it is already sorted by destination rank for MPI_Alltoallv.
 *
 * n_triples     : number of (i,j,k) triples in this chunk
 * nvir          : number of virtual orbitals
 * size          : number of MPI ranks
 * chunk_i       : array of `i` indices (length n_triples)
 * chunk_j       : array of `j` indices (length n_triples)
 * chunk_k       : array of `k` indices (length n_triples)
 * t3_chunk      : flattened chunk of T3 data, shape (n_triples, nvir^3)
 * rank_offsets  : offsets for each rank's precomputed mapping info, size (size
 * + 1) pack_abc_idx  : flattened valid flat `abc` indices, size (nvir^3)
 * pack_g_idx    : flattened corresponding global indices, size (nvir^3)
 * pack_perm     : flattened permutation types (0-5), size (nvir^3)
 * send_buffer   : flattened output double array, size (n_triples * nvir^3 * 5)
 */
void pack_redistributed_send_buffer_c(int64_t n_triples, int64_t nvir, int64_t size, int64_t *chunk_i, int64_t *chunk_j,
                                      int64_t *chunk_k, double *t3_chunk, int64_t *rank_offsets, int64_t *pack_abc_idx,
                                      int64_t *pack_g_idx, int64_t *pack_perm, double *send_buffer) {
    size_t nvir3 = (size_t)nvir * (size_t)nvir * (size_t)nvir;

#pragma omp parallel for schedule(dynamic)
    for (int64_t r = 0; r < size; r++) {
        int64_t start = rank_offsets[r];
        int64_t end = rank_offsets[r + 1];
        int64_t num_items = end - start;

        /* Exact memory offset for rank r's flat slice in the send sequence buffer
         */
        size_t out_idx = (size_t)start * (size_t)n_triples * 5;

        for (int64_t t = 0; t < n_triples; t++) {
            int64_t i = chunk_i[t];
            int64_t j = chunk_j[t];
            int64_t k = chunk_k[t];

            double *t3_blk = t3_chunk + (size_t)t * nvir3;

            for (int64_t idx = start; idx < end; idx++) {
                int64_t flat_abc = pack_abc_idx[idx];
                int64_t g_idx = pack_g_idx[idx];
                int64_t p = pack_perm[idx];

                int64_t pi, pj, pk;
                switch (p) {
                case 0:
                    pi = i;
                    pj = j;
                    pk = k;
                    break;
                case 1:
                    pi = i;
                    pj = k;
                    pk = j;
                    break;
                case 2:
                    pi = j;
                    pj = i;
                    pk = k;
                    break;
                case 3:
                    pi = j;
                    pk = i;
                    pj = j;
                    break; /* wait actually: it's not simply (2,0,1)? Let me check... */
                           /* No, the original python permutation is applied direct to elements
                            * of i,j,k: */
                }

                /* Recalibrating exactly from python:
                   (p==0) => (0,1,2) -> pi=i, pj=j, pk=k
                   (p==1) => (0,2,1) -> pi=i, pj=k, pk=j
                   (p==2) => (1,0,2) -> pi=j, pj=i, pk=k
                   (p==3) => (1,2,0) -> pi=j, pj=k, pk=i
                   (p==4) => (2,0,1) -> pi=k, pj=i, pk=j
                   (p==5) => (2,1,0) -> pi=k, pj=j, pk=i
                */
                if (p == 0) {
                    pi = i;
                    pj = j;
                    pk = k;
                } else if (p == 1) {
                    pi = i;
                    pj = k;
                    pk = j;
                } else if (p == 2) {
                    pi = j;
                    pj = i;
                    pk = k;
                } else if (p == 3) {
                    pi = j;
                    pj = k;
                    pk = i;
                } else if (p == 4) {
                    pi = k;
                    pj = i;
                    pk = j;
                } else {
                    pi = k;
                    pj = j;
                    pk = i;
                }

                /* Every rank pushes n_triples batches consisting of num_items */
                size_t actual_out_idx = out_idx + ((size_t)t * (size_t)num_items + (size_t)(idx - start)) * 5;

                send_buffer[actual_out_idx + 0] = (double)g_idx;
                send_buffer[actual_out_idx + 1] = (double)pi;
                send_buffer[actual_out_idx + 2] = (double)pj;
                send_buffer[actual_out_idx + 3] = (double)pk;
                send_buffer[actual_out_idx + 4] = t3_blk[flat_abc];
            }
        }
    }
}
