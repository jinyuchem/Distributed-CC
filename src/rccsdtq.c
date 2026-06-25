/* Copyright 2025-2026 The distr_cc Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Yu Jin <yjin@flatironinstitute.org>
 *         Huanchen Zhai <hczhai.ok@gmail.com>
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int64_t tp_t4[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
    {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
    {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
};

static inline int64_t idx4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t nvir, int64_t nvv, int64_t nvvv) {
    return a * nvvv + b * nvv + c * nvir + d;
}

static inline void swap4(int64_t in[4], int p, int q, int64_t out[4]) {
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];

    int64_t tmp = out[p];
    out[p] = out[q];
    out[q] = tmp;
}

static inline int same_tuple4(int64_t x[4], int64_t y[4]) {
    return x[0] == y[0] && x[1] == y[1] && x[2] == y[2] && x[3] == y[3];
}

static int find_tuple4(int64_t tuples[24][4], int ntuples, int64_t target[4]) {
    for (int i = 0; i < ntuples; i++) {
        if (same_tuple4(tuples[i], target))
            return i;
    }

    fprintf(stderr, "Error: tuple not found in orbit.\n");
    return -1;
}

static void apply_omega_local(double *y, const double *x, int64_t tuples[24][4], int ntuples) {
    int swap_pairs[6][2] = {
        {0, 1}, // ab
        {0, 2}, // ac
        {0, 3}, // ad
        {1, 2}, // bc
        {1, 3}, // bd
        {2, 3}  // cd
    };

    for (int i = 0; i < ntuples; i++) {
        y[i] = 0.0;

        for (int s = 0; s < 6; s++) {
            int64_t target[4];
            swap4(tuples[i], swap_pairs[s][0], swap_pairs[s][1], target);

            int j = find_tuple4(tuples, ntuples, target);
            y[i] += x[j];
        }
    }
}

static int build_unique_orbit(int64_t a, int64_t b, int64_t c, int64_t d, int64_t tuples[24][4]) {

    int64_t base[4] = {a, b, c, d};
    int ntuples = 0;

    for (int p = 0; p < 24; p++) {
        int64_t cand[4] = {base[tp_t4[p][0]], base[tp_t4[p][1]], base[tp_t4[p][2]], base[tp_t4[p][3]]};

        int duplicate = 0;
        for (int q = 0; q < ntuples; q++) {
            if (same_tuple4(tuples[q], cand)) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            tuples[ntuples][0] = cand[0];
            tuples[ntuples][1] = cand[1];
            tuples[ntuples][2] = cand[2];
            tuples[ntuples][3] = cand[3];
            ntuples++;
        }
    }
    return ntuples;
}

// omega_action_24[p][s] = orbit index q such that applying transposition s
// to perms[p](a,b,c,d) gives perms[q](a,b,c,d), for all-distinct (a,b,c,d).
// This is a property of the abstract S4 group, independent of actual values.
static int s_omega_action_24[24][6];
static int s_omega_action_24_ready = 0;

static void init_omega_action_24(void) {
    if (s_omega_action_24_ready)
        return;

    static const int swaps[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    // rev[p0][p1][p2][p3] = perm index (valid for all permutations of {0,1,2,3})
    int rev[4][4][4][4];
    for (int p = 0; p < 24; p++)
        rev[tp_t4[p][0]][tp_t4[p][1]][tp_t4[p][2]][tp_t4[p][3]] = p;

    for (int p = 0; p < 24; p++)
        for (int s = 0; s < 6; s++) {
            int t[4] = {tp_t4[p][0], tp_t4[p][1], tp_t4[p][2], tp_t4[p][3]};
            int i = swaps[s][0], j = swaps[s][1];
            int tmp = t[i];
            t[i] = t[j];
            t[j] = tmp;
            s_omega_action_24[p][s] = rev[t[0]][t[1]][t[2]][t[3]];
        }

    s_omega_action_24_ready = 1;
}

// Apply Omega to x[0..23] using the precomputed S4 action table.
static inline void apply_omega_24(double *restrict y, const double *restrict x) {
    for (int p = 0; p < 24; p++)
        y[p] = x[s_omega_action_24[p][0]] + x[s_omega_action_24[p][1]] + x[s_omega_action_24[p][2]] +
               x[s_omega_action_24[p][3]] + x[s_omega_action_24[p][4]] + x[s_omega_action_24[p][5]];
}

// Process one orbit: apply Q = 1 - P_[4] - P_[3,1] to the orbit of (a,b,c,d)
// in the ijkl-th virtual block at offset h.
//
// Fast path (a > b > c > d): uses precomputed S4 action table, no search.
// Slow path (degenerate orbit): uses build_unique_orbit + precomputed perm_map.
static inline void project_orbit_(double *restrict A, int64_t h, int64_t a, int64_t b, int64_t c, int64_t d,
                                  int64_t nvir, int64_t nvv, int64_t nvvv, double alpha, double beta) {
    if (a > b && b > c && c > d) {
        // All-distinct: 24-element orbit, indices follow the perms[] ordering.
        int64_t idx[24];
        idx[0] = a * nvvv + b * nvv + c * nvir + d;
        idx[1] = a * nvvv + b * nvv + d * nvir + c;
        idx[2] = a * nvvv + c * nvv + b * nvir + d;
        idx[3] = a * nvvv + c * nvv + d * nvir + b;
        idx[4] = a * nvvv + d * nvv + b * nvir + c;
        idx[5] = a * nvvv + d * nvv + c * nvir + b;
        idx[6] = b * nvvv + a * nvv + c * nvir + d;
        idx[7] = b * nvvv + a * nvv + d * nvir + c;
        idx[8] = b * nvvv + c * nvv + a * nvir + d;
        idx[9] = b * nvvv + c * nvv + d * nvir + a;
        idx[10] = b * nvvv + d * nvv + a * nvir + c;
        idx[11] = b * nvvv + d * nvv + c * nvir + a;
        idx[12] = c * nvvv + a * nvv + b * nvir + d;
        idx[13] = c * nvvv + a * nvv + d * nvir + b;
        idx[14] = c * nvvv + b * nvv + a * nvir + d;
        idx[15] = c * nvvv + b * nvv + d * nvir + a;
        idx[16] = c * nvvv + d * nvv + a * nvir + b;
        idx[17] = c * nvvv + d * nvv + b * nvir + a;
        idx[18] = d * nvvv + a * nvv + b * nvir + c;
        idx[19] = d * nvvv + a * nvv + c * nvir + b;
        idx[20] = d * nvvv + b * nvv + a * nvir + c;
        idx[21] = d * nvvv + b * nvv + c * nvir + a;
        idx[22] = d * nvvv + c * nvv + a * nvir + b;
        idx[23] = d * nvvv + c * nvv + b * nvir + a;

        double x[24], x1[24], x2[24], x3[24], x4[24], y[24];
        for (int p = 0; p < 24; p++)
            x[p] = A[h + idx[p]];

        // x1 = (Omega - 6) x
        apply_omega_24(x1, x);
        for (int p = 0; p < 24; p++)
            x1[p] -= 6.0 * x[p];

        // x2 = (Omega - 2) x1
        apply_omega_24(x2, x1);
        for (int p = 0; p < 24; p++)
            x2[p] -= 2.0 * x1[p];

        // x3 = Omega x2,  x4 = Omega x3
        apply_omega_24(x3, x2);
        apply_omega_24(x4, x3);

        for (int p = 0; p < 24; p++)
            y[p] = (2.0 * x4[p] + 19.0 * x3[p] + 48.0 * x2[p]) / 576.0;

        for (int p = 0; p < 24; p++)
            A[h + idx[p]] = beta * x[p] + alpha * y[p];
    } else {
        // Degenerate orbit (some indices equal): build orbit and perm_map once.
        int64_t tuples[24][4];
        int64_t indices[24];
        int perm_map[24][6];
        double x[24], x1[24], x2[24], x3[24], x4[24], y[24];
        static const int swaps[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

        int ntuples = build_unique_orbit(a, b, c, d, tuples);

        // Precompute perm_map once for this orbit (avoids repeated find_tuple4 calls).
        for (int p = 0; p < ntuples; p++)
            for (int s = 0; s < 6; s++) {
                int64_t target[4];
                swap4(tuples[p], swaps[s][0], swaps[s][1], target);
                perm_map[p][s] = find_tuple4(tuples, ntuples, target);
            }

        for (int p = 0; p < ntuples; p++) {
            indices[p] = idx4(tuples[p][0], tuples[p][1], tuples[p][2], tuples[p][3], nvir, nvv, nvvv);
            x[p] = A[h + indices[p]];
        }

        for (int p = 0; p < ntuples; p++) {
            double om = 0.0;
            for (int s = 0; s < 6; s++)
                om += x[perm_map[p][s]];
            x1[p] = om - 6.0 * x[p];
        }
        for (int p = 0; p < ntuples; p++) {
            double om = 0.0;
            for (int s = 0; s < 6; s++)
                om += x1[perm_map[p][s]];
            x2[p] = om - 2.0 * x1[p];
        }
        for (int p = 0; p < ntuples; p++) {
            double om = 0.0;
            for (int s = 0; s < 6; s++)
                om += x2[perm_map[p][s]];
            x3[p] = om;
        }
        for (int p = 0; p < ntuples; p++) {
            double om = 0.0;
            for (int s = 0; s < 6; s++)
                om += x3[perm_map[p][s]];
            x4[p] = om;
        }

        for (int p = 0; p < ntuples; p++)
            y[p] = (2.0 * x4[p] + 19.0 * x3[p] + 48.0 * x2[p]) / 576.0;

        for (int p = 0; p < ntuples; p++)
            A[h + indices[p]] = beta * x[p] + alpha * y[p];
    }
}

// Apply Q = 1 - P_[4] - P_[3,1] to T4 amplitudes in place.
//
// A: flattened tensor with shape
//
//      A[nocc4, nvir, nvir, nvir, nvir]
//
//    in C-order layout:
//
//      A[ijkl, a, b, c, d]
//
// nocc4: number of occupied-index blocks, e.g. compact ijkl or full ijkl
// nvir : number of virtual orbitals
//
// alpha, beta:
//
//      A <- beta * A + alpha * Q(A)
//
// where
//
//      Q = ((Omega - 6)(Omega - 2)(2 Omega^2 + 19 Omega + 48)) / 576
//
// and
//
//      Omega = P_ab + P_ac + P_ad + P_bc + P_bd + P_cd.
//
void t4_project_1_minus_p4_p31_inplace_(double *A, int64_t nocc4, int64_t nvir, double alpha, double beta) {
    init_omega_action_24();
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;
    const int64_t bl = 8;

#pragma omp parallel for schedule(static)
    for (int64_t ijkl = 0; ijkl < nocc4; ijkl++) {
        int64_t h = ijkl * nvvvv;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                        project_orbit_(A, h, a, b, c, d, nvir, nvv, nvvv, alpha, beta);
    }
}

// Apply spin summation projection to a single T4 amplitude block in place.
// A: pointer to T4 tensor (size nvir**4)
// pattern: "P4_full" : P(A) = (1 + P_c^d) (1 + P_b^c + P_b^d) (1 + P_a^b + P_a^c + P_a^d) A
//          "P4_422"  : P(A) = (1 + 0 * P_c^d) (1 + 0 * P_b^c + 0 * P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
//          "P4_201"  : P(A) = (1 + 0 * P_c^d) (2 - P_b^c - P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
// alpha, beta: A = beta * A + alpha * P(A)
void t4_single_spin_summation_inplace_(double *A, int64_t nvir, char *pattern, double alpha, double beta) {
    const int64_t bl = 8;
    int64_t h = 0;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;

    double p[9];

    if (strcmp(pattern, "P4_full") == 0) {
        for (int i = 0; i < 9; i++)
            p[i] = 1.0;
    } else if (strcmp(pattern, "P4_201") == 0) {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 1.0;
        p[5] = 0.0;
        p[6] = 0.0;
        p[7] = 1.0;
        p[8] = 0.0;
    } else if (strcmp(pattern, "P4_442") == 0) {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 2.0;
        p[5] = -1.0;
        p[6] = -1.0;
        p[7] = 1.0;
        p[8] = 0.0;
    } else {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

    const int64_t nblk = (nvir + bl - 1) / bl;
    const int64_t nblock4 = nblk * (nblk + 1) * (nblk + 2) * (nblk + 3) / 24;

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t block_idx = 0; block_idx < nblock4; block_idx++) {
        int64_t rem = block_idx;
        int64_t a_blk = 0;
        while (a_blk < nblk) {
            int64_t n_bcd = (a_blk + 1) * (a_blk + 2) * (a_blk + 3) / 6;
            if (rem < n_bcd)
                break;
            rem -= n_bcd;
            a_blk++;
        }

        int64_t b_blk = 0;
        while (b_blk <= a_blk) {
            int64_t n_cd = (b_blk + 1) * (b_blk + 2) / 2;
            if (rem < n_cd)
                break;
            rem -= n_cd;
            b_blk++;
        }

        int64_t c_blk = 0;
        while (c_blk <= b_blk) {
            int64_t n_d = c_blk + 1;
            if (rem < n_d)
                break;
            rem -= n_d;
            c_blk++;
        }

        int64_t d_blk = rem;

        int64_t a0 = a_blk * bl;
        int64_t b0 = b_blk * bl;
        int64_t c0 = c_blk * bl;
        int64_t d0 = d_blk * bl;

        for (int64_t a = a0; a < a0 + bl && a < nvir; a++) {
            for (int64_t b = b0; b < b0 + bl && b <= a; b++) {
                for (int64_t c = c0; c < c0 + bl && c <= b; c++) {
                    for (int64_t d = d0; d < d0 + bl && d <= c; d++) {
                        if (a > b && b > c && c > d) {
                            double T1_local[24];
                            double T2_local[24];

                            int64_t indices[24];
                            indices[0] = a * nvvv + b * nvv + c * nvir + d;
                            indices[1] = a * nvvv + b * nvv + d * nvir + c;
                            indices[2] = a * nvvv + c * nvv + b * nvir + d;
                            indices[3] = a * nvvv + c * nvv + d * nvir + b;
                            indices[4] = a * nvvv + d * nvv + b * nvir + c;
                            indices[5] = a * nvvv + d * nvv + c * nvir + b;
                            indices[6] = b * nvvv + a * nvv + c * nvir + d;
                            indices[7] = b * nvvv + a * nvv + d * nvir + c;
                            indices[8] = b * nvvv + c * nvv + a * nvir + d;
                            indices[9] = b * nvvv + c * nvv + d * nvir + a;
                            indices[10] = b * nvvv + d * nvv + a * nvir + c;
                            indices[11] = b * nvvv + d * nvv + c * nvir + a;
                            indices[12] = c * nvvv + a * nvv + b * nvir + d;
                            indices[13] = c * nvvv + a * nvv + d * nvir + b;
                            indices[14] = c * nvvv + b * nvv + a * nvir + d;
                            indices[15] = c * nvvv + b * nvv + d * nvir + a;
                            indices[16] = c * nvvv + d * nvv + a * nvir + b;
                            indices[17] = c * nvvv + d * nvv + b * nvir + a;
                            indices[18] = d * nvvv + a * nvv + b * nvir + c;
                            indices[19] = d * nvvv + a * nvv + c * nvir + b;
                            indices[20] = d * nvvv + b * nvv + a * nvir + c;
                            indices[21] = d * nvvv + b * nvv + c * nvir + a;
                            indices[22] = d * nvvv + c * nvv + a * nvir + b;
                            indices[23] = d * nvvv + c * nvv + b * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[6]] +
                                          p[2] * A[h + indices[14]] + p[3] * A[h + indices[21]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[7]] +
                                          p[2] * A[h + indices[20]] + p[3] * A[h + indices[15]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[12]] +
                                          p[2] * A[h + indices[8]] + p[3] * A[h + indices[23]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[13]] +
                                          p[2] * A[h + indices[22]] + p[3] * A[h + indices[9]];
                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[18]] +
                                          p[2] * A[h + indices[10]] + p[3] * A[h + indices[17]];
                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[19]] +
                                          p[2] * A[h + indices[16]] + p[3] * A[h + indices[11]];
                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[12]] + p[3] * A[h + indices[19]];
                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[18]] + p[3] * A[h + indices[13]];
                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[14]] +
                                          p[2] * A[h + indices[2]] + p[3] * A[h + indices[22]];
                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[15]] +
                                          p[2] * A[h + indices[23]] + p[3] * A[h + indices[3]];
                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[20]] +
                                           p[2] * A[h + indices[4]] + p[3] * A[h + indices[16]];
                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[21]] +
                                           p[2] * A[h + indices[17]] + p[3] * A[h + indices[5]];
                            T1_local[12] = p[0] * A[h + indices[12]] + p[1] * A[h + indices[2]] +
                                           p[2] * A[h + indices[6]] + p[3] * A[h + indices[18]];
                            T1_local[13] = p[0] * A[h + indices[13]] + p[1] * A[h + indices[3]] +
                                           p[2] * A[h + indices[19]] + p[3] * A[h + indices[7]];
                            T1_local[14] = p[0] * A[h + indices[14]] + p[1] * A[h + indices[8]] +
                                           p[2] * A[h + indices[0]] + p[3] * A[h + indices[20]];
                            T1_local[15] = p[0] * A[h + indices[15]] + p[1] * A[h + indices[9]] +
                                           p[2] * A[h + indices[21]] + p[3] * A[h + indices[1]];
                            T1_local[16] = p[0] * A[h + indices[16]] + p[1] * A[h + indices[22]] +
                                           p[2] * A[h + indices[5]] + p[3] * A[h + indices[10]];
                            T1_local[17] = p[0] * A[h + indices[17]] + p[1] * A[h + indices[23]] +
                                           p[2] * A[h + indices[11]] + p[3] * A[h + indices[4]];
                            T1_local[18] = p[0] * A[h + indices[18]] + p[1] * A[h + indices[4]] +
                                           p[2] * A[h + indices[7]] + p[3] * A[h + indices[12]];
                            T1_local[19] = p[0] * A[h + indices[19]] + p[1] * A[h + indices[5]] +
                                           p[2] * A[h + indices[13]] + p[3] * A[h + indices[6]];
                            T1_local[20] = p[0] * A[h + indices[20]] + p[1] * A[h + indices[10]] +
                                           p[2] * A[h + indices[1]] + p[3] * A[h + indices[14]];
                            T1_local[21] = p[0] * A[h + indices[21]] + p[1] * A[h + indices[11]] +
                                           p[2] * A[h + indices[15]] + p[3] * A[h + indices[0]];
                            T1_local[22] = p[0] * A[h + indices[22]] + p[1] * A[h + indices[16]] +
                                           p[2] * A[h + indices[3]] + p[3] * A[h + indices[8]];
                            T1_local[23] = p[0] * A[h + indices[23]] + p[1] * A[h + indices[17]] +
                                           p[2] * A[h + indices[9]] + p[3] * A[h + indices[2]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];
                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];
                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];
                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];
                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];
                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];
                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];
                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];
                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];
                            T2_local[12] = p[4] * T1_local[12] + p[5] * T1_local[14] + p[6] * T1_local[17];
                            T2_local[13] = p[4] * T1_local[13] + p[5] * T1_local[16] + p[6] * T1_local[15];
                            T2_local[14] = p[4] * T1_local[14] + p[5] * T1_local[12] + p[6] * T1_local[16];
                            T2_local[15] = p[4] * T1_local[15] + p[5] * T1_local[17] + p[6] * T1_local[13];
                            T2_local[16] = p[4] * T1_local[16] + p[5] * T1_local[13] + p[6] * T1_local[14];
                            T2_local[17] = p[4] * T1_local[17] + p[5] * T1_local[15] + p[6] * T1_local[12];
                            T2_local[18] = p[4] * T1_local[18] + p[5] * T1_local[20] + p[6] * T1_local[23];
                            T2_local[19] = p[4] * T1_local[19] + p[5] * T1_local[22] + p[6] * T1_local[21];
                            T2_local[20] = p[4] * T1_local[20] + p[5] * T1_local[18] + p[6] * T1_local[22];
                            T2_local[21] = p[4] * T1_local[21] + p[5] * T1_local[23] + p[6] * T1_local[19];
                            T2_local[22] = p[4] * T1_local[22] + p[5] * T1_local[19] + p[6] * T1_local[20];
                            T2_local[23] = p[4] * T1_local[23] + p[5] * T1_local[21] + p[6] * T1_local[18];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                            A[h + indices[4]] =
                                alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                            A[h + indices[5]] =
                                alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                            A[h + indices[6]] =
                                alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                            A[h + indices[7]] =
                                alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                            A[h + indices[8]] =
                                alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                            A[h + indices[9]] =
                                alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                            A[h + indices[10]] =
                                alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                            A[h + indices[11]] =
                                alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                            A[h + indices[12]] =
                                alpha * (p[7] * T2_local[12] + p[8] * T2_local[13]) + beta * A[h + indices[12]];
                            A[h + indices[13]] =
                                alpha * (p[7] * T2_local[13] + p[8] * T2_local[12]) + beta * A[h + indices[13]];
                            A[h + indices[14]] =
                                alpha * (p[7] * T2_local[14] + p[8] * T2_local[15]) + beta * A[h + indices[14]];
                            A[h + indices[15]] =
                                alpha * (p[7] * T2_local[15] + p[8] * T2_local[14]) + beta * A[h + indices[15]];
                            A[h + indices[16]] =
                                alpha * (p[7] * T2_local[16] + p[8] * T2_local[17]) + beta * A[h + indices[16]];
                            A[h + indices[17]] =
                                alpha * (p[7] * T2_local[17] + p[8] * T2_local[16]) + beta * A[h + indices[17]];
                            A[h + indices[18]] =
                                alpha * (p[7] * T2_local[18] + p[8] * T2_local[19]) + beta * A[h + indices[18]];
                            A[h + indices[19]] =
                                alpha * (p[7] * T2_local[19] + p[8] * T2_local[18]) + beta * A[h + indices[19]];
                            A[h + indices[20]] =
                                alpha * (p[7] * T2_local[20] + p[8] * T2_local[21]) + beta * A[h + indices[20]];
                            A[h + indices[21]] =
                                alpha * (p[7] * T2_local[21] + p[8] * T2_local[20]) + beta * A[h + indices[21]];
                            A[h + indices[22]] =
                                alpha * (p[7] * T2_local[22] + p[8] * T2_local[23]) + beta * A[h + indices[22]];
                            A[h + indices[23]] =
                                alpha * (p[7] * T2_local[23] + p[8] * T2_local[22]) + beta * A[h + indices[23]];
                        } else if (a > b && b > c && c == d) {
                            double T1_local[12];
                            double T2_local[12];

                            int64_t indices[12];
                            indices[0] = a * nvvv + b * nvv + c * nvir + c;
                            indices[1] = a * nvvv + c * nvv + b * nvir + c;
                            indices[2] = a * nvvv + c * nvv + c * nvir + b;
                            indices[3] = b * nvvv + a * nvv + c * nvir + c;
                            indices[4] = b * nvvv + c * nvv + a * nvir + c;
                            indices[5] = b * nvvv + c * nvv + c * nvir + a;
                            indices[6] = c * nvvv + a * nvv + b * nvir + c;
                            indices[7] = c * nvvv + a * nvv + c * nvir + b;
                            indices[8] = c * nvvv + b * nvv + a * nvir + c;
                            indices[9] = c * nvvv + b * nvv + c * nvir + a;
                            indices[10] = c * nvvv + c * nvv + a * nvir + b;
                            indices[11] = c * nvvv + c * nvv + b * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] +
                                          p[2] * A[h + indices[8]] + p[3] * A[h + indices[9]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[6]] +
                                          p[2] * A[h + indices[4]] + p[3] * A[h + indices[11]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[7]] +
                                          p[2] * A[h + indices[10]] + p[3] * A[h + indices[5]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[6]] + p[3] * A[h + indices[7]];
                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[8]] +
                                          p[2] * A[h + indices[1]] + p[3] * A[h + indices[10]];
                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[9]] +
                                          p[2] * A[h + indices[11]] + p[3] * A[h + indices[2]];
                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[3]] + p[3] * A[h + indices[6]];
                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[2]] +
                                          p[2] * A[h + indices[7]] + p[3] * A[h + indices[3]];
                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[4]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[8]];
                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[5]] +
                                          p[2] * A[h + indices[9]] + p[3] * A[h + indices[0]];
                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[10]] +
                                           p[2] * A[h + indices[2]] + p[3] * A[h + indices[4]];
                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[11]] +
                                           p[2] * A[h + indices[5]] + p[3] * A[h + indices[1]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[4] + p[6] * T1_local[5];
                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[3] + p[6] * T1_local[4];
                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[5] + p[6] * T1_local[3];
                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];
                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];
                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];
                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];
                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];
                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                            A[h + indices[4]] =
                                alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                            A[h + indices[5]] =
                                alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                            A[h + indices[6]] =
                                alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                            A[h + indices[7]] =
                                alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                            A[h + indices[8]] =
                                alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                            A[h + indices[9]] =
                                alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                            A[h + indices[10]] =
                                alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                            A[h + indices[11]] =
                                alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                        } else if (a > b && b == c && c > d) {
                            double T1_local[12];
                            double T2_local[12];

                            int64_t indices[12];
                            indices[0] = a * nvvv + b * nvv + b * nvir + d;
                            indices[1] = a * nvvv + b * nvv + d * nvir + b;
                            indices[2] = a * nvvv + d * nvv + b * nvir + b;
                            indices[3] = b * nvvv + a * nvv + b * nvir + d;
                            indices[4] = b * nvvv + a * nvv + d * nvir + b;
                            indices[5] = b * nvvv + b * nvv + a * nvir + d;
                            indices[6] = b * nvvv + b * nvv + d * nvir + a;
                            indices[7] = b * nvvv + d * nvv + a * nvir + b;
                            indices[8] = b * nvvv + d * nvv + b * nvir + a;
                            indices[9] = d * nvvv + a * nvv + b * nvir + b;
                            indices[10] = d * nvvv + b * nvv + a * nvir + b;
                            indices[11] = d * nvvv + b * nvv + b * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] +
                                          p[2] * A[h + indices[5]] + p[3] * A[h + indices[11]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[4]] +
                                          p[2] * A[h + indices[10]] + p[3] * A[h + indices[6]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[9]] +
                                          p[2] * A[h + indices[7]] + p[3] * A[h + indices[8]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[3]] + p[3] * A[h + indices[9]];
                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[9]] + p[3] * A[h + indices[4]];
                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[10]];
                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[6]] +
                                          p[2] * A[h + indices[11]] + p[3] * A[h + indices[1]];
                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[10]] +
                                          p[2] * A[h + indices[2]] + p[3] * A[h + indices[7]];
                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] +
                                          p[2] * A[h + indices[8]] + p[3] * A[h + indices[2]];
                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[2]] +
                                          p[2] * A[h + indices[4]] + p[3] * A[h + indices[3]];
                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[7]] +
                                           p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] +
                                           p[2] * A[h + indices[6]] + p[3] * A[h + indices[0]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[8];
                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[7] + p[6] * T1_local[6];
                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[7];
                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[4];
                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[4] + p[6] * T1_local[5];
                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[3];
                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[10] + p[6] * T1_local[11];
                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[9] + p[6] * T1_local[10];
                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[11] + p[6] * T1_local[9];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                            A[h + indices[4]] =
                                alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                            A[h + indices[5]] =
                                alpha * (p[7] * T2_local[5] + p[8] * T2_local[6]) + beta * A[h + indices[5]];
                            A[h + indices[6]] =
                                alpha * (p[7] * T2_local[6] + p[8] * T2_local[5]) + beta * A[h + indices[6]];
                            A[h + indices[7]] =
                                alpha * (p[7] * T2_local[7] + p[8] * T2_local[8]) + beta * A[h + indices[7]];
                            A[h + indices[8]] =
                                alpha * (p[7] * T2_local[8] + p[8] * T2_local[7]) + beta * A[h + indices[8]];
                            A[h + indices[9]] =
                                alpha * (p[7] * T2_local[9] + p[8] * T2_local[9]) + beta * A[h + indices[9]];
                            A[h + indices[10]] =
                                alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                            A[h + indices[11]] =
                                alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                        } else if (a == b && b > c && c > d) {
                            double T1_local[12];
                            double T2_local[12];

                            int64_t indices[12];
                            indices[0] = a * nvvv + a * nvv + c * nvir + d;
                            indices[1] = a * nvvv + a * nvv + d * nvir + c;
                            indices[2] = a * nvvv + c * nvv + a * nvir + d;
                            indices[3] = a * nvvv + c * nvv + d * nvir + a;
                            indices[4] = a * nvvv + d * nvv + a * nvir + c;
                            indices[5] = a * nvvv + d * nvv + c * nvir + a;
                            indices[6] = c * nvvv + a * nvv + a * nvir + d;
                            indices[7] = c * nvvv + a * nvv + d * nvir + a;
                            indices[8] = c * nvvv + d * nvv + a * nvir + a;
                            indices[9] = d * nvvv + a * nvv + a * nvir + c;
                            indices[10] = d * nvvv + a * nvv + c * nvir + a;
                            indices[11] = d * nvvv + c * nvv + a * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[6]] + p[3] * A[h + indices[10]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[9]] + p[3] * A[h + indices[7]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[6]] +
                                          p[2] * A[h + indices[2]] + p[3] * A[h + indices[11]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[7]] +
                                          p[2] * A[h + indices[11]] + p[3] * A[h + indices[3]];
                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[9]] +
                                          p[2] * A[h + indices[4]] + p[3] * A[h + indices[8]];
                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[10]] +
                                          p[2] * A[h + indices[8]] + p[3] * A[h + indices[5]];
                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[2]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[9]];
                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[3]] +
                                          p[2] * A[h + indices[10]] + p[3] * A[h + indices[1]];
                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] +
                                          p[2] * A[h + indices[5]] + p[3] * A[h + indices[4]];
                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[4]] +
                                          p[2] * A[h + indices[1]] + p[3] * A[h + indices[6]];
                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[5]] +
                                           p[2] * A[h + indices[7]] + p[3] * A[h + indices[0]];
                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] +
                                           p[2] * A[h + indices[3]] + p[3] * A[h + indices[2]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];
                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];
                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];
                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[6] + p[6] * T1_local[8];
                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[8] + p[6] * T1_local[7];
                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[7] + p[6] * T1_local[6];
                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[9] + p[6] * T1_local[11];
                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[11] + p[6] * T1_local[10];
                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[10] + p[6] * T1_local[9];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                            A[h + indices[4]] =
                                alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                            A[h + indices[5]] =
                                alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                            A[h + indices[6]] =
                                alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                            A[h + indices[7]] =
                                alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                            A[h + indices[8]] =
                                alpha * (p[7] * T2_local[8] + p[8] * T2_local[8]) + beta * A[h + indices[8]];
                            A[h + indices[9]] =
                                alpha * (p[7] * T2_local[9] + p[8] * T2_local[10]) + beta * A[h + indices[9]];
                            A[h + indices[10]] =
                                alpha * (p[7] * T2_local[10] + p[8] * T2_local[9]) + beta * A[h + indices[10]];
                            A[h + indices[11]] =
                                alpha * (p[7] * T2_local[11] + p[8] * T2_local[11]) + beta * A[h + indices[11]];
                        } else if (a > b && b == c && c == d) {
                            double T1_local[4];
                            double T2_local[4];

                            int64_t indices[4];
                            indices[0] = a * nvvv + b * nvv + b * nvir + b;
                            indices[1] = b * nvvv + a * nvv + b * nvir + b;
                            indices[2] = b * nvvv + b * nvv + a * nvir + b;
                            indices[3] = b * nvvv + b * nvv + b * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[2]] + p[3] * A[h + indices[3]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[1]] + p[3] * A[h + indices[1]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[2]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[2]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[3]] +
                                          p[2] * A[h + indices[3]] + p[3] * A[h + indices[0]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[3];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[2];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[1];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[1]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                        } else if (a == b && b == c && c > d) {
                            double T1_local[4];
                            double T2_local[4];

                            int64_t indices[4];
                            indices[0] = a * nvvv + a * nvv + a * nvir + d;
                            indices[1] = a * nvvv + a * nvv + d * nvir + a;
                            indices[2] = a * nvvv + d * nvv + a * nvir + a;
                            indices[3] = d * nvvv + a * nvv + a * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[3]] + p[3] * A[h + indices[1]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[3]] +
                                          p[2] * A[h + indices[2]] + p[3] * A[h + indices[2]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[2]] +
                                          p[2] * A[h + indices[1]] + p[3] * A[h + indices[0]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[3];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                        } else if (a == b && b > c && c == d) {
                            double T1_local[6];
                            double T2_local[6];

                            int64_t indices[6];
                            indices[0] = b * nvvv + b * nvv + c * nvir + c;
                            indices[1] = b * nvvv + c * nvv + b * nvir + c;
                            indices[2] = b * nvvv + c * nvv + c * nvir + b;
                            indices[3] = c * nvvv + b * nvv + b * nvir + c;
                            indices[4] = c * nvvv + b * nvv + c * nvir + b;
                            indices[5] = c * nvvv + c * nvv + b * nvir + b;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[3]] + p[3] * A[h + indices[4]];
                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[3]] +
                                          p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[4]] +
                                          p[2] * A[h + indices[5]] + p[3] * A[h + indices[2]];
                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[1]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[2]] +
                                          p[2] * A[h + indices[4]] + p[3] * A[h + indices[0]];
                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] +
                                          p[2] * A[h + indices[2]] + p[3] * A[h + indices[1]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[5];
                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[5] + p[6] * T1_local[4];
                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[4] + p[6] * T1_local[3];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                            A[h + indices[1]] =
                                alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                            A[h + indices[2]] =
                                alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                            A[h + indices[3]] =
                                alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                            A[h + indices[4]] =
                                alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                            A[h + indices[5]] =
                                alpha * (p[7] * T2_local[5] + p[8] * T2_local[5]) + beta * A[h + indices[5]];
                        } else if (a == b && b == c && c == d) {
                            double T1_local[1];
                            double T2_local[1];

                            int64_t indices[1];
                            indices[0] = a * nvvv + a * nvv + a * nvir + a;

                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] +
                                          p[2] * A[h + indices[0]] + p[3] * A[h + indices[0]];

                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];

                            A[h + indices[0]] =
                                alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                        }
                    }
                }
            }
        }
    }
}

static inline int64_t comb2(int64_t n) { return (n >= 2) ? n * (n - 1) / 2 : 0; }

static inline int64_t comb3(int64_t n) { return (n >= 3) ? n * (n - 1) * (n - 2) / 6 : 0; }

static inline int64_t comb4(int64_t n) { return (n >= 4) ? n * (n - 1) * (n - 2) * (n - 3) / 24 : 0; }

static inline int64_t ijkl_to_linear(int64_t i, int64_t j, int64_t k, int64_t l, int64_t nocc) {
    int64_t n_before_i = comb4(nocc + 3) - comb4(nocc - i + 3);
    int64_t n_before_j = comb3(nocc - i + 2) - comb3(nocc - j + 2);
    int64_t n_before_k = comb2(nocc - j + 1) - comb2(nocc - k + 1);
    return n_before_i + n_before_j + n_before_k + (l - k);
}

static void get_canonical_and_perm4(int64_t i, int64_t j, int64_t k, int64_t l, int64_t canonical[4], int64_t perm[4]) {
    int64_t values[4] = {i, j, k, l};
    int64_t positions[4] = {0, 1, 2, 3};

    for (int a = 1; a < 4; a++) {
        int64_t value = values[a];
        int64_t pos = positions[a];
        int b = a - 1;
        while (b >= 0 && (values[b] > value || (values[b] == value && positions[b] > pos))) {
            values[b + 1] = values[b];
            positions[b + 1] = positions[b];
            b--;
        }
        values[b + 1] = value;
        positions[b + 1] = pos;
    }

    for (int src_axis = 0; src_axis < 4; src_axis++) {
        canonical[src_axis] = values[src_axis];
    }

    for (int dst_axis = 0; dst_axis < 4; dst_axis++) {
        for (int src_axis = 0; src_axis < 4; src_axis++) {
            if (positions[src_axis] == dst_axis) {
                perm[dst_axis] = src_axis;
                break;
            }
        }
    }
}

static void apply_perm4_and_copy(const double *src, double *dst, int64_t nvir, const int64_t perm[4]) {
    int64_t nvir2 = nvir * nvir;
    int64_t nvir3 = nvir2 * nvir;

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t a = 0; a < nvir; a++) {
        for (int64_t b = 0; b < nvir; b++) {
            for (int64_t c = 0; c < nvir; c++) {
                for (int64_t d = 0; d < nvir; d++) {
                    int64_t abcd[4] = {a, b, c, d};
                    int64_t aa = abcd[perm[0]];
                    int64_t bb = abcd[perm[1]];
                    int64_t cc = abcd[perm[2]];
                    int64_t dd = abcd[perm[3]];
                    int64_t src_idx = a * nvir3 + b * nvir2 + c * nvir + d;
                    int64_t dst_idx = aa * nvir3 + bb * nvir2 + cc * nvir + dd;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// Construct the three-term virtual transpose sum used by unpack_t4_tri2block_triples_:
//
//   dst[a,b,c,d] = src[a,b,c,d] + src[c,a,b,d] + src[d,a,c,b]
//
// src and dst must be separate contiguous buffers of shape [nvir,nvir,nvir,nvir].
void t4_transpose_add_(const double *restrict src, double *restrict dst, int64_t nvir) {
#define VIDX(a, b, c, d) ((((a) * nvir + (b)) * nvir + (c)) * nvir + (d))

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t a = 0; a < nvir; a++) {
        for (int64_t b = 0; b < nvir; b++) {
            for (int64_t c = 0; c < nvir; c++) {
                for (int64_t d = 0; d < nvir; d++) {
                    dst[VIDX(a, b, c, d)] = src[VIDX(a, b, c, d)] + src[VIDX(c, a, b, d)] + src[VIDX(d, a, c, b)];
                }
            }
        }
    }

#undef VIDX
}

// Generate the four P4_201 spin-summed virtual-axis variants used by the
// W_ovvo contractions.  The output order mirrors t3_spin_summation_triple_sym_:
// B0 selects axis d, B1 selects axis c, B2 selects axis b, and B3 selects axis a.
//
// For T = src[a,b,c,d]:
//   B0 = 2*T - T[d,b,c,a] - T[a,d,c,b] - T[a,b,d,c]
//   B1 = 2*T - T[c,b,a,d] - T[a,c,b,d] - T[a,b,d,c]
//   B2 = 2*T - T[b,a,c,d] - T[a,c,b,d] - T[a,d,c,b]
//   B3 = 2*T - T[b,a,c,d] - T[c,b,a,d] - T[d,b,c,a]
//
// B3 is the direct P4_201 result from t4_single_spin_summation_inplace_.
void t4_spin_summation_quadruple_sym_(const double *restrict src, double *restrict B0, double *restrict B1,
                                      double *restrict B2, double *restrict B3, int64_t nvir) {
#define VIDX(a, b, c, d) ((((a) * nvir + (b)) * nvir + (c)) * nvir + (d))

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t a = 0; a < nvir; a++) {
        for (int64_t b = 0; b < nvir; b++) {
            for (int64_t c = 0; c < nvir; c++) {
                for (int64_t d = 0; d < nvir; d++) {
                    const int64_t idx = VIDX(a, b, c, d);
                    const double two_t = 2.0 * src[idx];

                    const double t_bacd = src[VIDX(b, a, c, d)];
                    const double t_cbad = src[VIDX(c, b, a, d)];
                    const double t_dbca = src[VIDX(d, b, c, a)];
                    const double t_acbd = src[VIDX(a, c, b, d)];
                    const double t_adcb = src[VIDX(a, d, c, b)];
                    const double t_abdc = src[VIDX(a, b, d, c)];

                    B0[idx] = two_t - t_dbca - t_adcb - t_abdc;
                    B1[idx] = two_t - t_cbad - t_acbd - t_abdc;
                    B2[idx] = two_t - t_bacd - t_acbd - t_adcb;
                    B3[idx] = two_t - t_bacd - t_cbad - t_dbca;
                }
            }
        }
    }

#undef VIDX
}

// Divide distributed local R4 blocks by their true ijklabcd denominator.
//
// r4_local   : shape (nlocal, nvir, nvir, nvir, nvir), C-contiguous float64.
// eia        : shape (nocc, nvir), eia[i*nvir+a] = e_i - e_a - level_shift.
// local_ijkl : shape (nlocal, 4), int32 canonical occupied quadruples owned by
//              the current rank.
void r4_local_tri_divide_e_(double *restrict r4_local, const double *restrict eia, const int32_t *restrict local_ijkl,
                            int64_t nlocal, int64_t nvir) {
    const int64_t nvir2 = nvir * nvir;
    const int64_t nvir3 = nvir2 * nvir;
    const int64_t nvir4 = nvir3 * nvir;

#pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < nlocal; q++) {
        const int64_t i = local_ijkl[q * 4 + 0];
        const int64_t j = local_ijkl[q * 4 + 1];
        const int64_t k = local_ijkl[q * 4 + 2];
        const int64_t l = local_ijkl[q * 4 + 3];

        const double *eia_i = eia + i * nvir;
        const double *eia_j = eia + j * nvir;
        const double *eia_k = eia + k * nvir;
        const double *eia_l = eia + l * nvir;
        double *blk = r4_local + q * nvir4;

        for (int64_t a = 0; a < nvir; a++) {
            const double eia_ia = eia_i[a];
            for (int64_t b = 0; b < nvir; b++) {
                const double eijab = eia_ia + eia_j[b];
                for (int64_t c = 0; c < nvir; c++) {
                    const double eijkabc = eijab + eia_k[c];
                    double *ptr = blk + a * nvir3 + b * nvir2 + c * nvir;
                    for (int64_t d = 0; d < nvir; d++) {
                        ptr[d] /= eijkabc + eia_l[d];
                    }
                }
            }
        }
    }
}

void fill_local_data_ijkl_(const double *t4_local, double *send_data, const int32_t *requests,
                           const int64_t *ijkl_to_local_idx, int64_t n_requests, int64_t nvir) {
    int64_t nvir4 = nvir * nvir * nvir * nvir;
    size_t block_bytes = (size_t)nvir4 * sizeof(double);

#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n_requests; r++) {
        int64_t local_idx = ijkl_to_local_idx[requests[r]];
        if (local_idx >= 0) {
            memcpy(send_data + r * nvir4, t4_local + local_idx * nvir4, block_bytes);
        }
    }
}

void unpack_t4_ijkl_single_(const double *t4_local, double *t4_blk, const int64_t *ijkl_to_local_idx, int64_t i,
                            int64_t j, int64_t k, int64_t l, int64_t nocc, int64_t nvir) {
    int64_t canonical[4];
    int64_t perm[4];
    int64_t nvir4 = nvir * nvir * nvir * nvir;

    get_canonical_and_perm4(i, j, k, l, canonical, perm);

    if ((canonical[0] == canonical[1] && canonical[1] == canonical[2]) ||
        (canonical[1] == canonical[2] && canonical[2] == canonical[3])) {
        memset(t4_blk, 0, (size_t)nvir4 * sizeof(double));
        return;
    }

    int64_t ijkl_idx = ijkl_to_linear(canonical[0], canonical[1], canonical[2], canonical[3], nocc);
    int64_t local_idx = ijkl_to_local_idx[ijkl_idx];
    if (local_idx < 0) {
        return;
    }

    apply_perm4_and_copy(t4_local + local_idx * nvir4, t4_blk, nvir, perm);
}
