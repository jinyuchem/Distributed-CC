/*
Copyright 2025-2026 The Distributed-CC Developers. All Rights Reserved.

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
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif
#endif

// Apply spin summation projection to T4 amplitudes in place.
// A: pointer to T4 tensor (size nocc4 * nvir**4)
// pattern: "P4_full" : P(A) = (1 + P_c^d) (1 + P_b^c + P_b^d) (1 + P_a^b + P_a^c + P_a^d) A
//          "P4_444"  : P(A) = (2 - P_c^d) (2 - P_b^c - P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
//          "P4_442"  : P(A) = (1 + 0 * P_c^d) (2 - P_b^c - P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
//          "P4_201"  : P(A) = (1 + 0 * P_c^d) (1 + 0 * P_b^c + 0 * P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
// alpha, beta: A = beta * A + alpha * P(A)
void t4_spin_summation_inplace_(double *A, int64_t nocc4, int64_t nvir, char *pattern, double alpha, double beta) {
    int64_t ijkl;
    const int64_t bl = 8;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;

    double p[9];

    if (strcmp(pattern, "P4_full") == 0) {
        for (int i = 0; i < 9; i++)
            p[i] = 1.0;
    } else if (strcmp(pattern, "P4_444") == 0) {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 2.0;
        p[5] = -1.0;
        p[6] = -1.0;
        p[7] = 2.0;
        p[8] = -1.0;
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

#pragma omp parallel for schedule(static)
    for (ijkl = 0; ijkl < nocc4; ijkl++) {
        int64_t h = ijkl * nvvvv;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl) {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl) {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl) {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl) {
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

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[14]] + p[3] * A[h + indices[21]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[20]] + p[3] * A[h + indices[15]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[12]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[23]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[13]] + p[2] * A[h + indices[22]] + p[3] * A[h + indices[9]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[18]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[17]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[19]] + p[2] * A[h + indices[16]] + p[3] * A[h + indices[11]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[12]] + p[3] * A[h + indices[19]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[18]] + p[3] * A[h + indices[13]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[14]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[22]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[15]] + p[2] * A[h + indices[23]] + p[3] * A[h + indices[3]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[20]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[16]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[21]] + p[2] * A[h + indices[17]] + p[3] * A[h + indices[5]];
                                            T1_local[12] = p[0] * A[h + indices[12]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[18]];
                                            T1_local[13] = p[0] * A[h + indices[13]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[19]] + p[3] * A[h + indices[7]];
                                            T1_local[14] = p[0] * A[h + indices[14]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[20]];
                                            T1_local[15] = p[0] * A[h + indices[15]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[21]] + p[3] * A[h + indices[1]];
                                            T1_local[16] = p[0] * A[h + indices[16]] + p[1] * A[h + indices[22]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[10]];
                                            T1_local[17] = p[0] * A[h + indices[17]] + p[1] * A[h + indices[23]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[4]];
                                            T1_local[18] = p[0] * A[h + indices[18]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[12]];
                                            T1_local[19] = p[0] * A[h + indices[19]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[13]] + p[3] * A[h + indices[6]];
                                            T1_local[20] = p[0] * A[h + indices[20]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[14]];
                                            T1_local[21] = p[0] * A[h + indices[21]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[15]] + p[3] * A[h + indices[0]];
                                            T1_local[22] = p[0] * A[h + indices[22]] + p[1] * A[h + indices[16]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[8]];
                                            T1_local[23] = p[0] * A[h + indices[23]] + p[1] * A[h + indices[17]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[2]];

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

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                            A[h + indices[12]] = alpha * (p[7] * T2_local[12] + p[8] * T2_local[13]) + beta * A[h + indices[12]];
                                            A[h + indices[13]] = alpha * (p[7] * T2_local[13] + p[8] * T2_local[12]) + beta * A[h + indices[13]];
                                            A[h + indices[14]] = alpha * (p[7] * T2_local[14] + p[8] * T2_local[15]) + beta * A[h + indices[14]];
                                            A[h + indices[15]] = alpha * (p[7] * T2_local[15] + p[8] * T2_local[14]) + beta * A[h + indices[15]];
                                            A[h + indices[16]] = alpha * (p[7] * T2_local[16] + p[8] * T2_local[17]) + beta * A[h + indices[16]];
                                            A[h + indices[17]] = alpha * (p[7] * T2_local[17] + p[8] * T2_local[16]) + beta * A[h + indices[17]];
                                            A[h + indices[18]] = alpha * (p[7] * T2_local[18] + p[8] * T2_local[19]) + beta * A[h + indices[18]];
                                            A[h + indices[19]] = alpha * (p[7] * T2_local[19] + p[8] * T2_local[18]) + beta * A[h + indices[19]];
                                            A[h + indices[20]] = alpha * (p[7] * T2_local[20] + p[8] * T2_local[21]) + beta * A[h + indices[20]];
                                            A[h + indices[21]] = alpha * (p[7] * T2_local[21] + p[8] * T2_local[20]) + beta * A[h + indices[21]];
                                            A[h + indices[22]] = alpha * (p[7] * T2_local[22] + p[8] * T2_local[23]) + beta * A[h + indices[22]];
                                            A[h + indices[23]] = alpha * (p[7] * T2_local[23] + p[8] * T2_local[22]) + beta * A[h + indices[23]];
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

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[9]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[11]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[5]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[7]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[10]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[2]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[6]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[3]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[8]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[0]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[4]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[1]];

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

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
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

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[11]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[6]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[8]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[9]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[4]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[10]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[1]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[7]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[2]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[3]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[0]];

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

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[6]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[5]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[8]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[7]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[9]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
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

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[10]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[7]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[11]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[3]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[8]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[5]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[9]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[1]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[4]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[6]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[0]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[2]];

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

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[8]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[10]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[9]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[11]) + beta * A[h + indices[11]];
                                        } else if (a > b && b == c && c == d) {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + b;
                                            indices[1] = b * nvvv + a * nvv + b * nvir + b;
                                            indices[2] = b * nvvv + b * nvv + a * nvir + b;
                                            indices[3] = b * nvvv + b * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[3]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[1]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[1];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[1]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                        } else if (a == b && b == c && c > d) {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + d;
                                            indices[1] = a * nvvv + a * nvv + d * nvir + a;
                                            indices[2] = a * nvvv + d * nvv + a * nvir + a;
                                            indices[3] = d * nvvv + a * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[1]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[3];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
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

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[4]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[0]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[1]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[5];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[5] + p[6] * T1_local[4];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[4] + p[6] * T1_local[3];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[5]) + beta * A[h + indices[5]];
                                        } else if (a == b && b == c && c == d) {
                                            double T1_local[1];
                                            double T2_local[1];

                                            int64_t indices[1];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void e_abcdijkl_division_(double *restrict r4, const double *restrict e_occ, const double *restrict e_vir, const int64_t a0, const int64_t a1, const int64_t b0, const int64_t b1, const int64_t c0, const int64_t c1, const int64_t d0, const int64_t d1, const int64_t blk_a, const int64_t blk_b,
                          const int64_t blk_c, const int64_t blk_d, const int64_t nocc) {
    const int64_t nocc2 = nocc * nocc;
    const int64_t nocc3 = nocc2 * nocc;
    const int64_t nocc4 = nocc3 * nocc;

    double *e_kl = (double *)aligned_alloc(64, (nocc2 * sizeof(double) + 63) & ~63);
    for (int64_t k = 0; k < nocc; k++) {
        double ek = e_occ[k];
        for (int64_t l = 0; l < nocc; l++) {
            e_kl[k * nocc + l] = ek + e_occ[l];
        }
    }

    double *e_ij = (double *)aligned_alloc(64, (nocc2 * sizeof(double) + 63) & ~63);
    for (int64_t i = 0; i < nocc; i++) {
        double ei = e_occ[i];
        for (int64_t j = 0; j < nocc; j++) {
            e_ij[i * nocc + j] = ei + e_occ[j];
        }
    }

    double *e_ijkl = (double *)aligned_alloc(64, (nocc4 * sizeof(double) + 63) & ~63);
    for (int64_t ij = 0; ij < nocc2; ij++) {
        double eij = e_ij[ij];
        for (int64_t kl = 0; kl < nocc2; kl++) {
            e_ijkl[ij * nocc2 + kl] = eij + e_kl[kl];
        }
    }

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t a = a0; a < a1; a++) {
        for (int64_t b = b0; b < b1; b++) {
            for (int64_t c = c0; c < c1; c++) {
                for (int64_t d = d0; d < d1; d++) {
                    int64_t abcd_idx = ((((a - a0) * blk_b + (b - b0)) * blk_c + (c - c0)) * blk_d + (d - d0)) * nocc4;
                    // FIXED: denom = e_occ_sum - e_vir_sum (sign was flipped)
                    double neg_eabcd = -(e_vir[a] + e_vir[b] + e_vir[c] + e_vir[d]);

                    double *r4_block = r4 + abcd_idx;

#ifdef __AVX512F__
                    __m512d v_neg_eabcd = _mm512_set1_pd(neg_eabcd);
                    __m512d v_thresh = _mm512_set1_pd(1e-15);
                    __m512d v_zero = _mm512_setzero_pd();

                    int64_t ijkl = 0;
                    for (; ijkl + 8 <= nocc4; ijkl += 8) {
                        __m512d v_e_ijkl = _mm512_loadu_pd(&e_ijkl[ijkl]);
                        // denom = e_ijkl + neg_eabcd = e_ijkl - eabcd
                        __m512d v_denom = _mm512_add_pd(v_e_ijkl, v_neg_eabcd);

                        __m512d v_abs_denom = _mm512_abs_pd(v_denom);
                        __mmask8 mask = _mm512_cmp_pd_mask(v_abs_denom, v_thresh, _CMP_GT_OQ);

                        __m512d v_r4 = _mm512_loadu_pd(&r4_block[ijkl]);
                        __m512d v_result = _mm512_mask_div_pd(v_zero, mask, v_r4, v_denom);
                        _mm512_storeu_pd(&r4_block[ijkl], v_result);
                    }
                    for (; ijkl < nocc4; ijkl++) {
                        double denom = e_ijkl[ijkl] + neg_eabcd;
                        if (fabs(denom) > 1e-15) {
                            r4_block[ijkl] /= denom;
                        } else {
                            r4_block[ijkl] = 0.0;
                        }
                    }
#elif defined(__AVX__)
                    __m256d v_neg_eabcd = _mm256_set1_pd(neg_eabcd);
                    __m256d v_thresh = _mm256_set1_pd(1e-15);
                    __m256d v_neg_thresh = _mm256_set1_pd(-1e-15);
                    __m256d v_zero = _mm256_setzero_pd();

                    int64_t ijkl = 0;
                    for (; ijkl + 4 <= nocc4; ijkl += 4) {
                        __m256d v_e_ijkl = _mm256_loadu_pd(&e_ijkl[ijkl]);
                        // denom = e_ijkl + neg_eabcd = e_ijkl - eabcd
                        __m256d v_denom = _mm256_add_pd(v_e_ijkl, v_neg_eabcd);

                        __m256d cmp_pos = _mm256_cmp_pd(v_denom, v_thresh, _CMP_GT_OQ);
                        __m256d cmp_neg = _mm256_cmp_pd(v_denom, v_neg_thresh, _CMP_LT_OQ);
                        __m256d mask = _mm256_or_pd(cmp_pos, cmp_neg);

                        __m256d v_r4 = _mm256_loadu_pd(&r4_block[ijkl]);
                        __m256d v_divided = _mm256_div_pd(v_r4, v_denom);
                        __m256d v_result = _mm256_blendv_pd(v_zero, v_divided, mask);
                        _mm256_storeu_pd(&r4_block[ijkl], v_result);
                    }
                    for (; ijkl < nocc4; ijkl++) {
                        double denom = e_ijkl[ijkl] + neg_eabcd;
                        if (fabs(denom) > 1e-15) {
                            r4_block[ijkl] /= denom;
                        } else {
                            r4_block[ijkl] = 0.0;
                        }
                    }
#else
                    for (int64_t ijkl = 0; ijkl < nocc4; ijkl++) {
                        double denom = e_ijkl[ijkl] + neg_eabcd;
                        if (fabs(denom) > 1e-15) {
                            r4_block[ijkl] /= denom;
                        } else {
                            r4_block[ijkl] = 0.0;
                        }
                    }
#endif
                }
            }
        }
    }

    free(e_kl);
    free(e_ij);
    free(e_ijkl);
}

void t4_multiply_factor_(double *restrict t4_blk, const double *restrict factor_blk, const int64_t blk_a, const int64_t blk_b, const int64_t blk_c, const int64_t blk_d, const int64_t nocc) {
    const int64_t nocc2 = nocc * nocc;
    const int64_t nocc3 = nocc2 * nocc;
    const int64_t nocc4 = nocc3 * nocc;

    const int64_t stride_a = blk_b * blk_c * blk_d * nocc4;
    const int64_t stride_b = blk_c * blk_d * nocc4;
    const int64_t stride_c = blk_d * nocc4;
    const int64_t stride_d = nocc4;

#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t a = 0; a < blk_a; a++) {
        for (int64_t b = 0; b < blk_b; b++) {
            for (int64_t c = 0; c < blk_c; c++) {
                for (int64_t d = 0; d < blk_d; d++) {

                    double factor = factor_blk[((a * blk_b + b) * blk_c + c) * blk_d + d];

                    double *t4_ptr = t4_blk + a * stride_a + b * stride_b + c * stride_c + d * stride_d;

                    if (factor == 0.0) {
                        memset(t4_ptr, 0, nocc4 * sizeof(double));
                        continue;
                    }

#ifdef __AVX512F__
                    __m512d v_factor = _mm512_set1_pd(factor);

                    int64_t ijkl = 0;
                    for (; ijkl + 8 <= nocc4; ijkl += 8) {
                        __m512d v_t4 = _mm512_loadu_pd(&t4_ptr[ijkl]);
                        __m512d v_result = _mm512_mul_pd(v_t4, v_factor);
                        _mm512_storeu_pd(&t4_ptr[ijkl], v_result);
                    }
                    // Scalar cleanup
                    for (; ijkl < nocc4; ijkl++) {
                        t4_ptr[ijkl] *= factor;
                    }
#elif defined(__AVX__)
                    __m256d v_factor = _mm256_set1_pd(factor);

                    int64_t ijkl = 0;
                    for (; ijkl + 4 <= nocc4; ijkl += 4) {
                        __m256d v_t4 = _mm256_loadu_pd(&t4_ptr[ijkl]);
                        __m256d v_result = _mm256_mul_pd(v_t4, v_factor);
                        _mm256_storeu_pd(&t4_ptr[ijkl], v_result);
                    }
                    // Scalar cleanup
                    for (; ijkl < nocc4; ijkl++) {
                        t4_ptr[ijkl] *= factor;
                    }
#else
                    for (int64_t ijkl = 0; ijkl < nocc4; ijkl++) {
                        t4_ptr[ijkl] *= factor;
                    }
#endif
                }
            }
        }
    }
}

/* Permutation table for T3 amplitudes (a, b, c) */
static const int64_t tp_t3[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

static inline int64_t get_abc_triplet_index(int64_t a, int64_t b, int64_t c, int64_t nvir) {
    int64_t te_n = nvir * (nvir + 1) * (nvir + 2) / 6;
    int64_t rem = nvir - a;
    int64_t te_rem = rem * (rem + 1) * (rem + 2) / 6;
    int64_t idx = te_n - te_rem;
    idx += (b - a) * (2 * nvir - a - b + 1) / 2;
    idx += (c - b);
    return idx;
}

void fill_t3_from_ptr_array(double *t3_blk_target, int64_t a0, int64_t a1, int64_t b0, int64_t b1, int64_t nvir, int64_t nocc, uintptr_t *ptr_table, int64_t ld_b, int64_t max_size, int64_t ptr_table_size) {
    int64_t ba = a1 - a0;
    int64_t bb = b1 - b0;
    int64_t nocc3 = nocc * nocc * nocc;

    if (ptr_table_size > 132) {
        if (ptr_table[132] != 0) {
        } else {
        }
    }

    int64_t miss_count = 0;
    int64_t total_count = 0;

#pragma omp parallel for collapse(2) reduction(+ : total_count, miss_count)
    for (int64_t ia = 0; ia < ba; ia++) {
        for (int64_t ib = 0; ib < bb; ib++) {
            int64_t a = a0 + ia;
            int64_t b = b0 + ib;

            int64_t min_ab = (a < b) ? a : b;
            int64_t max_ab = (a > b) ? a : b;
            int a_idx = (a < b) ? 0 : 1;
            int b_idx = (a < b) ? 1 : 0;

            int64_t limits[4] = {0, min_ab + 1, max_ab + 1, nvir};

            for (int region = 0; region < 3; region++) {
                int64_t start = limits[region];
                int64_t end = limits[region + 1];
                if (start < 0)
                    start = 0;
                if (end > nvir)
                    end = nvir;

                if (start >= end)
                    continue;

                int64_t ci_dummy, cj_dummy, ck_dummy;
                int perm_idx;

                int case_num;
                if (region == 0) {
                    if (a <= b) {
                        case_num = 3;
                    } else {
                        case_num = 5;
                    }
                } else if (region == 1) {
                    if (a <= b) {
                    } else {
                    }
                } else {
                    if (a <= b) {
                        case_num = 0;
                    } else {
                        case_num = 2;
                    }
                }

                for (int64_t c = start; c < end; c++) {
                    total_count++;

                    if (region == 1) {
                        if (a <= b) {
                            case_num = (b <= c) ? 0 : 1;
                        } else {
                            case_num = (a <= c) ? 2 : 4;
                        }
                    }

                    if (region == 0) {
                        if (a <= b)
                            case_num = (a <= c) ? 1 : 3; // Handle boundary c=a
                        else
                            case_num = (b <= c) ? 4 : 5; // Handle boundary c=b
                    }

                    int64_t ci, cj, ck;
                    if (case_num == 0) {
                        ci = a;
                        cj = b;
                        ck = c;
                    } else if (case_num == 1) {
                        ci = a;
                        cj = c;
                        ck = b;
                    } else if (case_num == 2) {
                        ci = b;
                        cj = a;
                        ck = c;
                    } else if (case_num == 3) {
                        ci = c;
                        cj = a;
                        ck = b;
                    } else if (case_num == 4) {
                        ci = b;
                        cj = c;
                        ck = a;
                    } else {
                        ci = c;
                        cj = b;
                        ck = a;
                    }

                    int64_t abc_idx = get_abc_triplet_index(ci, cj, ck, nvir);

                    if (abc_idx >= ptr_table_size || abc_idx < 0) {
                        printf("ERROR: ptr_table index out of bounds! idx=%lld max=%lld\n", (long long)abc_idx, (long long)ptr_table_size);
                        printf("Triplets: a=%lld b=%lld c=%lld -> ci=%lld cj=%lld ck=%lld\n", (long long)a, (long long)b, (long long)c, (long long)ci, (long long)cj, (long long)ck);
                        abort();
                    }

                    uintptr_t ptr_val = ptr_table[abc_idx];

                    if (ptr_val != 0) {
                        double *src_ptr = (double *)ptr_val;
                        int64_t offset = (ia * ld_b * nvir + ib * nvir + c) * nocc3;

                        if (offset + nocc3 > max_size) {
                            printf("ERROR: Write out of bounds! offset=%lld size=%lld "
                                   "max=%lld\n",
                                   (long long)offset, (long long)nocc3, (long long)max_size);
                            printf("Current indices: ia=%lld ib=%lld c=%lld\n", (long long)ia, (long long)ib, (long long)c);
                            printf("Params: ld_b=%lld nvir=%lld nocc=%lld\n", (long long)ld_b, (long long)nvir, (long long)nocc);
                            abort();
                        }

                        double *dst_ptr = t3_blk_target + offset;

                        const int64_t *perm = tp_t3[case_num];
                        if (case_num == 0) {
                            memcpy(dst_ptr, src_ptr, nocc3 * sizeof(double));
                        } else {
                            int64_t p0 = perm[0], p1 = perm[1], p2 = perm[2];
                            int64_t nocc2 = nocc * nocc;
                            for (int64_t i = 0; i < nocc; i++) {
                                for (int64_t j = 0; j < nocc; j++) {
                                    for (int k = 0; k < nocc; k++) {
                                        int64_t ijk[3] = {i, j, k};
                                        dst_ptr[ijk[p0] * nocc2 + ijk[p1] * nocc + ijk[p2]] = src_ptr[i * nocc2 + j * nocc + k];
                                    }
                                }
                            }
                        }
                    } else {
                        miss_count++;
                        if (miss_count <= 5) {
                            printf("MISSING abc_idx=%lld (rank=?)\n", (long long)abc_idx);
                        }
                    }
                }
            }
        }
    }

    if (miss_count > 0) {
        if (miss_count == total_count) {
            printf("WARN: fill_t3 ALL BLOCKS MISSING %lld / %lld\n", (long long)miss_count, (long long)total_count);
        } else {
            printf("WARN: fill_t3 miss_count=%lld / %lld (%.1f%%)\n", (long long)miss_count, (long long)total_count, 100.0 * miss_count / total_count);
        }
    }
}

void pack_t3_indices_(double *out, double *src, int64_t *indices, int64_t n_blocks, int64_t block_size) {
#pragma omp parallel for
    for (int64_t i = 0; i < n_blocks; i++) {
        int64_t idx = indices[i];
        double *dest_ptr = out + i * block_size;
        double *src_ptr = src + idx * block_size;
        memcpy(dest_ptr, src_ptr, block_size * sizeof(double));
    }
}

void promote_t3_blocks_(double *out, uintptr_t *src_ptrs, int64_t n_blocks, int64_t block_size) {
#pragma omp parallel for
    for (int64_t i = 0; i < n_blocks; i++) {
        double *dest_ptr = out + i * block_size;
        double *src_ptr = (double *)src_ptrs[i];
        memcpy(dest_ptr, src_ptr, block_size * sizeof(double));
    }
}
