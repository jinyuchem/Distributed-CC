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
import ctypes
import numpy as np
from distr_cc._lib import _as_array, _void_p, _lib


_lib.t4_spin_summation_inplace_.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p,
                                            ctypes.c_double, ctypes.c_double]
_lib.t4_spin_summation_inplace_.restype = None
def t4_spin_summation_inplace_(A, nvir4, nocc, pattern, alpha=1.0, beta=0.0):
    A = _as_array(A, np.float64, "A")
    pattern_c = pattern.encode("utf-8")
    _lib.t4_spin_summation_inplace_(_void_p(A), ctypes.c_int64(nvir4), ctypes.c_int64(nocc),
                                    ctypes.c_char_p(pattern_c), ctypes.c_double(alpha), ctypes.c_double(beta))
    return A


_lib.fill_t3_from_ptr_array.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
_lib.fill_t3_from_ptr_array.restype = None
def fill_t3_from_ptr_array(t3_blk_target, a0, a1, b0, b1, nvir, nocc, ptr_table, ld_b, max_size, ptr_table_size):
    t3_blk_target = _as_array(t3_blk_target, np.float64, "t3_blk_target")
    ptr_table = _as_array(ptr_table, np.uintp, "ptr_table")
    _lib.fill_t3_from_ptr_array(_void_p(t3_blk_target), ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1), ctypes.c_int64(nvir), ctypes.c_int64(nocc),
        _void_p(ptr_table), ctypes.c_int64(ld_b), ctypes.c_int64(max_size), ctypes.c_int64(ptr_table_size))
    return t3_blk_target


_lib.pack_t3_indices_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
_lib.pack_t3_indices_.restype = None
def pack_t3_indices_(out, src, indices, n_blocks, block_size):
    out = _as_array(out, np.float64, "out")
    src = _as_array(src, np.float64, "src")
    indices = _as_array(indices, np.int64, "indices")
    _lib.pack_t3_indices_(_void_p(out), _void_p(src), _void_p(indices), ctypes.c_int64(n_blocks), ctypes.c_int64(block_size))
    return out


_lib.promote_t3_blocks_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
_lib.promote_t3_blocks_.restype = None
def promote_t3_blocks_(out, src_ptrs, n_blocks, block_size):
    out = _as_array(out, np.float64, "out")
    src_ptrs = _as_array(src_ptrs, np.uintp, "src_ptrs")
    _lib.promote_t3_blocks_(_void_p(out), _void_p(src_ptrs), ctypes.c_int64(n_blocks), ctypes.c_int64(block_size))
    return out


_lib.e_abcdijkl_division_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
_lib.e_abcdijkl_division_.restype = None
def e_abcdijkl_division_(r4, e_occ, e_vir, a0, a1, b0, b1, c0, c1, d0, d1, blk_a, blk_b, blk_c, blk_d, nocc):
    r4 = _as_array(r4, np.float64, "r4")
    e_occ = _as_array(e_occ, np.float64, "e_occ")
    e_vir = _as_array(e_vir, np.float64, "e_vir")
    _lib.e_abcdijkl_division_(_void_p(r4), _void_p(e_occ), _void_p(e_vir),
        ctypes.c_int64(a0), ctypes.c_int64(a1), ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(c0), ctypes.c_int64(c1), ctypes.c_int64(d0), ctypes.c_int64(d1),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b), ctypes.c_int64(blk_c), ctypes.c_int64(blk_d), ctypes.c_int64(nocc))
    return r4


_lib.t4_multiply_factor_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
_lib.t4_multiply_factor_.restype = None
def t4_multiply_factor_(t4_blk, factor_blk, blk_a, blk_b, blk_c, blk_d, nocc):
    t4_blk = _as_array(t4_blk, np.float64, "t4_blk")
    factor_blk = _as_array(factor_blk, np.float64, "factor_blk")
    _lib.t4_multiply_factor_(_void_p(t4_blk), _void_p(factor_blk), ctypes.c_int64(blk_a), ctypes.c_int64(blk_b),
                            ctypes.c_int64(blk_c), ctypes.c_int64(blk_d), ctypes.c_int64(nocc))
    return t4_blk
