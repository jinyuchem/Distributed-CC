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
from distr_cc._lib import _as_array, _double_p, _int32_p, _int64_p, _void_p, _lib

# NOTE: To be removed after the new release of pyscf
_lib.t4_project_1_minus_p4_p31_inplace_.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_double]
_lib.t4_project_1_minus_p4_p31_inplace_.restype = None
def t4_project_1_minus_p4_p31_inplace_(A, nocc4, nvir, alpha=1.0, beta=0.0):
    A = _as_array(A, np.float64, "A")
    _lib.t4_project_1_minus_p4_p31_inplace_(_void_p(A), ctypes.c_int64(nocc4), ctypes.c_int64(nvir), ctypes.c_double(alpha), ctypes.c_double(beta))
    return A


_lib.t4_single_spin_summation_inplace_.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
_lib.t4_single_spin_summation_inplace_.restype = None
def t4_single_spin_summation_inplace_(A, nvir, pattern, alpha=1.0, beta=0.0):
    A = _as_array(A, np.float64, "A")
    pattern_c = pattern.encode("utf-8")
    _lib.t4_single_spin_summation_inplace_(
        _void_p(A), ctypes.c_int64(nvir), ctypes.c_char_p(pattern_c), ctypes.c_double(alpha), ctypes.c_double(beta))
    return A


_lib.t4_transpose_add_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
_lib.t4_transpose_add_.restype = None
def t4_transpose_add_(src, dst, nvir):
    src = _as_array(src, np.float64, "src")
    dst = _as_array(dst, np.float64, "dst")
    if src.shape != (nvir,) * 4 or dst.shape != (nvir,) * 4:
        raise ValueError("src and dst must have shape (nvir, nvir, nvir, nvir)")
    _lib.t4_transpose_add_(_void_p(src), _void_p(dst), ctypes.c_int64(nvir))
    return dst


_lib.t4_spin_summation_quadruple_sym_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
_lib.t4_spin_summation_quadruple_sym_.restype = None
def t4_spin_summation_quadruple_sym_(src, B0, B1, B2, B3, nvir):
    src = _as_array(src, np.float64, "src")
    B0 = _as_array(B0, np.float64, "B0")
    B1 = _as_array(B1, np.float64, "B1")
    B2 = _as_array(B2, np.float64, "B2")
    B3 = _as_array(B3, np.float64, "B3")
    if src.shape != (nvir,) * 4 or B0.shape != src.shape or B1.shape != src.shape or B2.shape != src.shape or B3.shape != src.shape:
        raise ValueError("all T4 buffers must have shape (nvir, nvir, nvir, nvir)")
    _lib.t4_spin_summation_quadruple_sym_(_void_p(src), _void_p(B0), _void_p(B1), _void_p(B2), _void_p(B3), ctypes.c_int64(nvir))
    return B0, B1, B2, B3


_lib.r4_local_tri_divide_e_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
_lib.r4_local_tri_divide_e_.restype = None
def r4_local_tri_divide_e_(mycc, dt4, r4_local, mo_energy):
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    r4_local = _as_array(r4_local, np.float64, "r4_local")
    if r4_local.shape != (len(dt4.local_quadruples),) + (nvir,) * 4:
        raise ValueError("r4_local has inconsistent local T4 shape")
    local_ijkl = np.ascontiguousarray(np.asarray(dt4.local_quadruples, dtype=np.int32).reshape(-1, 4))
    eia = np.ascontiguousarray(mo_energy[:nocc, None] - mo_energy[None, nocc:] - mycc.level_shift, dtype=np.float64)
    _lib.r4_local_tri_divide_e_(
        _void_p(r4_local), _void_p(eia), _void_p(local_ijkl), ctypes.c_int64(r4_local.shape[0]), ctypes.c_int64(nvir))
    return r4_local


_lib.fill_local_data_ijkl_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64),
                                        ctypes.c_int64, ctypes.c_int64]
_lib.fill_local_data_ijkl_.restype = None
def fill_local_data_ijkl_(t4_local, send_data, requests, ijkl_to_local_idx, n_requests, nvir):
    t4_local = _as_array(t4_local, np.float64, "t4_local")
    send_data = _as_array(send_data, np.float64, "send_data")
    requests = _as_array(requests, np.int32, "requests")
    ijkl_to_local_idx = _as_array(ijkl_to_local_idx, np.int64, "ijkl_to_local_idx")
    _lib.fill_local_data_ijkl_(_double_p(t4_local), _double_p(send_data), _int32_p(requests),
                            _int64_p(ijkl_to_local_idx), ctypes.c_int64(n_requests), ctypes.c_int64(nvir))
    return send_data


_lib.unpack_t4_ijkl_single_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
_lib.unpack_t4_ijkl_single_.restype = None
def unpack_t4_ijkl_single_(t4_local, t4_blk, ijkl_to_local_idx, i, j, k, l, nocc, nvir):
    t4_local = _as_array(t4_local, np.float64, "t4_local")
    t4_blk = _as_array(t4_blk, np.float64, "t4_blk")
    ijkl_to_local_idx = _as_array(ijkl_to_local_idx, np.int64, "ijkl_to_local_idx")
    _lib.unpack_t4_ijkl_single_(_double_p(t4_local), _double_p(t4_blk), _int64_p(ijkl_to_local_idx),
                                    ctypes.c_int64(i), ctypes.c_int64(j), ctypes.c_int64(k), ctypes.c_int64(l),
                                    ctypes.c_int64(nocc), ctypes.c_int64(nvir))
    return t4_blk
