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
from distr_cc._lib import _as_array, _double_p, _int32_p, _int64_p, _longlong_p, _void_p, _lib

def _has_symbol(name):
    return hasattr(_lib, name)

_lib.t3_single_spin_summation_inplace_.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_char_p,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.t3_single_spin_summation_inplace_.restype = None
def t3_single_spin_summation_inplace_(A, nvir, pattern, alpha=1.0, beta=0.0):
    A = _as_array(A, np.float64, "A")
    pattern_c = pattern.encode("utf-8")
    _lib.t3_single_spin_summation_inplace_(_void_p(A), ctypes.c_int64(nvir), ctypes.c_char_p(pattern_c),
                                           ctypes.c_double(alpha), ctypes.c_double(beta))
    return A


_lib.t3_transpose_add_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
_lib.t3_transpose_add_.restype = None
def t3_transpose_add_(A, B, nvir):
    A = _as_array(A, np.float64, "A")
    B = _as_array(B, np.float64, "B")
    _lib.t3_transpose_add_(_void_p(A), _void_p(B), ctypes.c_int64(nvir))
    return B


_lib.t3_spin_summation_triple_sym_.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
]
_lib.t3_spin_summation_triple_sym_.restype = None
def t3_spin_summation_triple_sym_(A, B0, B1, B2, nvir):
    A = _as_array(A, np.float64, "A")
    B0 = _as_array(B0, np.float64, "B0")
    B1 = _as_array(B1, np.float64, "B1")
    B2 = _as_array(B2, np.float64, "B2")
    _lib.t3_spin_summation_triple_sym_(_void_p(A), _void_p(B0), _void_p(B1), _void_p(B2), ctypes.c_int64(nvir))
    return B0, B1, B2


_lib.fill_local_data_ijk_.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int64,
    ctypes.c_int64,
]
_lib.fill_local_data_ijk_.restype = None
def fill_local_data_ijk_(t3_local, send_data, requests, ijk_to_local_idx, n_requests, nvir):
    t3_local = _as_array(t3_local, np.float64, "t3_local")
    send_data = _as_array(send_data, np.float64, "send_data")
    requests = _as_array(requests, np.int32, "requests")
    ijk_to_local_idx = _as_array(ijk_to_local_idx, np.int64, "ijk_to_local_idx")
    _lib.fill_local_data_ijk_(_double_p(t3_local), _double_p(send_data), _int32_p(requests),
                                _int64_p(ijk_to_local_idx), ctypes.c_int64(n_requests), ctypes.c_int64(nvir))


_lib.take_t3_ijk_single_.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
_lib.take_t3_ijk_single_.restype = None
def take_t3_ijk_single_(t3_local, t3_blk, ijk_to_local_idx, i, j, k, nocc, nvir):
    t3_local = _as_array(t3_local, np.float64, "t3_local")
    t3_blk = _as_array(t3_blk, np.float64, "t3_blk")
    ijk_to_local_idx = _as_array(ijk_to_local_idx, np.int64, "ijk_to_local_idx")
    _lib.take_t3_ijk_single_(_double_p(t3_local), _double_p(t3_blk), _int64_p(ijk_to_local_idx),
        ctypes.c_int64(i), ctypes.c_int64(j), ctypes.c_int64(k), ctypes.c_int64(nocc), ctypes.c_int64(nvir))


_lib.pack_interleaved.argtypes = [
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_lib.pack_interleaved.restype = None
def pack_interleaved(n, abc, i, j, k, val, dest):
    abc = _as_array(abc, np.int32, "abc")
    i = _as_array(i, np.int32, "i")
    j = _as_array(j, np.int32, "j")
    k = _as_array(k, np.int32, "k")
    val = _as_array(val, np.float64, "val")
    dest = _as_array(dest, np.float64, "dest")
    _lib.pack_interleaved(ctypes.c_int64(n), _int32_p(abc), _int32_p(i), _int32_p(j), _int32_p(k), _double_p(val), _double_p(dest))


_lib.unpack_received_data_indices.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_int64,
]
_lib.unpack_received_data_indices.restype = None
def unpack_received_data_indices(recv_buffer, recv_counts, recv_displs, t3_local_abc, global_mapping_table, size, nocc, nvir):
    recv_buffer = _as_array(recv_buffer, np.float64, "recv_buffer")
    recv_counts = _as_array(recv_counts, np.int64, "recv_counts")
    recv_displs = _as_array(recv_displs, np.int64, "recv_displs")
    t3_local_abc = _as_array(t3_local_abc, np.float64, "t3_local_abc")
    global_mapping_table = _as_array(global_mapping_table, np.int64, "global_mapping_table")
    _lib.unpack_received_data_indices(_double_p(recv_buffer), _int64_p(recv_counts), _int64_p(recv_displs),
                                        _double_p(t3_local_abc), _int64_p(global_mapping_table),
                                        ctypes.c_int(size), ctypes.c_int64(nocc), ctypes.c_int64(nvir))


_lib.pack_redistributed_send_buffer_c.argtypes = [
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_double),
]
_lib.pack_redistributed_send_buffer_c.restype = None
def pack_redistributed_send_buffer_c(n_triples, nvir, size, chunk_i, chunk_j, chunk_k, t3_chunk,
                                      rank_offsets, pack_abc_idx, pack_g_idx, pack_perm, send_buffer):
    chunk_i = _as_array(chunk_i, np.int64, "chunk_i")
    chunk_j = _as_array(chunk_j, np.int64, "chunk_j")
    chunk_k = _as_array(chunk_k, np.int64, "chunk_k")
    t3_chunk = _as_array(t3_chunk, np.float64, "t3_chunk")
    rank_offsets = _as_array(rank_offsets, np.int64, "rank_offsets")
    pack_abc_idx = _as_array(pack_abc_idx, np.int64, "pack_abc_idx")
    pack_g_idx = _as_array(pack_g_idx, np.int64, "pack_g_idx")
    pack_perm = _as_array(pack_perm, np.int64, "pack_perm")
    send_buffer = _as_array(send_buffer, np.float64, "send_buffer")
    _lib.pack_redistributed_send_buffer_c(
        ctypes.c_longlong(n_triples), ctypes.c_longlong(nvir), ctypes.c_longlong(size),
        _longlong_p(chunk_i), _longlong_p(chunk_j), _longlong_p(chunk_k), _double_p(t3_chunk),
        _longlong_p(rank_offsets), _longlong_p(pack_abc_idx), _longlong_p(pack_g_idx),
        _longlong_p(pack_perm), _double_p(send_buffer)
    )
