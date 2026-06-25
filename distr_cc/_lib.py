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

import ctypes
import os
import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "build", "distr_cc.so"))

def _as_array(arr, dtype, name):
    out = np.asarray(arr)
    if out.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous")
    return out

def _void_p(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)

def _double_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def _int32_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

def _int64_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

def _longlong_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
