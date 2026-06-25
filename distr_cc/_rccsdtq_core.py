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

"""Tentative private RCCSDTQ core helpers.

This module is kept only to support the initial distributed RCCSDTQ port.
It is expected to be removed once the corresponding PySCF-native RCCSDTQ
implementation can be imported directly.
"""

import sys
import numpy as np
import ctypes

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distr_cc._c_rccsdtq import t4_project_1_minus_p4_p31_inplace_ as _native_t4_project_1_minus_p4_p31_inplace_
from pyscf import lib

_libccsdt = lib.load_library('libccsdt')

def t4_project_1_minus_p4_p31_inplace_(A, nocc4, nvir, alpha=1.0, beta=0.0):
    return _native_t4_project_1_minus_p4_p31_inplace_(A, nocc4, nvir, alpha=alpha, beta=beta)

def t4_add_(t4, r4, nocc4, nvir):
    assert t4.dtype == np.float64 and t4.flags['C_CONTIGUOUS'], "t4 must be a contiguous float64 array"
    assert r4.dtype == np.float64 and r4.flags['C_CONTIGUOUS'], "r4 must be a contiguous float64 array"
    _libccsdt.t4_add_(t4.ctypes.data_as(ctypes.c_void_p), r4.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int64(nocc4), ctypes.c_int64(nvir))
    return t4


class _IMDS:

    def __init__(self):
        self.t1_fock = None
        self.t1_eris = None
        self.F_oo = None
        self.F_vv = None
        self.W_oooo = None
        self.W_ovvo = None
        self.W_ovov = None
        self.W_vooo = None
        self.W_vvvo = None
        self.W_vvvv = None
        self.W_ovvvoo = None
        self.W_ovvovo = None
        self.W_vooooo = None
        self.W_vvoooo = None
        self.W_vvvvoo = None
