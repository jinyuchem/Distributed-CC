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

__all__ = ["RCCSDT", "RCCSDTQ"]


def __getattr__(name):
    try:
        if name == "RCCSDT":
            from distr_cc.rccsdt import RCCSDT
            return RCCSDT
        if name == "RCCSDTQ":
            from distr_cc.rccsdtq import RCCSDTQ
            return RCCSDTQ
    except ModuleNotFoundError as err:
        missing = err.name or ""
        if missing == "pyscf" or missing.startswith("pyscf."):
            raise ModuleNotFoundError(
                "distr_cc requires PySCF that provides pyscf.cc.rccsdt and related modules. "
                "Install or expose that PySCF checkout before importing RCCSDT/RCCSDTQ."
            ) from err
        raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
