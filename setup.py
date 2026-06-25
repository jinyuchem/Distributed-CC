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

from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


setup(
    name="Distributed-CC",
    version="0.1.0.dev0",
    description="Distributed high-order coupled-cluster methods for PySCF",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="The Distributed-CC Developers",
    author_email="yjin@flatironinstitute.org",
    license="Apache-2.0",
    url="https://github.com/jinyuchem/Distributed-CC",
    packages=["Distributed-CC"],
    python_requires=">=3.9",
    install_requires=["numpy", "pyscf>=2.13.1", "mpi4py"],
)
