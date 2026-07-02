# Distributed-CC

Distributed-CC is a development package for distributed high-order
coupled-cluster methods built on PySCF, MPI, and native C helper kernels.

It currently provides:

- distributed spin-restricted CCSDT (RCCSDT)
- distributed perturbative [Q] and (Q) energy corrections
- distributed RCCSDTQ

The code is developed at the Initiative for Computational Catalysis at the Flatiron Institute.

If you find this package useful for your scientific research, please cite the work as:

 - Y. Jin, C. Hillenbrand, T. C. Berkelbach, and H. Zhai. "High-performance parallel implementation of high-order coupled-cluster theories." arXiv: [10.48550/arXiv.2607.00981](https://arxiv.org/abs/2607.00981).


## Repository Layout

```text
distr_cc/      Python package
src/           C helper kernels built by CMake
tests/         pytest tests
examples/      runnable examples
doc/           documentation
```

## Requirements

Distributed-CC requires:

- Python 3.9 or newer
- MPI and `mpi4py`
- CMake and a C/C++ compiler
- NumPy
- PySCF 2.13.1 or newer

The native helper library is required and must be built before running the distributed methods.

[`pytblis`](https://github.com/chillenb/pytblis) is optional but strongly recommended for production calculations,
especially on Linux HPC systems. It can substantially improve tensor contraction performance.

## Installation

Distributed-CC uses native C helper kernels. Build the native library before installing or importing the Python package.

### Recommended editable install

From the repository root, compile the C code first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Then install the Python package in editable mode:

```bash
python -m pip install -e .
```

### Manual path-based use

For development or local testing, you may skip `pip install -e .` if all Python dependencies are already installed in the active environment.

```python
import sys

sys.path.insert(0, "/absolute/path/to/Distributed-CC")

from distr_cc import RCCSDT, RCCSDTQ
from distr_cc import rccsdt_q
```

or with `PYTHONPATH`:

```bash
export PYTHONPATH=/absolute/path/to/Distributed-CC:$PYTHONPATH
mpirun -n 4 python examples/00_rccsdt_q_water.py
```

### Native build options

The default native build uses portable optimization flags. Native CPU tuning and OpenMP are opt-in:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDISTR_CC_NATIVE=ON -DDISTR_CC_OPENMP=ON
cmake --build build --parallel
```

If you use GNU compilers, `g++` 10 or newer is required for the symmetrized tensor contraction code.

## Running Examples

Run examples with `mpirun`:

```bash
mpirun -n 4 python examples/00_rccsdt_q_water.py
mpirun -n 4 python examples/01_rccsdtq_water.py
mpirun -n 4 python examples/02_restart.py
```

Minimal example:

```python
from mpi4py import MPI
from pyscf import gto, scf

from distr_cc import RCCSDT, RCCSDTQ
from distr_cc import rccsdt_q

comm = MPI.COMM_WORLD

mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g")
mf = scf.RHF(mol).run()

mycc = RCCSDT(mf, comm=comm)
mycc.kernel()

sq_corr, pq_corr = rccsdt_q.kernel(mycc, comm=comm)

myccq = RCCSDTQ(mf, comm)
myccq.kernel()

if comm.rank == 0:
    print("RCCSDT correlation energy:", mycc.e_corr)
    print("[Q] correction:", sq_corr)
    print("(Q) correction:", pq_corr)
    print("RCCSDTQ correlation energy:", myccq.e_corr)
```

For recommended performance settings after installing `pytblis`, set the einsum backend before running the kernel:

```python
mycc.set_einsum_backend("pytblis")
```

<!-- ## Python Fallbacks

Python fallbacks for native helper kernels are intended for debugging only.
They are disabled by default. To allow them explicitly, set either:

```bash
export DISTR_CC_ALLOW_PYTHON_FALLBACK=1
```

or set `allow_python_fallback = True` on the relevant driver object. -->
