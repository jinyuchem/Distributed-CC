# Examples

This directory contains small runnable examples intended for quick validation on
a laptop or a small interactive MPI allocation.  They are not tuned production
input files.

Run the examples from the repository root with `mpirun`, for example:

```bash
mpirun -n 4 python examples/00_rccsdt_q_water.py
mpirun -n 4 python examples/01_rccsdtq_water.py
mpirun -n 4 python examples/02_restart.py
```

## Current Examples

- `00_rccsdt_q_water.py`: runs distributed RCCSDT followed by the perturbative
  `[Q]` and `(Q)` corrections on a small water dimer example.
- `01_rccsdtq_water.py`: runs distributed RCCSDTQ on a small water example.
- `02_restart.py`: demonstrates saving distributed T3 amplitudes, restarting
  RCCSDT from disk, and then running the perturbative `[Q]` and `(Q)`
  corrections.

## Production Examples

Production-scale run scripts and parameter templates will live in
`examples/production/`.  Those examples are intended to cover HPC job setup,
Slurm scripts, recommended `pytblis` builds, restart layout, logging options,
and parameter choices for long multi-node calculations.

