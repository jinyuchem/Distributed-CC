# Parameter and Flag Reference

This page summarizes common parameters and diagnostic flags for the distributed
RCCSDT, RCCSDT(Q), and RCCSDTQ drivers.

Set driver attributes before calling `kernel()`.

```python
mycc = RCCSDT(mf, frozen=chemcore(mol), comm=comm)
mycc.batch_size = 100
mycc.do_diis_max_t = True
mycc.nvir_diis = None
mycc.set_einsum_backend("pytblis")
mycc.kernel()
```

The same attribute-setting pattern applies to `RCCSDTQ`.

## RCCSDT

Common numerical and performance parameters:

- `max_cycle`, `conv_tol`, `conv_tol_normt`: convergence controls.
- `batch_size`: occupied-triple batch size for distributed T3 work and T3
  communication. Larger values reduce batching overhead but increase temporary
  memory.
- `set_einsum_backend("pytblis")`: use `pytblis` tensor contractions when it is
  installed and built for the target machine.
- See the DIIS section below for DIIS convergence and storage controls.

Useful diagnostics:

- `verbose`: normal rank-0 driver output.
- `log_memory`: enable memory logging.
- `log_memory_per_iter`: include per-batch or per-task memory logs.
- `log_memory_all_ranks`: print memory logs from every rank. The default is
  rank 0 only.
- `log_highest_t_contractions`: enable detailed high-rank contraction timing.
- `log_highest_t_contractions_all_ranks`: extend contraction timing to every rank.
- `log_highest_t_communication`: write per-rank communication CSV logs to
  `communication_log_dir`.
- `communication_log_dir`: directory for communication CSV logs. Default:
  `comm_logs`.
- `contraction_log_dir`: directory for per-task contraction CSV logs when a
  method writes them. Default: `contraction_logs`. This is currently used by
  the perturbative `(Q)` driver.
- `log_allreduce_timing`: print rank-0 allreduce timing summaries.

## RCCSDT(Q)

The perturbative `[Q]` and `(Q)` correction is called as a function on a
converged `RCCSDT` object:

```python
q_bracket, q_paren = rccsdt_q.kernel(
    mycc,
    comm=comm,
    blksize=8,
    job_idx=0,
    n_jobs=1,
    log_redistribution=False,
)
```

Important arguments:

- `blksize`: virtual-orbital block size for the ABCD task loop and T3
  IJK-to-ABC redistribution. Larger values reduce overhead but increase memory.
- `job_idx`, `n_jobs`: split the full perturbative task list across independent
  jobs. `job_idx` is 0-based. The split depends on `blksize`, so use the same
  `blksize` for all jobs in one split run.
- `log_redistribution`: print T3 redistribution information during the
  IJK-to-ABC transform.

When `kernel` receives IJK-distributed T3 amplitudes, it builds an
ABC-distributed T3 copy internally. Memory-constrained runs should assume both
IJK and ABC T3 layouts may remain resident. To control object lifetime
explicitly, call `rccsdt_q.prepare_tamps_for_q`, delete every reference to the
IJK T3 amplitudes, and then pass the prepared ABC amplitudes to `kernel`.

`RCCSDT(Q)` reads the diagnostic flags from the `RCCSDT` object. In particular,
`log_highest_t_communication=True` writes per-rank communication CSV files, and
`log_highest_t_contractions=True` with
`log_highest_t_contractions_all_ranks=True` writes per-task `[Q]` and `(Q)`
energy CSV files.

## RCCSDTQ

`RCCSDTQ` uses the same driver attributes as `RCCSDT`, but the distributed
max-rank amplitude is T4 instead of T3.

```python
mycc = RCCSDTQ(mf, frozen=chemcore(mol), comm=comm)
mycc.batch_size = 100
mycc.do_diis_max_t = True
mycc.nvir_diis = None
mycc.set_einsum_backend("pytblis")
mycc.kernel()
```

For `RCCSDTQ`, `batch_size` controls occupied-quadruple T4 batches and T4
communication. The logging flags have the same meaning as for `RCCSDT`;
communication logs are written as per-rank T4 CSV files when
`log_highest_t_communication=True`.

## DIIS Settings

The `RCCSDT` and `RCCSDTQ` drivers support two DIIS modes:

- Standard DIIS, used when `do_diis_max_t = False`, extrapolates only the
  replicated lower-order amplitudes. For `RCCSDT`, this means T1/T2. For
  `RCCSDTQ`, this means T1/T2/T3.
- Max-rank-amplitude DIIS, used when `do_diis_max_t = True`, extrapolates the
  lower-order amplitudes plus the distributed highest-rank amplitude: T3 for
  `RCCSDT` and T4 for `RCCSDTQ`.

The max-rank-amplitude DIIS path is necessary for difficult-to-converge
high-order CC iterations, but it requires more memory and may need scratch
storage for production runs.

Common DIIS controls:

- `diis`: enable or disable DIIS.
- `diis_space`: number of DIIS history vectors to keep. A larger value can make
  convergence more robust, but memory and scratch usage scale roughly linearly
  with this value.
- `diis_start_cycle`: first iteration where DIIS is allowed.
- `diis_start_energy_diff`: DIIS is used only when the absolute energy change
  is below this threshold.
- `do_diis_max_t`: choose whether the highest-rank amplitude is included in the
  DIIS vector.
- `nvir_diis`: active virtual dimension used for the highest-rank amplitude in
  max-rank DIIS. `None` means all virtual orbitals. A smaller value reduces DIIS
  memory, especially for T4, but the DIIS update then acts directly only on that
  active virtual block. The highest-rank part of the DIIS vector scales as
  `nvir_diis**3` for T3 and `nvir_diis**4` for T4.
- `incore_complete`: keep standard PySCF DIIS history in memory when standard
  DIIS is used.
- `diis_scratch`: scratch directory for distributed max-rank DIIS history. Use
  `None` to keep all DIIS history in memory. For large jobs, this should point
  to a fast filesystem with enough per-rank capacity.
- `diis_scratch_start`: number of DIIS slots kept in memory before writing the
  remaining slots to `diis_scratch`. For example, `0` means all history slots
  use scratch when `diis_scratch` is set.
- `diis_scratch_cleanup`: remove scratch files when the calculation finishes.
- `diis_scratch_mmap`: use memory-mapped scratch arrays.

If DIIS memory is too large, first consider reducing `nvir_diis` or moving DIIS
history to scratch with `diis_scratch`.
