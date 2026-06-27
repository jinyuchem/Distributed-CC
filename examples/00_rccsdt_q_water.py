#!/usr/bin/env python

from mpi4py import MPI
from pyscf import gto, scf
from distr_cc import RCCSDT
from distr_cc import rccsdt_q
from pyscf.data.elements import chemcore

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

atom = '''
O   -0.066999140   0.000000000   1.494354740
H    0.815734270   0.000000000   1.865866390
H    0.068855100   0.000000000   0.539142770
O    0.062547750   0.000000000  -1.422632080
H   -0.406965400  -0.760178410  -1.771744500
H   -0.406965400   0.760178410  -1.771744500
'''

mol = gto.M(atom=atom, basis="ccpvdz", verbose=0)
mf = scf.RHF(mol).run(verbose=0, conv_tol=1e-12)

mycc = RCCSDT(mf, frozen=chemcore(mol), comm=comm)
mycc.max_cycle = 100
mycc.conv_tol = 1e-9
mycc.conv_tol_normt = 1e-7
mycc.verbose = 5
# mycc.set_einsum_backend('pytblis')
mycc.batch_size = 11
mycc.diis = True
mycc.do_diis_max_t = True
mycc.nvir_diis = 13
mycc.incore_complete = True
mycc.diis_scratch = None
mycc.diis_scratch_start = 0
mycc.diis_scratch_cleanup = True
mycc.diis_scratch_mmap = False
mycc.log_highest_t_communication = False
mycc.log_highest_t_contractions = False
mycc.log_highest_t_contractions_all_ranks = False
mycc.log_memory = False
mycc.log_memory_all_ranks = False
mycc.log_memory_per_iter = False
mycc.use_mpi_progress_thread = False
mycc.mpi_progress_poll_interval = 0.001
mycc.gil_punctuate_duration = 0.0001
mycc.gil_punctuate_interval = 10
mycc.kernel()

q_bracket, q_paren = 0.0, 0.0
njobs = 4
for i in range(njobs):
    tmp1, tmp2 = rccsdt_q.kernel(mycc, comm=comm, blksize=4, job_idx=i, n_jobs=njobs)
    q_bracket += tmp1
    q_paren += tmp2

ref_ecorr = -0.431122120377
if rank == 0:
    print(f"RCCSDT correlation energy: {mycc.e_corr:.12f}  Ref: {ref_ecorr:.12f}  Diff: {mycc.e_corr - ref_ecorr: .12e}")

ref_q_bracket, ref_q_paren = -0.00087424734798, -0.00099299809898
if rank == 0:
    print(f"[Q] correction: {q_bracket:.12f}    Ref: {ref_q_bracket:.12f}    Diff: {q_bracket - ref_q_bracket: .12e}")
    print(f"(Q) correction: {q_paren:.12f}    Ref: {ref_q_paren:.12f}    Diff: {q_paren - ref_q_paren: .12e}")
