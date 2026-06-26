#!/usr/bin/env python

from mpi4py import MPI
from pyscf import gto, scf
from distr_cc import RCCSDTQ
from pyscf.data.elements import chemcore

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

atom = '''
O   -0.066999140   0.000000000   1.494354740
H    0.815734270   0.000000000   1.865866390
H    0.068855100   0.000000000   0.539142770
'''

mol = gto.M(atom=atom, basis="ccpvdz", verbose=0)
mf = scf.RHF(mol).run(verbose=0)

mycc = RCCSDTQ(mf, frozen=chemcore(mol), comm=comm)
mycc.max_cycle = 100
mycc.conv_tol = 1e-9
mycc.conv_tol_normt = 1e-7
mycc.verbose = 8
# mycc.set_einsum_backend('pytblis')
mycc.batch_size = 13
mycc.diis = True
mycc.do_diis_max_t = True
mycc.nvir_diis = 6
mycc.incore_complete = True
mycc.diis_scratch = None
mycc.diis_scratch_start = 0
mycc.diis_scratch_cleanup = True
mycc.diis_scratch_mmap = False

mycc.log_memory = False
mycc.log_memory_per_iter = False
mycc.log_memory_all_ranks = False
mycc.log_highest_t_contractions = False
mycc.log_highest_t_contractions_all_ranks = False
mycc.log_highest_t_communication = False

mycc.use_mpi_progress_thread = False
mycc.mpi_progress_poll_interval = 0.001
mycc.gil_punctuate_duration = 0.0001
mycc.gil_punctuate_interval = 10

mycc.kernel()

ref_ecorr = -0.21513329175363083
if rank == 0:
    print(f"RCCSDTQ correlation energy: {mycc.e_corr:.12f}  Ref: {ref_ecorr:.12f}  Diff: {mycc.e_corr - ref_ecorr:.12e}")
