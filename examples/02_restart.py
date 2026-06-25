#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf
from pyscf.scf.chkfile import load_scf
from pyscf.data.elements import chemcore
import shutil

from distr_cc import RCCSDT
from distr_cc.distribute_t3 import DistributedT3IJK
from distr_cc import rccsdt_q

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    os.makedirs('output', exist_ok=True)
outdir = 'output'

atom = '''
O   -0.066999140   0.000000000   1.494354740
H    0.815734270   0.000000000   1.865866390
H    0.068855100   0.000000000   0.539142770
'''

mol = gto.M(atom=atom, basis="ccpvdz", verbose=0)
mf = scf.RHF(mol)
if rank == 0:
    mf.chkfile = 'output/mf.chk'
mf.kernel()

mycc = RCCSDT(mf, frozen=chemcore(mol), comm=comm)
mycc.max_cycle = 100
mycc.verbose = 4
mycc.kernel()

if rank == 0:
    np.save('output/t1.npy', mycc.tamps[0])
    np.save('output/t2.npy', mycc.tamps[1])
comm.Barrier()
dt3, t3_local = mycc.tamps[2]
dt3.save_to_disk(t3_local, prefix='output/')
comm.Barrier()

mol2, scf_res = load_scf('output/mf.chk')
if isinstance(mol2, dict):
    mol2 = gto.M(**mol2)
mf2 = scf.RHF(mol2)
mf2.__dict__.update(scf_res)
mf2._eri = mol2.intor("int2e")
mf2.converged = True

cc2 = RCCSDT(mf2, frozen=chemcore(mol2), comm=MPI.COMM_WORLD)
cc2.conv_tol = 1e-8
cc2.max_cycle = 1
cc2.verbose = 0
t1 = np.load('output/t1.npy')
t2 = np.load('output/t2.npy')
dt3, t3_local = DistributedT3IJK.load_from_disk('output/', MPI.COMM_WORLD, mmap_mode=None)
tamps = [t1, t2, (dt3, t3_local)]
cc2.kernel(tamps=tamps)

ref_ecorr = -0.2146745758026043
if rank == 0:
    print('RCCSDT correlation energy               % .12f' % mycc.e_corr)
    print('RCCSDT correlation energy after restart % .12f' % cc2.e_corr)
    print('RCCSDT correlation energy reference     % .12f' % ref_ecorr)

q_bracket, q_paren = rccsdt_q.kernel(cc2, comm=comm, blksize=8)

ref_q_bracket, ref_q_paren = -0.000440859544, -0.000488539476
if rank == 0:
    print('RCCSDT Q-bracket % .12f  Ref. % .12f  Diff % .12e' % (q_bracket, ref_q_bracket, q_bracket - ref_q_bracket))
    print('RCCSDT Q-paren   % .12f  Ref. % .12f  Diff % .12e' % (q_paren, ref_q_paren, q_paren - ref_q_paren))

comm.Barrier()
if rank == 0:
    shutil.rmtree(outdir, ignore_errors=True)
comm.Barrier()
