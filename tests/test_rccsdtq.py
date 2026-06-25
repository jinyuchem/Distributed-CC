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

import os
import tempfile
import numpy as np
import unittest
from mpi4py import MPI
from pyscf import gto, scf, cc
from pyscf.scf import chkfile
from distr_cc import rccsdtq
from distr_cc.distribute_t4 import DistributedT4IJKL

EXPECTED_RCCSDTQ_ENERGY = -0.04958398899351724
ENERGY_TOL = 5e-8

def assert_allranks_close(testcase, value, expected, atol=ENERGY_TOL):
    comm = MPI.COMM_WORLD
    local_value = float(value)
    all_finite = comm.allreduce(int(np.isfinite(local_value)), op=MPI.MIN)
    testcase.assertEqual(all_finite, 1)
    local_error = abs(local_value - float(expected))
    max_error = comm.allreduce(local_error, op=MPI.MAX)
    testcase.assertLessEqual(max_error, atol)


def max_abs_or_zero(a):
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    return abs(a).max()


def setUpModule():
    global mol, mf, eris, mycc
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-8
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.kernel()

    comm = MPI.COMM_WORLD

    mycc = rccsdtq.RCCSDTQ(mf, comm=comm)
    # mycc.set_einsum_backend('pytblis')
    mycc.conv_tol = 1e-8
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)

def tearDownModule():
    global mol, mf, eris, mycc
    mol.stdout.close()
    del mol, mf, eris, mycc


class KnownValues(unittest.TestCase):

    def test_known_values(self):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.conv_tol = 1e-8
        cc1.kernel(eris=eris)
        assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_batch_size(self):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.conv_tol = 1e-8
        cc1.batch_size = 3
        cc1.verbose = 8
        cc1.kernel()
        assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)
        cc2 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc2.conv_tol = 1e-8
        cc2.batch_size = 11
        cc2.kernel()
        assert_allranks_close(self, cc2.e_corr, EXPECTED_RCCSDTQ_ENERGY)
        cc3 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc3.conv_tol = 1e-8
        cc3.batch_size = 17
        cc3.kernel()
        assert_allranks_close(self, cc3.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_no_do_diis_max_t(self):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.do_diis_max_t = False
        cc1.conv_tol = 1e-8
        cc1.kernel()
        assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_do_diis_max_t(self):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.do_diis_max_t = True
        cc1.conv_tol = 1e-8
        cc1.kernel()
        assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_nvir_diis(self):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.do_diis_max_t = True
        cc1.nvir_diis = 4
        cc1.conv_tol = 1e-8
        cc1.kernel()
        assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)
        cc2 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc2.do_diis_max_t = True
        cc2.nvir_diis = 1
        cc2.conv_tol = 1e-8
        cc2.kernel()
        assert_allranks_close(self, cc2.e_corr, EXPECTED_RCCSDTQ_ENERGY)
        cc3 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc3.do_diis_max_t = True
        cc3.nvir_diis = 13
        cc3.conv_tol = 1e-8
        cc3.kernel()
        assert_allranks_close(self, cc3.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_diis_outcore(self):
        comm = MPI.COMM_WORLD
        tmpdir = tempfile.TemporaryDirectory(prefix="distr_cc_diis_outcore_") if comm.rank == 0 else None
        scratch = comm.bcast(tmpdir.name if tmpdir is not None else None, root=0)
        try:
            cc1 = rccsdtq.RCCSDTQ(mf, comm=comm)
            cc1.diis = True
            cc1.do_diis_max_t = True
            cc1.nvir_diis = 9
            cc1.incore_complete = False
            cc1.diis_scratch = scratch
            cc1.diis_scratch_start = 0
            cc1.diis_scratch_cleanup = False
            cc1.diis_scratch_mmap = False
            cc1.conv_tol = 1e-8
            cc1.kernel()
            assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)

            comm.Barrier()
            remaining = sorted(os.listdir(scratch)) if comm.rank == 0 else None
            remaining = comm.bcast(remaining, root=0)
            self.assertTrue(any("-X" in name for name in remaining))
            self.assertTrue(any("-E" in name for name in remaining))
        finally:
            comm.Barrier()
            if tmpdir is not None:
                tmpdir.cleanup()
            comm.Barrier()

    def test_no_diis(self):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.diis = False
        cc1.conv_tol = 1e-8
        cc1.kernel()
        assert_allranks_close(self, cc1.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_restart(self):
        comm = MPI.COMM_WORLD
        tmpdir = tempfile.TemporaryDirectory(prefix="distr_cc_rccsdtq_restart_") if comm.rank == 0 else None
        outdir = comm.bcast(tmpdir.name if tmpdir is not None else None, root=0)
        prefix = os.path.join(outdir, "")
        try:
            self._test_restart(prefix, comm)
        finally:
            comm.Barrier()
            if tmpdir is not None:
                tmpdir.cleanup()
            comm.Barrier()

    def _test_restart(self, prefix, comm):
        cc1 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD)
        cc1.conv_tol = 1e-8
        cc1.kernel()

        t1, t2, t3 = cc1.tamps[:3]
        if cc1.rank == 0:
            chkfile.dump_scf(mol, prefix + 'hf.chk', mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
            np.save(prefix + 't1.npy', t1)
            np.save(prefix + 't2.npy', t2)
            np.save(prefix + 't3.npy', t3)
        comm.Barrier()
        dt4, t4_local = cc1.tamps[3]
        dt4.save_to_disk(t4_local, prefix)

        mol2, scf_dict = scf.chkfile.load_scf(prefix + 'hf.chk')
        if isinstance(mol2, dict):
            mol2 = gto.M(**mol2)
        mol2.output = '/dev/null'
        mf2 = scf.RHF(mol2)
        mf2.__dict__.update(scf_dict)
        mf2._eri = mol2.intor("int2e")
        mf2.converged = True

        cc2 = rccsdtq.RCCSDTQ(mf2, comm=MPI.COMM_WORLD)
        cc2.conv_tol = 1e-8
        cc2.max_cycle = 1
        cc2.verbose = 0
        t1 = np.load(prefix + 't1.npy')
        t2 = np.load(prefix + 't2.npy')
        t3 = np.load(prefix + 't3.npy')
        dt4, t4_local = DistributedT4IJKL.load_from_disk(prefix, MPI.COMM_WORLD, mmap_mode=None)
        tamps = [t1, t2, t3, (dt4, t4_local)]
        cc2.kernel(tamps=tamps)
        self.assertTrue(cc2.converged)
        assert_allranks_close(self, cc2.e_corr, EXPECTED_RCCSDTQ_ENERGY)

    def test_two_electrons(self):
        mol = gto.M(atom='He', basis=('631g', [[0, (.2, 1)], [0, (.5, 1)]]), verbose=0)
        mf = scf.RHF(mol).run()
        mycc1 = cc.CCSD(mf).run(conv_tol=1e-10)
        mycc2 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD).run(conv_tol=1e-10)
        dt4, t4_local = mycc2.tamps[3]
        self.assertAlmostEqual(mycc1.e_corr, mycc2.e_corr, 8)
        self.assertAlmostEqual(max_abs_or_zero(mycc1.t1 - mycc2.t1), 0, 8)
        self.assertAlmostEqual(max_abs_or_zero(mycc1.t2 - mycc2.t2), 0, 8)
        self.assertAlmostEqual(max_abs_or_zero(mycc2.tamps[2]), 0, 9)
        self.assertEqual(dt4.nocc4, 0)
        self.assertEqual(t4_local.size, 0)
        self.assertAlmostEqual(max_abs_or_zero(t4_local), 0, 9)

    def test_vs_fci(self):
        from pyscf import fci
        mol = gto.M(atom='''
            Li .8    0.2      0.
            H  0.   -0.19    0.587
            H  0.   0.757    0.587''', basis='6-31g', charge=1, verbose=0)
        mf = mol.RHF().run(conv_tol=1e-12)
        mycc2 = rccsdtq.RCCSDTQ(mf, comm=MPI.COMM_WORLD).run(conv_tol=1e-10)
        ref = fci.FCI(mf).run().e_tot
        self.assertAlmostEqual(mycc2.e_tot, ref, 8)


if __name__ == "__main__":
    print("Full Tests for rccsdtq.RCCSDTQ")
    unittest.main()
