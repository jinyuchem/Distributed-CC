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

import unittest
from pyscf import gto, scf
from distr_cc import rccsdt, rccsdt_q
from mpi4py import MPI

EXPECTED_BRACKET_Q_ENERGY = -0.00044374834015582527
EXPECTED_PAREN_Q_ENERGY = -0.0004917163848923114

def setUpModule():
    global mol, rhf, mcc, mcc2
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .487)],
        [1 , (0. ,  .757 , .687)]]
    mol.symmetry = True
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()

    mcc = rccsdt.RCCSDT(rhf, comm=MPI.COMM_WORLD)
    mcc.conv_tol = 1e-10
    mcc.ccsdt()


def tearDownModule():
    global mol, rhf, mcc
    mol.stdout.close()
    del mol, rhf, mcc

class KnownValues(unittest.TestCase):
    def test_rccsdt_q(self):
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mcc, release_ijk_t3=False)
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)

    def test_blksize(self):
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mcc, blksize=2, release_ijk_t3=False)
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mcc, blksize=3, release_ijk_t3=False)
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mcc, blksize=4, release_ijk_t3=False)
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mcc, blksize=7, release_ijk_t3=False)
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)

    def test_split_jobs(self):
        njobs = 3
        e_q_bracket, e_q_paren = 0.0, 0.0
        for i in range(njobs):
            tmp1, tmp2 = rccsdt_q.kernel(mcc, blksize=4, job_idx=i, n_jobs=njobs, release_ijk_t3=False)
            e_q_bracket += tmp1
            e_q_paren += tmp2
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)
        njobs = 9
        e_q_bracket, e_q_paren = 0.0, 0.0
        for i in range(njobs):
            tmp1, tmp2 = rccsdt_q.kernel(mcc, blksize=6, job_idx=i, n_jobs=njobs, release_ijk_t3=False)
            e_q_bracket += tmp1
            e_q_paren += tmp2
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)
        njobs = 13
        e_q_bracket, e_q_paren = 0.0, 0.0
        for i in range(njobs):
            tmp1, tmp2 = rccsdt_q.kernel(mcc, blksize=3, job_idx=i, n_jobs=njobs, release_ijk_t3=False)
            e_q_bracket += tmp1
            e_q_paren += tmp2
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)

    def test_feedin_tamps_eris(self):
        mcc2 = rccsdt.RCCSDT(rhf, comm=MPI.COMM_WORLD)
        tamps = mcc.tamps
        eris = mcc2.ao2mo()
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mcc2, tamps=tamps, eris=eris, release_ijk_t3=False)
        self.assertAlmostEqual(e_q_bracket, EXPECTED_BRACKET_Q_ENERGY, 10)
        self.assertAlmostEqual(e_q_paren, EXPECTED_PAREN_Q_ENERGY, 10)


if __name__ == "__main__":
    print("Full Tests for rccsdt_q")
    unittest.main()
