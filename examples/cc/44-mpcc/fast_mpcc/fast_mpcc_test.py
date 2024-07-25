#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD calculation.
'''

import pyscf
from pyscf import mp, cc

mol = pyscf.M(
    atom = 'N 0 0 0',
    basis = 'dz',
    spin = 3)

mf = mol.UHF().newton()
mf = mf.run()
mo1 = mf.stability()[0]

dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mo1 = mf.stability()[0]

mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()

mymp = mp.UMP2(mf).run()

mycc = cc.umpcc_fast_driver.CCSD(mf)

mycc.kernel()

print('CCSD correlation energy', mycc.e_corr)
