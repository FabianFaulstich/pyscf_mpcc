import numpy as np
from pyscf import ao2mo, gto, scf, mp, cc, lo
from pyscf.tools import ring
import matplotlib.pyplot as plt


N = 10
bd = 1.5
atom = 'H'
basis = 'ccpvdz'

structure = ring.make(N, bd)
atom = [(atom +' %f %f %f' % xyz) for xyz in structure]

mol = gto.M()
mol.basis = basis
mol.atom  = atom
mol.verbose = 3
mol.build()

mf = mol.RHF().run()
mf.verbose = 3

orbitals = mf.mo_coeff
nocc = mol.nelectron//2
c_mo = np.zeros((orbitals.shape),orbitals.dtype)

#orbitals = mf.mo_coeff[:,mf.mo_occ>0]
pm_occ = lo.PM(mol, mf.mo_coeff[:,:nocc] , mf)
pm_occ.pop_method = 'meta-lowdin'  
C = pm_occ.kernel()
c_mo[:, :nocc] = C

pm_vir = lo.PM(mol, mf.mo_coeff[:,nocc:] , mf)
pm_vir.pop_method = 'meta-lowdin'  
C = pm_vir.kernel()
c_mo[:, nocc:] = C

# Building MP2 objects for the _make_eris routine
mymp = mp.MP2(mf, mo_coeff=c_mo)
mymp.max_cycle = 30

mymp.diis_space = 8
#Running local MP2 should be conceptually very easy!
eris = mp.mp2._make_eris(mymp)
#breakpoint()
lo_mp = mp.mp2._iterative_kernel(mymp, eris, verbose=5) 

# Running regular Mp2 and CCSD 
#mymp = mp.MP2(mf).run()
