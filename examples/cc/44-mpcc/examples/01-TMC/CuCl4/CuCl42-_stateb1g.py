from pyscf import gto, scf, cc
import numpy as np
from numpy import linalg as la
import matplotlib
import matplotlib.pyplot as plt 
import copy
import time

if __name__ == '__main__':

    # Geometries taken from JCTC 2018, 14, 12, 6240–6252:
    # XYZ coordinates (AU) of  [CuCl4]2-
    Mol1 = [
    ['Cu', [ 0.0000000000000,   0.000000000000,    0.0000000000000]],
    ['Cl', [-3.0065358670850,  -3.006535867085,    0.0000000000000]],
    ['Cl', [ 3.0065358670850,   3.006535867085,    0.0000000000000]],
    ['Cl', [-3.0065358670850,   3.006535867085,    0.0000000000000]],
    ['Cl', [ 3.0065358670850,  -3.006535867085,    0.0000000000000]]
    ]

    #basis = 'sto-3g' 
    basis= '6-31g' 
    symmetry = True
    #basis= 'aug-cc-pwcvtz-dk.0.nw'

    pre_out= 'output/03-CuCl4_2-/'+ basis+ '/' 
    
    print("Computing [CuCl4]2- in ", basis, " Basis:")

    mol           = gto.M()
    mol.basis     = basis
#    mol.symmetry     = True
    mol.atom      = Mol1
    mol.unit      = 'bohr'
    mol.verbose   = 3
    mol.nelectron = 99 #97=29 + 4*17 
    mol.spin      = 1
    mol.charge    = -2
    mol.build()

#    mf = mol.UHF().newton()
    mf = mol.ROHF().newton()

    mf.irrep_nelec = {'Ag': (15,15), 'B1g': (5,4), 'B2g': (3,3), 'B3g': (3,3),'B1u': (6,6),'B2u': (9,9),'B3u': (9,9)}

    mf.run()
    mf.stability()
#   mf = scf.UHF(mol).run(verbose = 4)
    mf.analyze()
    mfe = mf.e_tot

#    mo0 = mf.mo_coeff
  # occ = mf.mo_occ

  # occ[0][49]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
  # occ[0][50]=1 

# # occ[49]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
# # occ[50]=1 

  # # Construct new dnesity matrix with new occpuation pattern
  # dm_u = mf.make_rdm1(mo0, occ)
  # # Apply mom occupation principle
  # mf = scf.addons.mom_occ(mf, mo0, occ)
  # # Start new SCF with new density matrix
  # mf.scf(dm_u)

  # mf.verbose = 5
  # mo1 = mf.stability()[0]
  # dm_u = mf.make_rdm1(mo1, occ)
  # mf.scf(dm_u)

  # mo1 = mf.stability()[0]
  # dm_u = mf.make_rdm1(mo1, occ)
  # mf.scf(dm_u)

  # mf.stability()

    mycc  = cc.CCSD(mf)

    freezing = True
    if freezing:
        # Freezing He, Ne, and Ar core in N, Cu, Cl, O 
        # => 58 core electrons, i.e., 26 core orbitals
        core = [i for i in range(29)]
        mycc.set(frozen = core)

    mycc.verbose = 4
    mycc.max_cycle = 100
    mycc.kernel()
    ecc = mycc.e_tot
