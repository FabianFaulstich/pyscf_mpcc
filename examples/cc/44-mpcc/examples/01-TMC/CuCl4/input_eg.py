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
    #basis= 'aug-cc-pwcvtz-dk.0.nw'

    pre_out= 'output/03-CuCl4_2-/'+ basis+ '/' 
    
    print("Computing [CuCl4]2- in ", basis, " Basis:")

    mol           = gto.M()
    mol.basis     = basis
    mol.symmetry = True
    mol.atom      = Mol1
    mol.unit      = 'bohr'
    mol.verbose   = 5
    mol.nelectron = 99 #97=29 + 4*17 
    mol.spin      = 1
    mol.charge    = -2
    mol.build()

    mf = mol.UHF().newton()
#    mf = mol.ROHF().newton()
    mf.irrep_nelec = {'Ag': (15,15), 'B1g': (5,5), 'B2g': (3,2), 'B3g': (3,3),'B1u': (6,6),'B2u': (9,9),'B3u': (9,9)}
#   mf = mol.ROHF().newton()
    mf.run()
    print("MO Occupation:")
    print(mf.mo_occ)

    mf.stability()
 #  mo0 = mf.mo_coeff
 #  occ = mf.mo_occ

 #  occ[1][45]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
 #  occ[1][52]=1 

 #  dm_u = mf.make_rdm1(mo0, occ)
 #  mf = scf.addons.mom_occ(mf, mo0, occ)
 #  mf.scf(dm_u)

 #  mf.stability()


#    mf = scf.UHF(mol).run(verbose = 4)
    mf.analyze()
    mfe = mf.e_tot

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
