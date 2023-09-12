from pyscf import gto, scf, cc
import numpy as np
from numpy import linalg as la
import matplotlib
import matplotlib.pyplot as plt
import copy
import time
from pyscf import symm 

def get_wvf_symm(mol,mf):

    ovlp_ao = mf.get_ovlp()
    orbsyma, orbsymb = mf.get_orbsym(mf.mo_coeff, ovlp_ao)
    orbsyma_in_d2h = np.asarray(orbsyma) % 10
    orbsymb_in_d2h = np.asarray(orbsymb) % 10
    tot_sym = 0
    noccsa = [sum(orbsyma_in_d2h[mf.mo_occ[0]>0]==ir) for ir in mol.irrep_id]
    noccsb = [sum(orbsymb_in_d2h[mf.mo_occ[1]>0]==ir) for ir in mol.irrep_id]
    for i, ir in enumerate(mol.irrep_id):
        if (noccsa[i]+noccsb[i]) % 2:
            tot_sym ^= ir
    print(f'Wave-function symmetry = {symm.irrep_id2name(mol.groupname, tot_sym)}')



if __name__ == '__main__':

    # Geometries taken from JCTC 2018, 14, 12, 6240–6252:
    # XYZ coordinates (AU) of  [Cu(H2O)4]2+
    Mol1 = [
    ['Cu', [  0.0000000000000,   0.0000000000000,   0.0000000000000]],
    ['O',  [  0.0000000000000,   3.6666299464596,   0.0000000000000]],
    ['H',  [  1.4666221209110,   4.7748050350325,   0.0000000000000]],
    ['O',  [  3.6666299464596,   0.0000000000000,   0.0000000000000]],
    ['H',  [  4.7748050350325,   1.4666221209110,   0.0000000000000]],
    ['O',  [  0.0000000000000,  -3.6666299464596,   0.0000000000000]],
    ['H',  [ -1.4666221209110,   4.7748050350325,   0.0000000000000]],
    ['H',  [  1.4666221209110,  -4.7748050350325,   0.0000000000000]],
    ['H',  [ -1.4666221209110,  -4.7748050350325,   0.0000000000000]],
    ['O',  [ -3.6666299464596,   0.0000000000000,   0.0000000000000]],
    ['H',  [ -4.7748050350325,   1.4666221209110,   0.0000000000000]],
    ['H',  [  4.7748050350325,  -1.4666221209110,   0.0000000000000]],
    ['H',  [ -4.7748050350325,  -1.4666221209110,   0.0000000000000]],
    ]

    #basis= 'sto-3g' 
    basis= 'ccpvdz'
    #basis= '6-31g'
    #basis= 'aug-cc-pwcvtz-dk.0.nw' 

    # For saving data later
    pre_out= 'output/01-Cu_H2O_4_2+/'+ basis+ '/'  

    print("Computing [Cu(H2O)4]2+ in ", basis, " Basis:")

    mol           = gto.M()
    mol.basis     = basis
    mol.atom      = Mol1
    mol.unit      = 'Bohr'
    mol.verbose   = 3
    mol.nelectron = 67 # 69 = 29 + 4*10  
    mol.spin      = 1
    mol.charge    = 2
    mol.symmetry  = True
    mol.build()
    
    # UHF
    mf = mol.ROHF().newton()
    # Ground state irreducible representation
    mf.irrep_nelec = {'Ag': (11, 10), 'B1g': (3, 3), 'B2g': (2, 2), 'B3g': (2, 2), 'B1u': (4, 4), 'B2u': (6, 6), 'B3u': (6, 6)}

    mf.run()
    mf.stability()
    mf.analyze()
    mfe = mf.e_tot
    print(f'Energy: {mfe}')

    breakpoint()
    mycc  = cc.CCSD(mf)
    freezing = True
    if freezing:
        # Freezing He, Ne, and Ar core in N, Cu, Cl, O 
        # => 26 core electrons, i.e., 13 core orbitals
        core = [i for i in range(13)]
        mycc.set(frozen = core)

    mycc.verbose = 4
    mycc.kernel()

    print(mycc.e_tot)
