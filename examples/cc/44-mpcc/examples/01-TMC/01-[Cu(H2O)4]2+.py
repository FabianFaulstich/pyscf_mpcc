from pyscf import gto, scf, cc
import numpy as np
from numpy import linalg as la
import matplotlib
import matplotlib.pyplot as plt
import copy
import time

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
    #basis= '6-31g'
    basis= 'aug-cc-pwcvtz-dk.0.nw' 

    # For saving data later
    pre_out= 'output/01-Cu_H2O_4_2+/'+ basis+ '/'  

    print("Computing [Cu(H2O)4]2+ in ", basis, " Basis:")

    mol           = gto.M()
    mol.basis     = basis
    mol.atom      = Mol1
    mol.unit      = 'bohr'
    mol.verbose   = 3
    mol.nelectron = 67 # 69 = 29 + 4*10  
    mol.spin      = 1
    mol.charge    = 2
    mol.build()
    
    # UHF
    mf = mol.UHF()
    mf = scf.UHF(mol).run(verbose = 4)
    mfe = mf.e_tot


    mycc  = cc.CCSD(mf)
    freezing = True
    if freezing:
        # Freezing He, Ne, and Ar core in N, Cu, Cl, O 
        # => 26 core electrons, i.e., 13 core orbitals
        core = [i for i in range(13)]
        mycc.set(frozen = core)

    mycc.verbose = 4
    mycc.kernel()
    ecc   = mycc.e_tot

