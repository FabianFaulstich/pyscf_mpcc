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
    #basis= '6-31g' 
    basis= 'aug-cc-pwcvtz-dk.0.nw'

    pre_out= 'output/03-CuCl4_2-/'+ basis+ '/' 
    
    print("Computing [CuCl4]2- in ", basis, " Basis:")

    mol           = gto.M()
    mol.basis     = basis
    mol.atom      = Mol1
    mol.unit      = 'bohr'
    mol.verbose   = 3
    mol.nelectron = 99 #97=29 + 4*17 
    mol.spin      = 1
    mol.charge    = -2
    mol.build()

    mf = mol.UHF()
    mf = scf.UHF(mol).run(verbose = 4)
    mfe = mf.e_tot

    mycc  = cc.CCSD(mf)

    freezing = True
    if freezing:
        # Freezing He, Ne, and Ar core in N, Cu, Cl, O 
        # => 58 core electrons, i.e., 26 core orbitals
        core = [i for i in range(26)]
        mycc.set(frozen = core)

    mycc.verbose = 4
    mycc.max_cycle = 100
    mycc.kernel()
    ecc = mycc.e_tot

