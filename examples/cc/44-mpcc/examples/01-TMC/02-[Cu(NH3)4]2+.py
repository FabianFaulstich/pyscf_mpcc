from pyscf import gto, scf, cc
import numpy as np
from numpy import linalg as la
import matplotlib
import matplotlib.pyplot as plt
import copy
import time

if __name__ == '__main__':

    # Geometries taken from JCTC 2018, 14, 12, 6240–6252:
    # XYZ coordinates (AU) of  [Cu(NH3)4]2+
    Mol1 = [
    ['Cu', [  0.000000000000,    0.000000000000,    0.0000000000000]],
    ['N',  [ -2.695137496500,   -2.695137496500,    0.0000000000000]],
    ['N',  [  2.695137496500,    2.695137496500,    0.0000000000000]],
    ['N',  [ -2.695137496500,    2.695137496500,    0.0000000000000]],
    ['N',  [  2.695137496500,   -2.695137496500,    0.0000000000000]],
    ['H',  [ -3.251701244000,   -3.251701244000,   -1.7594000000000]],
    ['H',  [  3.251701244000,    3.251701244000,   -1.7594000000000]],
    ['H',  [ -3.251701244000,    3.251701244000,    1.7594000000000]],
    ['H',  [  3.251701244000,   -3.251701244000,    1.7594000000000]],
    ['H',  [ -4.329331978500,   -2.174070509400,    0.8793000000000]],
    ['H',  [  4.329331978500,    2.174070509400,    0.8793000000000]],
    ['H',  [ -4.329331978500,    2.174070509400,   -0.8793000000000]],
    ['H',  [  4.329331978500,   -2.174070509400,   -0.8793000000000]],
    ['H',  [ -2.174070509400,   -4.329331978500,    0.8793000000000]],
    ['H',  [  2.174070509400,    4.329331978500,    0.8793000000000]],
    ['H',  [ -2.174070509400,    4.329331978500,   -0.8793000000000]],
    ['H',  [  2.174070509400,   -4.329331978500,   -0.8793000000000]]
    ]

    #basis= 'sto-3g' 
    #basis= '6-31g' 
    basis= 'aug-cc-pwcvtz-dk.0.nw'

    pre_out= 'output/02-Cu_NH3_4_2+/'+ basis+ '/' 

    print("Computing [Cu(NH3)4]2+ in ", basis, " Basis:")

    mol           = gto.M()
    mol.basis     = basis
    mol.atom      = Mol1
    mol.unit      = 'bohr'
    mol.verbose   = 3
    mol.nelectron = 67 # 69 = 29 + 4*10  
    mol.spin      = 1
    mol.charge    = 2
    mol.build()

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

