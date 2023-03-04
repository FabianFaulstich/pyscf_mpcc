'''
    Vanilla version of mp2-ccsd
'''

import numpy as np
from pyscf import gto, scf, mp, cc


def get_t_int(t2, eps = 0.9):

    max_val = np.max(np.abs(t2))
    idx = np.where(np.abs(t2)> eps * max_val)
    idx = np.asarray(idx)
    return [idx[:,i] for i in range(len(idx[0,:]))]

def ex_h2(basis):
    
    
    mol = gto.M()
    mol.basis = basis
    mol.atom  = [['H', [0,0,0]] , ['H', [1,0,0]]]
    mol.unit  = 'bohr'
    mol.verbose = 3
    mol.build()


    mf = mol.RHF().run()
    mymp = mp.MP2(mf).run()
    
    t_int = get_t_int(mymp.t2)
    breakpoint()

if __name__ == "__main__":


    # Run example H2
    ex_h2('ccpvtz')

