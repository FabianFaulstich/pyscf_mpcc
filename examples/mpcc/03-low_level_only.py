from pyscf import gto, scf, cc
from pyscf import mpcc

from pyscf.mpcc import mpcc_tools as mpt

from pyscf.mcscf import avas

import numpy as np
import time

if __name__ == "__main__":

    mol = gto.Mole()
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.757, 0.587)],
        [1, (0.0, 0.757, 0.587)],
    ]
    mol.basis = "cc-pvdz"
    mol.build()
    mf = scf.RHF(mol).density_fit().run()

    # Generating LO basis 
    ao_labels = mpt.get_ao_labels(mol)
    minao="sto-3g"
    openshell_option = 3

    ncore = 0
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]

    avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
    avas_obj.with_iao = True
    avas_obj.threshold = 1e-7 
    
    _, _, c_lo = avas_obj.kernel()

    # Generating empty fragment infor
    frag_info = {'frag': [[[0], [0]]]}
    conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
    kwargs = frag_info|conv_info  #union operation

    mympcc = mpcc.RMPCC(mf, 'True', c_lo, **kwargs)
  
    print('Initializing ...')
    # Initializing the input for low-level solver
    st = time.time()
    mycc = cc.CCSD(mf)
    mycc.max_cycle = 6
    mycc.kernel()
    _, _, Y = mympcc.lowlevel.init_amps()
    print(f'Done! Elapsed time: {time.time() - st} sec')


    # Running low-level solver
    print('Starting Low-Level Solver')
    t1, t2, Y = mympcc.lowlevel.kernel(mycc.t1, [0], Y)
    print('Finished Low-Level solver!')


