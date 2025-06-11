from pyscf import gto, scf, cc
from pyscf.mp.dfmp2_native import DFMP2

from pyscf.mcscf import avas
from pyscf.data.elements import chemcore

import numpy as np

from pyscf import mpcc

if __name__ == "__main__":

    mol = gto.Mole()
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.757, 0.587)],
        [1, (0.0, 0.757, 0.587)],
    ]
    mol.basis = "cc-pvtz"
    mol.build()

    mf = scf.RHF(mol).density_fit().run()

    # Orbital localization    


    
    ncore = chemcore(mol)
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]

    breakpoint()

    ao_labels = ["O 2p", "O 2s","O 3p", "O 3s"]
    minao="sto-3g"
    openshell_option = 3
    
    avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
    avas_obj.with_iao = True
    avas_obj.threshold = 1e-7 
    
    _, _, mocas = avas_obj.kernel()
    act_hole = (
        np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
    )

    act_part = (
        np.where(avas_obj.vir_weights > avas_obj.threshold)[0] + mf.mol.nelec[0])

    c_lo = mocas

    print ("dimension of active hole", len(act_hole)) 
    print ("dimension of active part", len(act_part)) 

    frag = [[act_hole, act_part]]


    mymp = DFMP2(mf).run()
    #mycc = cc.CCSD(mf).density_fit().run()

    # No computation
    mympcc = mpcc.MPCC(mf)

    # NOTE setting the following variables explicitly to the default value
    # This can also be passed through mpcc.MPCC(mf, 'arg'= value)
    # mympcc = mpcc.MPCC(mf, ll_con_tol = 1e-6, ll_max_its = 50)
    
    mympcc.lowlevel.ll_max_its = 50
    mympcc.lowlevel.ll_con_tol = 1e-6


    # localization input
    # 
    mympcc.kernel(localization = True, )

    print("Quit MPCC")
    print(mympcc.lowlevel.e_tot)
    breakpoint()
    # localization, where?
    # a-a, i-a do this in ERIs
    # 

    # NOTE what we want:
    #mympcc.lowlevel set #its, tol, 
    #   "ll method" set this RPA here 


