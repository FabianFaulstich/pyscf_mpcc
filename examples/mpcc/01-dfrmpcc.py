from pyscf import gto, scf, mp, cc
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
    mol.basis = "cc-pvdz"
    mol.build()

    mf = scf.RHF(mol).density_fit().run()
    mf.threshold = 1e-6

    mycc = cc.CCSD(mf)
    mycc.kernel()

    # Orbital localization    
    
    ncore = 0
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]

    ao_labels = ["O 2p", "O 2s","H 1s"]
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
        np.where(avas_obj.vir_weights > avas_obj.threshold)[0])

    print(act_part)

    c_lo = mocas
    c_lo = mf.mo_coeff

    print ("dimension of active hole", len(act_hole)) 
    print ("dimension of active part", len(act_part)) 

    frag = [[act_hole, act_part]]

    #mymp = DFMP2(mf).run()
    mycc = cc.CCSD(mf).density_fit().run()

    kwargs = {'frag': [[act_hole, act_part]],
                'll_con_tol': 1e-6, 
                'll_max_its': 80,
                'll_kernel_type' : 'unfactorized'
            }

    # FIXME input has been currupted, fix this!
    mympcc = mpcc.RMPCC(mf, 'True', c_lo, **kwargs)

    mympcc.kernel()

    print("Finished MPCC!")

    #NOTE take the correlation energy from teh HL solver

    print(f'CCSD:\n Total energy: {mycc.e_tot} Correlation energ: {mycc.e_corr}')
    print(f'DF-MPCCSD:\n Total energy: {mympcc.lowlevel.e_tot} Correlation energ: {mympcc.lowlevel.e_corr}')
    print(f'Difference:\n Total energy: {float(mympcc.lowlevel.e_tot - mycc.e_tot)} Correlation energ: {mympcc.lowlevel.e_corr - mycc.e_corr}')
    breakpoint()
    # localization, where?
    # a-a, i-a do this in ERIs
    # 

    # NOTE what we want:
    #mympcc.lowlevel set #its, tol, 
    #   "ll method" set this RPA here 


