from pyscf import gto, scf, mp, cc
from pyscf.mp.dfmp2_native import DFMP2

from pyscf.mcscf import avas
from pyscf.data.elements import chemcore

import numpy as np
import glob
import os
from pyscf import mpcc
from pyscf.mpcc import mpcc_tools as mpt

if __name__ == "__main__":
    
    mOption = 1 
    mol = gto.Mole()
    if mOption == 1:
        mol.atom = [
            [8, (0.0, 0.0, 0.0)],
            [1, (0.0, -0.757, 0.587)],
            [1, (0.0, 0.757, 0.587)],
            ]
    elif mOption == 2:
        xyz_folder = "water_cluster"
        xyz_files = sorted(glob.glob(os.path.join(xyz_folder, "TIP4P-2.xyz")))

        if not xyz_files:
            raise FileNotFoundError(f"No TIP4P-*.xyz files found in {water_cluster}/")

        for xyz_file in xyz_files:
            print(f"\nProcessing: {xyz_file}")
            mol.atom = read_xyz(xyz_file)
    else:
        mol.atom= [
                [6,   (-0.6695, 0.0000, 0.0000)],
                [6, (0.6695,   0.0000,   0.0000)],
                [1, (-1.2335,   0.9238,   0.0000)],
                [1,   (-1.2335,  -0.9238,   0.0000)],
                [1,    (1.2335,   0.9238,   0.0000)],
                [1,    (1.2335,  -0.9238,   0.0000)],
                ]

    mol.basis = "cc-pvdz"
    mol.build()

    mf = scf.RHF(mol).density_fit().run()
    mf.threshold = 1e-6

    # DFMP2 and DFCCSD reference calculations
    mymp = DFMP2(mf).run()
    mycc = cc.CCSD(mf).density_fit().run()

    # Orbital localization    
    ncore = 0
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]

    ao_labels = mpt.get_ao_labels(mol)
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

    c_lo = mocas
    #c_lo = mf.mo_coeff

    print ("dimension of active hole", len(act_hole)) 
    print ("dimension of active part", len(act_part)) 

    frag = [[act_hole, act_part]]

    kwargs = {'frag': [[act_hole, act_part]],
                'll_con_tol': 1e-6, 
                'll_max_its': 80,
                'll_kernel_type' : 'unfactorized',
                'lo_coeff' : c_lo
            }

    rank_control = {
    'rank_reduced': False,
    'rank_opts': {
        'Loo': 0.5,   # rank = 0.5 * naux
        'Lov': 1.5,   # rank = 1.5 * naux
        'Lvv': 3.0    # rank = 3.0 * naux
        }
}

    kwargs = rank_control


    
    mympcc = mpcc.MPCC(mf, **kwargs)
    mympcc.kernel()

    print("Finished MPCC!")

    #NOTE take the correlation energy from teh HL solver

    print(f'CCSD:\n Total energy: {mycc.e_tot} Correlation energ: {mycc.e_corr}')
    print(f'DF-MPCCSD:\n Total energy: {mympcc.lowlevel.e_tot} Correlation energ: {mympcc.lowlevel.e_corr}')
    print(f'Difference:\n Total energy: {float(mympcc.lowlevel.e_tot - mycc.e_tot)} Correlation energ: {mympcc.lowlevel.e_corr - mycc.e_corr}')
    breakpoint()

