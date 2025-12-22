from pyscf import gto, scf, cc
from pyscf import mpcc

from pyscf.mpcc import mpcc_tools as mpt

from pyscf.mcscf import avas

import numpy as np
import time

if __name__ == "__main__":

    mol = gto.Mole()

    mol.atom = [
            [6	,(0	        ,0	    ,0.5891	)],
            [6	,(0	        ,1.2676	,-0.2605)],
            [6	,(0	        ,-1.2676,-0.2605)],
            [1	,(0.8749    ,0	    ,1.243	)],
            [1	,(-0.8749   ,0	    ,1.243	)],
            [1	,(0	        ,2.1642	,0.3602	)],
            [1	,(0	        ,-2.1642,0.3602	)],
            [1	,(0.8811	,1.3045	,-0.9037)],
            [1	,(-0.8811	,1.3045	,-0.9037)],
            [1	,(-0.8811	,-1.3045,-0.9037)],
            [1	,(0.8811	,-1.3045,-0.9037)]
    ]

    mol.basis = "cc-pvtz"
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

    act_hole = (
        np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
    )

    act_part = (
        np.where(avas_obj.vir_weights > avas_obj.threshold)[0])

    kwargs = {'frag'            : [[act_hole, act_part]],
                'll_con_tol'    : 1e-6, 
                'll_max_its'    : 80,
                'll_kernel_type': 'unfactorized',
                'lo_coeff'      : c_lo
            }

    mympcc = mpcc.MPCC(mf, **kwargs)
 
    # Initializaing
    t1, t2 = mympcc.lowlevel.init_amps()

    num_its = 1
    for i in range(num_its):
        # run low-level solver
        t1, t2 = mympcc.lowlevel.kernel(t1, t2)
       
        # runnning high-lelvel solver
        t1_act = []
        t2_act = []

        mympcc.screened.frag = mympcc.frags[0]
        mympcc.highlevel.frag = mympcc.frags[0]

        imds = mympcc.screened.kernel(t1, t2)
        t1_act_tmp, t2_act_tmp = mympcc.highlevel.kernel(imds, t1, t2)

        t1_act.append(t1_act_tmp)    
        t2_act.append(t2_act_tmp) 

        act_hole = mympcc.frags[0][0]
        act_particle = mympcc.frags[0][1]

        t1[np.ix_(act_hole, act_particle)] = t1_act[0]
        t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[0]

        mpcc_ene = mympcc.highlevel.get_cc_energy(t1, t2, t1_act, t2_act)
        print(f'DFMPCC It: {i}; correlation energy: {mpcc_ene}')

