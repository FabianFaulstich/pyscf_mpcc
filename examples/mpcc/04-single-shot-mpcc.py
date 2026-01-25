import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pyscf import gto, scf, cc
from pyscf import mpcc

from pyscf.mpcc import mpcc_tools as mpt

from pyscf.mcscf import avas

import numpy as np
import time

from helper_fun import build_molecule, capture_output,parse_iteration_energies
from arg_parse import parse_arg
import os
os.environ['PYSCF_TMPDIR'] = '/home/talha/pyscf_tmp'

from pathlib import Path



if __name__ == "__main__":
    species, basis, rank_reduced_option ,rank_value, result_path,scan= parse_arg() 
    mol, mol_name = build_molecule(species,basis)
    print(f"\n=== Running calculation for {mol_name} ===")
    #results = "/Users/talha/Documents/Nov_11_MPCC/All_output_data"
    if result_path:
        results = Path(os.path.expanduser(result_path)).resolve()
    else:
        results = Path(__file__).parent / "output_data"
    
    base_folder = os.path.join(results,basis, mol_name)
    energy_folder_CC2 = os.path.join(base_folder,f"energies_CC2")
    energy_folder_CCSD = os.path.join(base_folder,f"energies_CCSD")
    energy_folder_CC2_CCSD = os.path.join(base_folder, f"energies_CC2_CCSD")
    energy_folder_MPCC = os.path.join(base_folder,f"energies_MPCC")
    Ω_folder_Lov_Lvv = os.path.join(base_folder,f"Ω_{scan}")
    energy_folder_CC_SD = os.path.join(base_folder,f"energies_CC_SD")
 
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(energy_folder_CC2, exist_ok=True)
    os.makedirs(energy_folder_CCSD, exist_ok=True)
    os.makedirs(energy_folder_CC2_CCSD, exist_ok=True)
    os.makedirs(energy_folder_MPCC, exist_ok=True)
    os.makedirs(Ω_folder_Lov_Lvv, exist_ok=True)
    os.makedirs(energy_folder_CC_SD, exist_ok=True)

    rank_Loo = 1 if "fix_Loo_1" in scan else 0.5
    print(f"value of Loo rank: {rank_Loo}")

    rank_Lov = 1.5 if basis == "cc-pvdz" else 2 
    rank_Lvv = 2.5
    scan_str = str(scan).replace("_fix_Loo_1", "")
    
    
    if scan in ("Lov", "Lov_fix_Loo_1"):
        rank_Lov = rank_value
        rank_Lvv = 2.5
    elif scan in ("Lvv", "Lvv_fix_Loo_1"):
        rank_Lvv = rank_value
    elif scan in ("Lov_Lvv","Lov_Lvv_fix_Loo_1"):
        rank_Lov = rank_value
        rank_Lvv = rank_value

    if rank_reduced_option:
        suffix = f"CPD_{scan_str}_rank{rank_value}X"
    else:
        suffix = "DF"


    mf = scf.RHF(mol).density_fit().run()
    
    print("\n=== Running DF-CCSD reference ===")

    #cc_sd = cc.CCSD(mf)
    
    # Optional but recommended for clean comparison
    #cc_sd.conv_tol = 1e-6
    #cc_sd.max_cycle = 100
    
    #t0 = time.time()
    #e_ccsd, t1_ccsd, t2_ccsd = cc_sd.kernel()
    #t1_time = time.time() - t0
    #np.savetxt(os.path.join(energy_folder_CC_SD,"DF_CCSD_energy.txt"),np.array([e_ccsd]))
 
    #print(f"DF-CCSD correlation energy: {e_ccsd:.10f}")
    #print(f"DF-CCSD total energy: {mf.e_tot + e_ccsd:.10f}")
    #print(f"DF-CCSD wall time: {t1_time:.2f} s")
    
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
                'lo_coeff'      : c_lo,
                'rank_reduced': rank_reduced_option,
                'rank_opts': {'Loo': rank_Loo, 'Lov': rank_Lov, 'Lvv': rank_Lvv}
              }

    mympcc = mpcc.MPCC(mf, **kwargs)
 
    # Initializaing
    t1, t2 = mympcc.lowlevel.init_amps()

    #num_its = 2
    energy_tol = 1e-6
    max_macro_its = 20

    prev_mpcc_ene = None
    i = 0
    mpcc_macro_energies = []
    while i < max_macro_its:
        i += 1
    #for i in range(num_its):

        # run low-level solver
        output, result = capture_output(mympcc.lowlevel.kernel, t1, t2)
        t1, t2 = result
        print(f"output:{output}")
        energy_pattern = r"It\s+\d+;\s+correlation energy\s+([-+]?\d+\.\d+e[-+]\d+)"
        iter_energies, n_iter = parse_iteration_energies(output,energy_pattern)
        if i == 2:
            np.savetxt(os.path.join(energy_folder_CC2,f"CC2_iter_energy_{suffix}_{i}.txt"),np.array(iter_energies))
        Xoo, Xvo, X = mympcc.lowlevel.get_X(t1)
        Foo, Fvv, Fov = mympcc.lowlevel.get_F(t1, X, Xoo, Xvo)
        
        Ω = mympcc.lowlevel.get_Ω_slow(X, Xvo, Foo, Fvv, Fov, t1, t2)
        omega_n = np.linalg.norm(Ω)

        if i == 2:
            np.save(os.path.join(Ω_folder_Lov_Lvv, f"CC2_Ω_{suffix}_{i}.npy"), Ω)

        # runnning high-lelvel solver
        t1_act = []
        t2_act = []

        mympcc.screened.frag = mympcc.frags[0]
        mympcc.highlevel.frag = mympcc.frags[0]

        imds = mympcc.screened.kernel(t1, t2)
        output_hl, result_hl = capture_output(mympcc.highlevel.kernel, imds, t1, t2)

        t1_act_tmp, t2_act_tmp = result_hl
        ccsd_energy_pattern = r"CCSD correlation energy:\s+([-+]?\d+\.\d+)"
        ccsd_iter_energies, n_ccsd_iter = parse_iteration_energies(output_hl, ccsd_energy_pattern)
        if i == 2:
            np.savetxt(os.path.join(energy_folder_CCSD, f"CCSD_iter_energy_{suffix}_{i}.txt"),np.array(ccsd_iter_energies))
        
        t1_act.append(t1_act_tmp)    
        t2_act.append(t2_act_tmp) 

        act_hole = mympcc.frags[0][0]
        act_particle = mympcc.frags[0][1]

        t1[np.ix_(act_hole, act_particle)] = t1_act[0]
        t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[0]
        mpcc_ene = mympcc.highlevel.get_cc_energy(t1, t2, t1_act, t2_act)
        mpcc_macro_energies.append(mpcc_ene)

        print(f'DFMPCC It: {i}; correlation energy: {mpcc_ene:.10f}')

        if prev_mpcc_ene is not None:
            delta_e = abs(mpcc_ene - prev_mpcc_ene)
            print(f'  ΔE = {delta_e:.3e}')

            if delta_e < energy_tol:
                print(f'  MPCC converged after {i} macro iterations')
                break

        prev_mpcc_ene = mpcc_ene
    mpcc_macro_energies = np.array(mpcc_macro_energies)

    np.savetxt(os.path.join(energy_folder_MPCC, f"MPCC_iter_energy_{suffix}.txt"),mpcc_macro_energies)
