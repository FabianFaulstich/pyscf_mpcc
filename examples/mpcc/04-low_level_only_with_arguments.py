import gc
import sys
from pyscf.mpcc.df_eri import ERIs
from pathlib import Path
import os
os.environ['PYSCF_TMPDIR'] = '/home/talha/pyscf_tmp'
import glob
import numpy as np
from pyscf import gto, scf, cc, mp, mpcc,lib
import time
from helper_fun import build_molecule, capture_output,parse_iteration_energies
from arg_parse import parse_arg
#from memory_profiler import memory_usage

if __name__ == "__main__":

    species, basis, rank_reduced_option ,rank_value, result_path,scan= parse_arg() 
    mol, mol_name = build_molecule(species,basis)
    print(f"\n=== Running calculation for {mol_name} ===")
    #results = "/Users/talha/Documents/Nov_11_MPCC/All_output_data"
    if result_path:
        results = Path(os.path.expanduser(result_path)).resolve()
    else:
        results = Path(__file__).parent / "output_data"
    # Create folder
    base_folder = os.path.join(results,basis, mol_name)
    
    energy_folder_Lov_Lvv = os.path.join(base_folder,f"energies_{scan}")
    Y_amp_folder_Lov_Lvv = os.path.join(base_folder,f"Y_amp_{scan}")
    Ω_folder_Lov_Lvv = os.path.join(base_folder,f"Ω_{scan}")
    
    Foo_folder_Lov_Lvv = os.path.join(base_folder,f"Foo_{scan}")
    Fov_folder_Lov_Lvv = os.path.join(base_folder,f"Fov_{scan}")
    Fvv_folder_Lov_Lvv = os.path.join(base_folder,f"Fvv_{scan}")

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(energy_folder_Lov_Lvv, exist_ok=True)
    os.makedirs(Y_amp_folder_Lov_Lvv, exist_ok= True)
    os.makedirs(Ω_folder_Lov_Lvv, exist_ok = True)
    os.makedirs(Foo_folder_Lov_Lvv, exist_ok=True)
    os.makedirs(Fov_folder_Lov_Lvv, exist_ok= True)
    os.makedirs(Fvv_folder_Lov_Lvv, exist_ok = True)
   
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
    mf.mol.max_memory = 20000   
    c_lo = mf.mo_coeff
    
    frag_info = {'frag': [[[0], [0]]]}
    conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
    rank_control = {
            'rank_reduced': rank_reduced_option,
            'rank_opts': {'Loo': rank_Loo, 'Lov': rank_Lov, 'Lvv': rank_Lvv}
        }
    kwargs = frag_info | conv_info | rank_control
    
    mympcc = mpcc.RMPCC(mf,'True', c_lo, **kwargs)
    
    print('Initializing ...')
    # Initializing the input for low-level solver
    st = time.time()
    mycc = cc.CCSD(mf)
    mycc.max_cycle = 50
    mycc.kernel()
   
    #eris_obj = ERIs(mf)
    #Lov = eris_obj.Lov
    #dD = eris_obj.dD
    #print(f"size of dD:{dD.shape}")
    t1_init = mycc.t1.copy()
    #t2_init = mycc.t2.copy()
    del mycc
    gc.collect()
    #print(f"shape of t2:{t2_init.shape}")
    #Y = lib.einsum("ijab,Ljb->Lia", t2_init, Lov)
    #Y = Y.transpose(0,2,1)[:, None, :, :] * dD.transpose(2, 1, 0)[None, :, :, :]
    #print(f"shape of Y:{Y.shape}") 
    _, _, Y = mympcc.lowlevel.init_amps()
    print(f'Done! Elapsed time: {time.time() - st} sec')

    Xoo,Xvo,X = mympcc.lowlevel.get_X(t1_init)
    Foo,Fvv,Fov = mympcc.lowlevel.get_F(t1_init,X,Xoo,Xvo)
    Ω = mympcc.lowlevel.get_Ω(X, Xvo, Foo, Fvv, Fov, t1_init, Y)

    # Running low-level solver
    print('Starting Low-Level Solver')
    output, result = capture_output(mympcc.lowlevel.kernel,t1_init, [0], Y)
    t1,t2,Y = result
 
    e_corr = mympcc.lowlevel.get_energy(t1, t2) 
    print('Finished Low-Level solver!')
    
    # Save full printed MPCC output
    full_output_file = os.path.join(energy_folder_Lov_Lvv,f"CC2_full_output_{suffix}.txt")
    with open(full_output_file, "w") as f:
        f.write(output)


    energy_pattern = r"CC2 correlation energy:\s*([0-9.eE+-]+)"
    iter_energies, n_iter = parse_iteration_energies(output,energy_pattern)

    # Save MPCC data
    np.save(os.path.join(Foo_folder_Lov_Lvv, f"CC2_Foo_{suffix}.npy"), Foo)
    np.save(os.path.join(Fov_folder_Lov_Lvv, f"CC2_Fov_{suffix}.npy"), Fov)
    np.save(os.path.join(Fvv_folder_Lov_Lvv, f"CC2_Fvv_{suffix}.npy"), Fvv)

    np.save(os.path.join(Y_amp_folder_Lov_Lvv, f"CC2_Y_{suffix}.npy"), Y)
   
    np.save(os.path.join(Ω_folder_Lov_Lvv, f"CC2_Ω_{suffix}.npy"), Ω)


    np.save(os.path.join(energy_folder_Lov_Lvv, f"CC2_energy{suffix}.npy"), e_corr)
    

    np.savetxt(os.path.join(energy_folder_Lov_Lvv,f"CC2_iter_energies_{suffix}.txt"),np.array(iter_energies))
    #np.savetxt(os.path.join(energy_folder_Lov,f"CC2_iter_energies_{suffix}.txt"),np.array(iter_energies))
    #np.savetxt(os.path.join(energy_folder_Lvv,f"CC2_iter_energies_{suffix}.txt"),np.array(iter_energies))

    print("\nSaved:")
    print(f"  Y amplitudes → {Y_amp_folder_Lov_Lvv}/CC2_Y_{suffix}.npy")
    #print(f"  Ω → {Ω_folder_Lov_Lvv}/CC2_Ω_{suffix}.npy")
    print(f"  Energy and iter_energy       → {energy_folder_Lov_Lvv}/CC2_energy_{suffix}.npy")
    print("\nDone.\n")
