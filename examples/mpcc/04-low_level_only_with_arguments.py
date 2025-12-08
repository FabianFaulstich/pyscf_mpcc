import gc
import sys
from pathlib import Path
import os
os.environ['PYSCF_TMPDIR'] = '/home/talha/pyscf_tmp'
import glob
import numpy as np
from pyscf import gto, scf, cc, mp, mpcc
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
    os.makedirs(f"energy_folder_{scan}", exist_ok=True)
    os.makedirs(f"Y_amp_folder_{scan}", exist_ok= True)
    os.makedirs(f"Ω_folder_{scan}", exist_ok = True)
    os.makedirs(f"Foo_folder_{scan}", exist_ok=True)
    os.makedirs(f"Fov_folder_{scan}", exist_ok= True)
    os.makedirs(f"Fvv_folder_{scan}", exist_ok = True)

    if rank_reduced_option:
        suffix = f"CPD_{scan}_rank{rank_value}X"
    else:
        suffix = "DF"

    mf = scf.RHF(mol).density_fit().run()
    mf.mol.max_memory = 20000   
    c_lo = mf.mo_coeff
    
    frag_info = {'frag': [[[0], [0]]]}
    conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
    rank_control = {
            'rank_reduced': rank_reduced_option,
            'rank_opts': {'Loo': 1, 'Lov': 2, 'Lvv': rank_value}
        }
    kwargs = frag_info | conv_info | rank_control
    
    mympcc = mpcc.RMPCC(mf,'True', c_lo, **kwargs)
    
    print('Initializing ...')
    # Initializing the input for low-level solver
    st = time.time()
    mycc = cc.CCSD(mf)
    mycc.max_cycle = 6
    mycc.kernel()
    
    t1_init = mycc.t1.copy()
    mycc.t2 = None
    del mycc
    gc.collect()
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
    np.save(os.path.join(f"Foo_folder_{scan}", f"CC2_Foo_{suffix}.npy"), Foo)
    np.save(os.path.join(f"Fov_folder_{scan}", f"CC2_Fov_{suffix}.npy"), Fov)
    np.save(os.path.join(f"Fvv_folder_{scan}", f"CC2_Fvv_{suffix}.npy"), Fvv)

    np.save(os.path.join(f"Y_amp_folder_{scan}", f"CC2_Y_{suffix}.npy"), Y)
   
    np.save(os.path.join(f"Ω_folder_{scan}", f"CC2_Ω_{suffix}.npy"), Ω)


    np.save(os.path.join(f"energy_folder_{scan}", f"CC2_energy{suffix}.npy"), e_corr)
    

    np.savetxt(os.path.join(energy_folder_Lov_Lvv,f"CC2_iter_energies_{suffix}.txt"),np.array(iter_energies))
    #np.savetxt(os.path.join(energy_folder_Lov,f"CC2_iter_energies_{suffix}.txt"),np.array(iter_energies))
    #np.savetxt(os.path.join(energy_folder_Lvv,f"CC2_iter_energies_{suffix}.txt"),np.array(iter_energies))

    print("\nSaved:")
    print(f"  Y amplitudes → {Y_amp_folder_Lov_Lvv}/CC2_Y_{suffix}.npy")
    print(f"  Ω → {Ω_folder_Lov_Lvv}/CC2_Ω_{suffix}.npy")
    print(f"  Energy and iter_energy       → {energy_folder_Lov_Lvv}/CC2_energy_{suffix}.npy")
    print("\nDone.\n")
