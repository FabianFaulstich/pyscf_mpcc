from pyscf import gto, scf, mp, cc, lib
import numpy as np
from pyscf.mpcc import mpcc_tools
from pyscf.mpcc.df_eri import ERIs
import matplotlib.pyplot as plt
from pyscf.ao2mo import _ao2mo
from helper_fun import build_molecule
from arg_parse import parse_arg
from pathlib import Path
import argparse
import os

####### CPD experiment #######
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CC2 iteration energies for a molecule and basis")
    parser.add_argument("basis", type=str, help="Basis set, e.g., cc-pvdz")
    parser.add_argument("molecule", type=str, help="Molecule name, e.g., H2O")
    parser.add_argument("--results", type=str, default=None, help="path to result folder")
    args = parser.parse_args()

    mol_name = args.molecule
    basis = args.basis
    basis_to_mol = {
            "cc-pvdz": "DZ",
            "cc-pvtz": "TZ",
            "aug-cc-pvdz": "aVDZ",
            "aug-cc-pvtz": "aug-cc-pvtz"
            }
    Basis = basis_to_mol[basis]

    mol_to_water = {
    "TIP4P-1": 1,
    "TIP4P-2": 2,
    "TIP4P-4": 4,
    "TIP4P-6": 6,
    "TIP4P-8": 8,
    "TIP4P-10": 10
            }
    water = mol_to_water[mol_name]

    if args.results:
        results = Path(os.path.expanduser(args.results)).resolve()
    else:
        results = Path(__file__).parent / "output_data"
    
    base_folder = os.path.join(results,basis, mol_name)
    Lvv_folder = os.path.join(base_folder,"Lvv_cgs")
    
    os.makedirs(Lvv_folder, exist_ok= True)
    species = f"Molecule/{mol_name}.xyz"    
    mol, mol_name = build_molecule(species,basis)

    mf = scf.RHF(mol).density_fit().run()
    mf.threshold = 1e-6
    
    nao = mf.mol.nao
    with_df = mf.with_df
    naux = with_df.get_naoaux()
    Lpq = np.empty((naux,nao,nao))

    for k, eri1 in enumerate(with_df.loop()):
        p1 = 0 
        Lpq = _ao2mo.nr_e2(eri1, np.eye(nao), (0, nao, 0, nao), aosym = 's2', mosym='s1', out = Lpq)
        p0, p1 =p1, p1+ Lpq.shape[0]
        Lpq = Lpq.reshape(p1 - p0, nao, nao)
    print(f"shape of Lpq:{Lpq.shape}")

    eris_obj = ERIs(mf)
    Loo, Lov, Lvo, Lvv = eris_obj.Loo, eris_obj.Lov, eris_obj.Lvo, eris_obj.Lvv
    print("\nAccessed _make_df_eris successfully!")
    print("Loo shape:", Loo.shape)
    print("Lov shape:", Lov.shape)
    print("Lvv shape:", Lvv.shape)
    print("Lvo shape:", Lvo.shape)
    def adjust_factor(factor_matrix, rank_new):
        """Trim or pad factor matrix to new rank."""
        old_rank = factor_matrix.shape[1]
        if rank_new == old_rank:
            return factor_matrix
        elif rank_new < old_rank:
            return factor_matrix[:, :rank_new]
        else:
            pad = mpcc_tools.init_from_pool_generalized(factor_matrix.shape[0], rank_new - old_rank)
            return np.hstack([factor_matrix, pad])

    max_iter = 200
    tol = 1e-4
    cOption = 1
    kOption = 0
    
    results = {
        "Lvv": {"ranks": [], "err_init3": [], "err_initfactors": []}
    }
    # === CPD on Loo ===
    r_oo = int(naux/2)
    weights, factors, _ = mpcc_tools.cp_als(Loo, r_oo, max_iter, tol, init=3,cOption=cOption, kOption=kOption)
    A, O, _ = factors
    Loo_hat = mpcc_tools.reconstruct_cp_tensor(weights, factors)
    rel_error = (np.linalg.norm(Loo - Loo_hat) / np.linalg.norm(Loo))*100
    print(f"[init=3] Loo rank {r_oo}: error = {rel_error:.3e}")
    # save factors for reuse
    init_factors = [A, O]
    # === CPD on Lov ===
    multiples = np.arange(1, 4, 0.5)   
    for mult in multiples:
        r_vv = int(mult * naux) 
        A_adj = adjust_factor(A, r_vv)
        O_adj = adjust_factor(O, r_vv)
        init_factors = [A_adj, O_adj, mpcc_tools.init_from_pool_generalized(Lov.shape[2], r_vv)]
        weights2, factors2, _ = mpcc_tools.cp_als(Lov, r_vv, max_iter, tol, init=init_factors,cOption=cOption, kOption=kOption)
        Lov_hat2 = mpcc_tools.reconstruct_cp_tensor(weights2, factors2)
        rel_err_fact = (np.linalg.norm(Lov - Lov_hat2) / np.linalg.norm(Lov))*100

        print(f" Lov rank {r_vv}: [init=factors]: {rel_err_fact:.3e}")
   ###### Lvv #######

        weights3, factors3, _ = mpcc_tools.cp_als(Lvv, r_vv, max_iter=max_iter, tol=tol, init=3,cOption=cOption, kOption=kOption)
        Lvv_hat = mpcc_tools.reconstruct_cp_tensor(weights3, factors3)
        rel_err_rand = (np.linalg.norm(Lvv - Lvv_hat) / np.linalg.norm(Lvv))*100
        
        # reuse factors
        factors2[0] = weights2 * factors2[0]
        A_adj = adjust_factor(factors2[0],r_vv)
        V_adj = adjust_factor(factors2[2], r_vv)
        init_factors = [A_adj, V_adj, V_adj]
        weights4, factors4, _ = mpcc_tools.cp_als(Lvv, r_vv, max_iter=max_iter, tol=tol, init=init_factors,cOption=cOption, kOption=kOption)
        Lvv_hat2 = mpcc_tools.reconstruct_cp_tensor(weights4, factors4)
        rel_err_fact = (np.linalg.norm(Lvv - Lvv_hat2) / np.linalg.norm(Lvv))* 100

        results["Lvv"]["ranks"].append(r_vv)
        results["Lvv"]["err_init3"].append(rel_err_rand)
        results["Lvv"]["err_initfactors"].append(rel_err_fact)
        print(f" Lvv rank {r_vv}: [init =3]: {rel_err_rand:.3e} [init=factors]: {rel_err_fact:.3e}")

###########Lvv_Plot###########

    # LaTeX-friendly font sizes
    plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 10,
    })

    markers = ["o", "s", "D", "^", "v", "P", "X"]


    plt.figure()
    plt.plot(results["Lvv"]["ranks"], results["Lvv"]["err_init3"], 'o-', label="init=random")
    plt.plot(results["Lvv"]["ranks"], results["Lvv"]["err_initfactors"], 's--', label="init=factor")
    plt.xticks([int(m * naux) for m in multiples],[f"{int(m) if m.is_integer() else m}X" for m in multiples])
    plt.xlabel("Rank r")
    plt.ylabel("Percent Error")
    plt.title("Percent Error vs Cp Rank\n"
                rf"$H2O_{water}$, {Basis}, "
              r"$R_{{\mathrm{oo}}}= X$")

    plt.yscale("log")  # log-scale y-axis
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(Lvv_folder,f"CPD_Error_CP_rank_{mol_name}_{basis}.png"), dpi=300,bbox_inches="tight")
    plt.show()

    print("Experiment complete.")
