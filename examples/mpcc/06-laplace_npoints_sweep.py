from pathlib import Path

import numpy as np

from pyscf import cc, gto, scf
from pyscf.mcscf import avas
from pyscf.mp.dfmp2_native import DFMP2

from pyscf import mpcc
from pyscf.mpcc import mpcc_tools as mpt


def build_water(basis = "ccpvtz"):
    mol = gto.Mole()
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.757, 0.587)],
        [1, (0.0, 0.757, 0.587)],
    ]
    mol.basis = basis
    mol.build()

    mf = scf.RHF(mol).density_fit().run()
    mf.threshold = 1e-6
    return mol, mf


def build_avas_space(mf, minao = "sto-3g"):
    mol = mf.mol
    ao_labels = mpt.get_ao_labels(mol)
    avas_obj = avas.AVAS(
        mf,
        ao_labels,
        minao=minao,
        openshell_option=3,
    )
    avas_obj.with_iao = True
    avas_obj.threshold = 1e-7

    _, _, c_lo = avas_obj.kernel()
    act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
    act_part = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]

    return c_lo, [[act_hole, act_part]]


def run_mpcc(mf, c_lo, frags, kwargs, macro_tol=1e-8):
    mympcc = mpcc.MPCC(mf, **kwargs, frag=frags, lo_coeff=c_lo)
    mympcc.kernel(tol=macro_tol)
    return mympcc.lowlevel.e_tot, mympcc.lowlevel.e_corr


if __name__ == "__main__":
    mol, mf = build_water()
    c_lo, frags = build_avas_space(mf)
    laplace_root = Path(__file__).resolve().parents[2] / "external" / "laplace-minimax"

    print("dimension of active hole", len(frags[0][0]))
    print("dimension of active part", len(frags[0][1]))
    print("Laplace quadrature root:", laplace_root)

    mymp = DFMP2(mf).run()
    mycc = cc.CCSD(mf).density_fit().run()

    common = {
        "ll_con_tol": 1e-8,
        "ll_max_its": 50,
        "ll_chol_tol": 1e-8,
    }

    print("\n=== Unfactorized T1_transform reference ===")
    e_tot_ref, e_corr_ref = run_mpcc(
        mf,
        c_lo,
        frags,
        {
            **common,
            "ll_kernel_type": "unfactorized",
            "ll_method": "T1_transform",
        },
    )

    print("\n=== Factorized Laplace n_L sweep ===")
    results = []
    for nlap in (4, 6, 8, 10, 12):
        print(f"\n--- n_L = {nlap} ---")
        e_tot, e_corr = run_mpcc(
            mf,
            c_lo,
            frags,
            {
                **common,
                "ll_kernel_type": "sylvester_laplace_factorized",
                "ll_laplace_root": laplace_root,
                "ll_laplace_npoints": nlap,
            },
        )
        results.append((nlap, e_tot, e_corr))

    print("\n=== Summary ===")
    print(f"CCSD total      {mycc.e_tot: .16f}")
    print(f"CCSD corr       {mycc.e_corr: .16f}")
    print(f"Unfact total    {e_tot_ref: .16f}")
    print(f"Unfact corr     {e_corr_ref: .16f}")
    print(f"Unfact dE(CCSD) {e_tot_ref - mycc.e_tot: .16f}")
    print()
    print(" n_L        E_tot Laplace        E_corr Laplace       dE vs unfact       dE vs CCSD")
    for nlap, e_tot, e_corr in results:
        print(
            f"{nlap:4d}  "
            f"{e_tot: .16f}  "
            f"{e_corr: .16f}  "
            f"{e_tot - e_tot_ref: .16f}  "
            f"{e_tot - mycc.e_tot: .16f}"
        )
