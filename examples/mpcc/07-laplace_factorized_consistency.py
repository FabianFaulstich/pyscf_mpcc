from pathlib import Path

import numpy as np

from pyscf import gto, scf
from pyscf.mcscf import avas

from pyscf import mpcc
from pyscf.mpcc import mpcc_tools as mpt


def build_water(basis = "cc-pvtz"):
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
    e_tot = mympcc.kernel(tol=macro_tol)
    return e_tot, e_tot - mf.e_tot


if __name__ == "__main__":
    mol, mf = build_water()
    c_lo, frags = build_avas_space(mf)
    laplace_root = Path(__file__).resolve().parents[2] / "external" / "laplace-minimax"
    nlap = 16

    print("dimension of active hole", len(frags[0][0]))
    print("dimension of active part", len(frags[0][1]))
    print("Laplace quadrature root:", laplace_root)
    print("Laplace points:", nlap)

    common = {
        "ll_con_tol": 1e-8,
        "ll_max_its": 50,
        "ll_chol_tol": 1e-8,
        "ll_laplace_root": laplace_root,
        "ll_laplace_npoints": nlap,
    }

    print("\n=== Dense/materialized Laplace ===")
    e_tot_dense, e_corr_dense = run_mpcc(
        mf,
        c_lo,
        frags,
        {
            **common,
            "ll_kernel_type": "unfactorized",
            "ll_method": "sylvester_laplace",
        },
    )

    print("\n=== Factorized memory-optimized Laplace ===")
    e_tot_factorized, e_corr_factorized = run_mpcc(
        mf,
        c_lo,
        frags,
        {
            **common,
            "ll_kernel_type": "sylvester_laplace_factorized",
        },
    )

    print("\n=== Summary ===")
    print(f"Dense total       {e_tot_dense: .16f}")
    print(f"Dense corr        {e_corr_dense: .16f}")
    print(f"Factorized total  {e_tot_factorized: .16f}")
    print(f"Factorized corr   {e_corr_factorized: .16f}")
    print(f"Delta total       {e_tot_factorized - e_tot_dense: .16e}")
    print(f"Delta corr        {e_corr_factorized - e_corr_dense: .16e}")
