from pathlib import Path

import numpy as np

from pyscf import cc, gto, scf
from pyscf.mcscf import avas
from pyscf.mp.dfmp2_native import DFMP2

from pyscf import mpcc
from pyscf.mpcc import mpcc_tools as mpt


def build_water(basis="cc-pvtz"):
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


def build_avas_space(mf, minao="sto-3g"):
    mol = mf.mol
    avas_obj = avas.AVAS(
        mf,
        mpt.get_ao_labels(mol),
        minao=minao,
        openshell_option=3,
    )
    avas_obj.with_iao = True
    avas_obj.threshold = 1e-7

    _, _, c_lo = avas_obj.kernel()
    act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
    act_part = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]

    return c_lo, [[act_hole, act_part]]


def run_mpcc(label, mf, c_lo, frags, kwargs, macro_tol=1e-8):
    print(f"\n=== {label} ===")
    mympcc = mpcc.MPCC(mf, **kwargs, frag=frags, lo_coeff=c_lo)
    e_tot = mympcc.kernel(tol=macro_tol)
    e_corr = e_tot - mf.e_tot
    return {
        "label": label,
        "e_tot": e_tot,
        "e_corr": e_corr,
        "lowlevel_e_tot": mympcc.lowlevel.e_tot,
        "lowlevel_e_corr": mympcc.lowlevel.e_corr,
    }


def print_summary(results, e_ccsd_tot, e_ccsd_corr, e_mp2_tot, e_mp2_corr):
    print("\n=== Summary ===")
    print(f"DF-MP2 total      {e_mp2_tot: .16f}")
    print(f"DF-MP2 corr       {e_mp2_corr: .16f}")
    print(f"DF-CCSD total     {e_ccsd_tot: .16f}")
    print(f"DF-CCSD corr      {e_ccsd_corr: .16f}")
    print()
    print("Kernel                       E_tot                 E_corr                dE vs T1")
    ref = results[0]
    for result in results:
        print(
            f"{result['label']:<24s} "
            f"{result['e_tot']: .16f}  "
            f"{result['e_corr']: .16f}  "
            f"{result['e_tot'] - ref['e_tot']: .16e}"
        )

    dense = results[1]
    factorized = results[2]
    print()
    print(f"Dense Laplace - T1 total        {dense['e_tot'] - ref['e_tot']: .16e}")
    print(f"Dense Laplace - T1 corr         {dense['e_corr'] - ref['e_corr']: .16e}")
    print(f"Factorized Laplace - T1 total   {factorized['e_tot'] - ref['e_tot']: .16e}")
    print(f"Factorized Laplace - T1 corr    {factorized['e_corr'] - ref['e_corr']: .16e}")
    print(f"Factorized - dense total        {factorized['e_tot'] - dense['e_tot']: .16e}")
    print(f"Factorized - dense corr         {factorized['e_corr'] - dense['e_corr']: .16e}")


if __name__ == "__main__":
    mol, mf = build_water()
    c_lo, frags = build_avas_space(mf)
    laplace_root = Path(__file__).resolve().parents[2] / "external" / "laplace-minimax"
    nlap = 16

    print("dimension of active hole", len(frags[0][0]))
    print("dimension of active part", len(frags[0][1]))
    print("Laplace quadrature root:", laplace_root)
    print("Laplace points:", nlap)

    mymp = DFMP2(mf).run()
    mycc = cc.CCSD(mf).density_fit().run()

    common = {
        "ll_con_tol": 1e-8,
        "ll_max_its": 50,
        "ll_chol_tol": 1e-8,
    }
    laplace_common = {
        **common,
        "ll_laplace_root": laplace_root,
        "ll_laplace_npoints": nlap,
    }

    results = [
        run_mpcc(
            "T1 transform",
            mf,
            c_lo,
            frags,
            {
                **common,
                "ll_kernel_type": "unfactorized",
                "ll_method": "T1_transform",
            },
        ),
        run_mpcc(
            "Dense Laplace",
            mf,
            c_lo,
            frags,
            {
                **laplace_common,
                "ll_kernel_type": "unfactorized",
                "ll_method": "sylvester_laplace",
            },
        ),
        run_mpcc(
            "Factorized Laplace",
            mf,
            c_lo,
            frags,
            {
                **laplace_common,
                "ll_kernel_type": "sylvester_laplace_factorized",
            },
        ),
    ]

    print_summary(results, mycc.e_tot, mycc.e_corr, mymp.e_tot, mymp.e_corr)
