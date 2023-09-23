from h5py._hl.dataset import local
import numpy as np
from numpy.testing._private.utils import verbose
from pyscf import ao2mo, lib, gto, scf, mp, cc, lo, fci
from pyscf.gto.mole import ao_labels
from pyscf.tools import ring
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from pyscf.tools import cubegen, molden
from pyscf.mcscf import avas
from pyscf import lib


def write_molden(mol, mf, name):

    if name is None:
        print("Did not provide name for molden file")
        exit()

    if isinstance(mf, scf.uhf.UHF):
        with open(name + "_mo.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(
                mol, f1, mf.mo_coeff[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0]
            )
            molden.header(mol, f1)
            molden.orbital_coeff(
                mol, f1, mf.mo_coeff[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1]
            )

            print("Wrote molden file: \n", f1)

        with open(name + f"_{localization}.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(
                mol, f1, c_lo[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0]
            )

            molden.header(mol, f1)
            molden.orbital_coeff(
                mol, f1, c_lo[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1]
            )
            print("Wrote molden file: \n", f1)
    else:
        with open(name + "_mo.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
            print("Wrote molden file: \n", f1)

        with open(name + f"_{localization}.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, c_lo, ene=mf.mo_energy, occ=mf.mo_occ)
            print("Wrote molden file: \n", f1)


def get_localized_orbs(
    mol,
    mf,
    ao_labels,
    minao="sto-3g",
    openshell_option=4,
    molden_bool=False,
    name=None,
):

    avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
    avas_obj.with_iao = True
    _, _, mocas = avas_obj.kernel()

    act_hole = (
        np.where(avas_obj.occ_weights[0] > avas_obj.threshold)[0],
        np.where(avas_obj.occ_weights[1] > avas_obj.threshold)[0],
    )
    act_part = (
        np.where(avas_obj.vir_weights[0] > avas_obj.threshold)[0] + mf.mol.nelec[0],
        np.where(avas_obj.vir_weights[1] > avas_obj.threshold)[0] + mf.mol.nelec[1],
    )

    c_lo = mocas

    if molden_bool:
        write_molden(mol, mf, name)

    return c_lo, act_hole, act_part


def fragmented_mpcc_unrestricted(frags, mycc, mp_t2, mp_t1, idx_s, idx_d):

    _, t2ab, _ = mp_t2
    nocca, noccb, _, _ = t2ab.shape

    t1a, t1b = mp_t1
    t2 = np.copy(mp_t2)
    t1 = t1a, t1b

    for frag in frags:
        act_hole = [frag[0][0], frag[0][1]]
        act_particle = [frag[1][0] - nocca, frag[1][1] - noccb]
        res = mycc.kernel(act_hole, act_particle, idx_s, idx_d, t1=t1, t2=t2, oo_mp2 = False)
        print("Exact Exact localized CCSD in MPCCSD (4,2) :", res[0])

        t2 = mycc.t2
        t1 = mycc.t1

    return res[0], t1, t2


def get_N2(basis, bond_length, spin, spin_restricted, dm0=None):

    mol = gto.M()
    mol.basis = basis
    mol.atom = [["N", [0, 0, 0]], ["N", [bond_length, 0, 0]]]
    mol.verbose = 3
    mol.unit = "angstrom"
    mol.spin = spin
    mol.build()

    if spin_restricted:
        if dm0 is not None:
            mf = mol.ROHF().run(dm0=dm0)
        else:
            mf = mol.ROHF().run()
    else:
        mf = mol.UHF().newton()
        if dm0 is not None:
            mf = mf.run(dm0)
        else:
            mf = mf.run()

        mo1 = mf.stability()[0]

        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1 = mf.stability()[0]

        mf = mf.newton().run(mo1, mf.mo_occ)
        mf.stability()

        mymp = mp.UMP2(mf).run()
        mycc = cc.UCCSD(mf).run()
        mycc.diis = 12
        mycc.max_cycle = 80


    return mol, mf, mf.e_tot, mymp.e_corr, mycc.e_corr


def oo_mp2(mf, c_lo, t1, t2, spin_restricted):

    if spin_restricted:
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mymp.async_io = True

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, t1=t1, t2=t2, verbose=0)

    else:
        _, sz = mf.spin_square()
        print("spin multiplicity", sz)
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()

        # Running localized MP2
        eris = mp.ump2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, t1=t1, t2=t2, verbose=0)

    return mp_lo


if __name__ == "__main__":

    np.set_printoptions(linewidth=280, suppress=True)

    # (4,2)
    # this shold be just one entry
    idx_s = [[0, 1, 2], [0, 1, 2]]
    idx_d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]

    # (2,1)
    # idx_s = [[2], [2]]
    # idx_d = [[3, 7, 9, 10, 11],[3, 7, 9, 10, 11]]

    bds = np.arange(1.0, 2.0, 0.05)

#   bds = [1.5]

    spin = 0
    spin_restricted = False
    basis = "aug-ccpvtz"

#   ao_labels = ["C 2p", "C 2s", "C 3p", "C 3s","O 2p", "O 2s", "O 3p", "O 3s" ]
    ao_labels = ["N 2p", "N 2s","N 3p", "N 3s", "N 3d"]

    res_hf = []
    res_mp = []
    res_cc = []
    res_mpcc = []
    res_cclo = []
    for bd in bds:
        print("Bond length :", bd)

        if bd == bds[0]:
            mol, mf, e_hf, e_mp, e_cc = get_N2(basis, bd, spin, spin_restricted)
        else:
            mol, mf, e_hf, e_mp, e_cc = get_N2(basis, bd, spin, spin_restricted, dm0=dm0)


        dm0 = mf.make_rdm1()

        c_lo, act_hole, act_part = get_localized_orbs(mol, mf, ao_labels, minao="dzp")
        frag = [[act_hole, act_part]]

        e_mp_cc_prev = -np.inf
        e_diff = np.inf
        tol = 1e-4
        count = 0
        count_tol = 100

        mycc = cc.umpccsd.CCSD(mf, mo_coeff = c_lo)
        mycc.verbose = 3
        mycc.max_cycle = 50
        mycc.diis_space = 14
        mycc.level_shift = 0.5

        e_mp_cc_its = []

        adiis = lib.diis.DIIS(mycc)

        while e_diff > tol and count < count_tol:

            if count > 0:

                print("=== Starting OOMP2 iterations ======")     

                mycc.max_cycle = 1
                # mp_lo = oo_mp2(mf, c_lo, mp_cc_t1, mp_cc_t2, spin_restricted)
                mycc.kernel(act_hole = None, 
                            act_particle = None, 
                            idx_s = None, 
                            idx_d = None, 
                          #  t1=mycc.t1, 
                          #  t2=mycc.t2, 
                            t1=mp_cc_t1, 
                            t2=mp_cc_t2, 
                            oo_mp2 = True)

                mp_t2 = mycc.t2
                mp_t1 = mycc.t1
            else:
                mp_lo = oo_mp2(mf, c_lo, None, None, spin_restricted)
                mp_t2 = mp_lo[2]
                mp_t1 = mp_lo[3]

            print("=== Starting UMPCCSD ======")     
            mycc.max_cycle = 50

            e_mp_cc, mp_cc_t1, mp_cc_t2 = fragmented_mpcc_unrestricted(
                frag, mycc, mp_t2, mp_t1, idx_s, idx_d
            )

#           t2shape = [x.shape for x in mp_cc_t2]
#           mp_cc_t2 = np.hstack([x.ravel() for x in mp_cc_t2])
#           mp_cc_t2 = adiis.update(mp_cc_t2)
#           mp_cc_t2 = lib.split_reshape(mp_cc_t2, t2shape)


            t1shape = [x.shape for x in mp_cc_t1]
            mp_cc_t1 = np.hstack([x.ravel() for x in mp_cc_t1])
            mp_cc_t1 = adiis.update(mp_cc_t1)
            mp_cc_t1 = lib.split_reshape(mp_cc_t1, t1shape)


            e_diff = np.abs(e_mp_cc - e_mp_cc_prev)
            e_mp_cc_its.append(e_mp_cc)
            e_mp_cc_prev = e_mp_cc
            count += 1

        print(f"SCF UMPCC stopped after {count} iterations, iterate difference: {e_diff}")
#        breakpoint()

        res_hf.append(e_hf)
        res_mp.append(e_mp)
        res_cc.append(e_cc)
        res_mpcc.append(e_mp_cc)

    res_hf = np.array(res_hf)
    res_mp = np.array(res_mp)
    res_cc = np.array(res_cc)
    res_mpcc = np.array(res_mpcc)

    #    breakpoint()

    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
    ax1.plot(bds, res_hf, label="UHF")
    ax1.plot(bds, res_hf + res_mp, label="UMP2")
    ax1.plot(bds, res_hf + res_cc, label="UCCSD")
    ax1.plot(bds, res_hf + res_mpcc, label="UMP-CCSD (4,2)")

    ax1.legend()
    ax1.set_title(f"Absolute energies ({basis}, {ao_labels}, spin = {spin})", pad=20)
    ax1.set_xlabel("Bond length")

    name = f"abs_energies_{basis}_N2_relaxed"
    #    for elem in ao_labels:
    #        name += f"_{elem}"
    name += f"_{spin}"
    name += ".png"

    print(name)

    fig.savefig(
        name,
        bbox_inches="tight",
        dpi=150,
        transparent=True,
        facecolor="w",
        edgecolor="w",
        orientation="landscape",
    )

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(bds, res_cc - res_mp, label="UMP2")
    ax1.plot(bds, res_cc - res_mpcc, label="UMP-CCSD (4,2)")

    ax1.legend()
    ax1.set_title(f"Corr. energy diff. ({basis}, {ao_labels}, spin = {spin})", pad=20)
    ax1.set_xlabel("Bond length")

    name = f"corr_energy_diff_{basis}_N2_relaxed"
    # for elem in ao_labels:
    #    name += f"_{elem}"
    name += f"_{spin}"
    name += ".png"

    fig.savefig(
        name,
        bbox_inches="tight",
        dpi=150,
        transparent=True,
        facecolor="w",
        edgecolor="w",
        orientation="landscape",
    )

    plt.show()


#    breakpoint()
