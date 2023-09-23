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

import argparse

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

    if isinstance(mf, scf.uhf.UHF):
        act_hole = (
            np.where(avas_obj.occ_weights[0] > avas_obj.threshold)[0],
            np.where(avas_obj.occ_weights[1] > avas_obj.threshold)[0]
        )
        act_part = (
            np.where(avas_obj.vir_weights[0] > avas_obj.threshold)[0] + mf.mol.nelec[0],
            np.where(avas_obj.vir_weights[1] > avas_obj.threshold)[0] + mf.mol.nelec[1]
        )
    elif isinstance(mf, scf.rohf.ROHF):
        # NOTE check the output format. Do we want a tuple here?
        act_hole = (
            np.where(avas_obj.occ_weights > avas_obj.threshold)[0],
            np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        )

        act_part = (
            np.where(avas_obj.vir_weights > avas_obj.threshold)[0] + mf.mol.nelec[0],
            np.where(avas_obj.vir_weights > avas_obj.threshold)[0] + mf.mol.nelec[1]
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


def get_mol(basis, spin_restricted, state, frozen, dm0=None):

    atoms = [
    ['Cu', [  0.0000000000000,   0.0000000000000,   0.0000000000000]],
    ['O',  [  0.0000000000000,   3.6666299464596,   0.0000000000000]],
    ['H',  [  1.4666221209110,   4.7748050350325,   0.0000000000000]],
    ['O',  [  3.6666299464596,   0.0000000000000,   0.0000000000000]],
    ['H',  [  4.7748050350325,   1.4666221209110,   0.0000000000000]],
    ['O',  [  0.0000000000000,  -3.6666299464596,   0.0000000000000]],
    ['H',  [ -1.4666221209110,   4.7748050350325,   0.0000000000000]],
    ['H',  [  1.4666221209110,  -4.7748050350325,   0.0000000000000]],
    ['H',  [ -1.4666221209110,  -4.7748050350325,   0.0000000000000]],
    ['O',  [ -3.6666299464596,   0.0000000000000,   0.0000000000000]],
    ['H',  [ -4.7748050350325,   1.4666221209110,   0.0000000000000]],
    ['H',  [  4.7748050350325,  -1.4666221209110,   0.0000000000000]],
    ['H',  [ -4.7748050350325,  -1.4666221209110,   0.0000000000000]],
    ]

    mol           = gto.M()
    mol.basis     = basis
    mol.atom      = atoms
    mol.unit      = 'Bohr'
    mol.verbose   = 3
    mol.nelectron = 67 # 69 = 29 + 4*10  
    mol.spin      = 1
    mol.charge    = 2
    mol.symmetry  = True
    mol.build()

    if spin_restricted == 'ROHF':
        mf = mol.ROHF().newton()
    elif spin_restricted == 'UHF':
        mf = mol.UHF().newton()
    else:
        print('Requested spin configuration not implemented')
        exit()

    if state == 'b2g':
        mf.irrep_nelec = {'Ag': (11, 10), 'B1g': (3, 3), 'B2g': (2, 2), 'B3g': (2, 2), 'B1u': (4, 4), 'B2u': (6, 6), 'B3u': (6, 6)}
    elif state == 'b1g':
        mf.irrep_nelec = {'Ag': (11, 11), 'B1g': (3, 2), 'B2g': (2, 2), 'B3g': (2, 2), 'B1u': (4, 4), 'B2u': (6, 6), 'B3u': (6, 6)}
    elif state == 'eg':
        mf.irrep_nelec = {'Ag': (11, 11), 'B1g': (3, 3), 'B2g': (2, 1), 'B3g': (2, 2), 'B1u': (4, 4), 'B2u': (6, 6), 'B3u': (6, 6)}
    else:
        print('Irrep for targeted state not defined!')
        exit()


    mf.run()
    mf.stability()
    mf.analyze()
    mfe = mf.e_tot
    print(f'Energy: {mfe}')

    # Freezing cores
    mymp = mp.UMP2(mf) 
    mycc = cc.UCCSD(mf)
    if frozen == 'True':
        # Freezing He, Ne, and Ar core in N, Cu, Cl, O 
        # => 26 core electrons, i.e., 13 core orbitals
        core = [i for i in range(13)]
        mycc.set(frozen = core)
        mymp.set(frozen = core)

    mycc.verbose = 4
    mymp.verbose = 4
    mymp.kernel()
    mycc.kernel()

    mycc.diis = 12
    mycc.max_cycle = 80

    return mol, mf, mf.e_tot, mymp.e_corr, mycc.e_corr

def oo_mp2(mf, c_lo, t1, t2):

    if isinstance(mf, scf.rohf.ROHF):
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mymp.async_io = True

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, t1=t1, t2=t2, verbose=0)

    elif isinstance(mf, scf.uhf.UHF):
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

    parser = argparse.ArgumentParser(
        description="Arguments passed to run UMPCC-OO for Cu_H2O_4_2+"
    )

    parser.add_argument("--spin_restricted", type = str, default = 'UHF', help="For restricted openshell set to ROHF for spin unrestricted to UHF")
    parser.add_argument("--openshell_option", type=int)
    parser.add_argument("--basis", type=str, help="Global sasis")
    parser.add_argument("--minao", type=str, help="MINAO for AVAS")
    parser.add_argument('--ao_labels', type=str)
    parser.add_argument('--state', type=str, help="b2g, b1g, or eg")
    parser.add_argument("--frozen", type=str, help="Freezing core electrons in reference calcs")

    args = parser.parse_args()

    spin_restricted = args.spin_restricted
    openshell_option = args.openshell_option
    basis = args.basis
    minao = args.minao
    ao_labels = args.ao_labels.split(',')
    state = args.state
    frozen = args.frozen

    # (4,2)
    # this shold be just one entry
    # NOTE These are the active-inactive indices that are kept fix in MPCC
    idx_s = [[0, 1, 2], [0, 1, 2]]
    idx_d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]

    # NOTE These are the active-inactive indices that are kept fix in the OO-MP2 orbital relaxation
    idx_s_oomp2 = [
            np.delete(np.arange(4), np.array(idx_s[0])), 
            np.delete(np.arange(4), np.array(idx_s[1]))
             ]
    idx_d_oomp2 = [
            np.delete(np.arange(16), np.array(idx_d[0])),
            np.delete(np.arange(16), np.array(idx_d[1])),
            np.delete(np.arange(16), np.array(idx_d[2]))
            ]

    # (2,1)
    # idx_s = [[2], [2]]
    # idx_d = [[3, 7, 9, 10, 11],[3, 7, 9, 10, 11]]

    # NOTE make this to an input
    # NOTE feed the targetted state as input

    mol, mf, e_hf, e_mp, e_cc = get_mol(basis, spin_restricted, state, frozen)

    dm0 = mf.make_rdm1()

    # NOTE why not always convert to UHF object?
    mf = scf.addons.convert_to_uhf(mf)

    c_lo, act_hole, act_part = get_localized_orbs(mol, mf, ao_labels, minao=minao, openshell_option = openshell_option)
    frag = [[act_hole, act_part]]

    print(f"AO labels:       : {ao_labels}")
    print(f"Active hole      : {act_hole}")
    print(f"No of act. hole  : ({len(act_hole[0])},{len(act_hole[1])})")
    print(f"Active particel  : {act_part}")
    print(f"No of act. part. : ({len(act_part[0])},{len(act_part[1])})")

    e_mp_cc_prev = -np.inf
    e_diff = np.inf
    tol = 1e-5
    count = 0
    count_tol = 100

    mycc = cc.umpccsd.CCSD(mf, mo_coeff = c_lo)
 
    # Freezing cores
    freezing = False
    if freezing:
        # Freezing He, Ne, and Ar core in N, Cu, Cl, O 
        # => 26 core electrons, i.e., 13 core orbitals
        core = [i for i in range(13)]
        mycc.set(frozen = core)

    mycc.verbose = 4
    mycc.max_cycle = 50
    mycc.diis_space = 8
    mycc.level_shift = 0.5

    e_mp_cc_its = []

    # DIIS
    adiis = lib.diis.DIIS(mycc)

    while e_diff > tol and count < count_tol:

        if count > 0:

            print("=== Starting OOMP2 iterations ======")     

            mycc.max_cycle = 50
            # mp_lo = oo_mp2(mf, c_lo, mp_cc_t1, mp_cc_t2, spin_restricted)
            nocca, noccb, _, _ = mycc.t2[1].shape
            mycc.kernel(act_hole = [frag[0][0][0], frag[0][0][1]], 
                        act_particle = [frag[0][1][0] - nocca, frag[0][1][1] - noccb], 
                        idx_s = idx_s_oomp2, 
                        idx_d = idx_d_oomp2, 
                      #  t1=mycc.t1, 
                      #  t2=mycc.t2, 
                        t1=mp_cc_t1, 
                        t2=mp_cc_t2, 
                        oo_mp2 = True)

            mp_t2 = mycc.t2
            mp_t1 = mycc.t1
        else:
            mp_lo = oo_mp2(mf, c_lo, None, None)
            mp_t2 = mp_lo[2]
            mp_t1 = mp_lo[3]

        print("=== Starting UMPCCSD ======")     
        mycc.max_cycle = 50

        e_mp_cc, mp_cc_t1, mp_cc_t2 = fragmented_mpcc_unrestricted(
            frag, mycc, mp_t2, mp_t1, idx_s, idx_d
        )


        t1shape = [x.shape for x in mp_cc_t1]
        mp_cc_t1 = np.hstack([x.ravel() for x in mp_cc_t1])
        mp_cc_t1 = adiis.update(mp_cc_t1)
        mp_cc_t1 = lib.split_reshape(mp_cc_t1, t1shape)

        e_diff = np.abs(e_mp_cc - e_mp_cc_prev)
        e_mp_cc_its.append(e_mp_cc)
        e_mp_cc_prev = e_mp_cc
        count += 1

        print(f'\n Iterate progress : e-diff = {e_diff} (tol = {tol}) \n')

    print(f"SCF UMPCC stopped after {count} iterations, iterate difference: {e_diff}")

    breakpoint()
