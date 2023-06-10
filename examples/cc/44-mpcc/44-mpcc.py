"""
    Vanilla version of mp2-ccsd
"""

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


def get_reference_values(system, **kwargs):

    if system == "H8":
        # Fetching basis
        if "basis" in kwargs:
            basis = kwargs["basis"]
        else:
            basis = "sto6g"

        # Fetching bond_length of H molecules
        if "bond_length" in kwargs:
            bond_length = kwargs["bond_length"]
        else:
            bond_length = 0.741  # AA

        # Fetching system parameter dist
        if "dist" in kwargs:
            dist = kwargs["dist"]
        else:
            dist = 2  # AA

        if "localization" in kwargs:
            localization = kwargs["localization"]
        else:
            localization = None

        if localization == "AVAS":
            if "ao_labels" in kwargs:
                ao_labels = kwargs["ao_labels"]
            else:
                print("did not provide ao labels for AVAS")
                exit()
        else:
            ao_labels = None

        return get_h8_reference_value(basis, bond_length, dist, localization, ao_labels)
    elif system == "Be-ring":
        # Fetching basis
        if "basis" in kwargs:
            basis = kwargs["basis"]
        else:
            basis = "sto6g"

        # Fetching number of Atoms
        if "num_atoms" in kwargs:
            num_atoms = kwargs["num_atoms"]
        else:
            num_atoms = 10

        # Fetching bond_length
        if "bond_length" in kwargs:
            bond_length = kwargs["bond_length"]
        else:
            bond_length = 0.741  # AA
        
        if "localization" in kwargs:
            localization = kwargs["localization"]
        else:
            localization = None

        if localization == "AVAS" or localization == "AVAS-DMET":
            if "ao_labels" in kwargs:
                ao_labels = kwargs["ao_labels"]
            else:
                print("did not provide ao labels for AVAS")
                exit()
        else:
            ao_labels = None

        return get_Be_ring_reference_value(
            num_atoms, basis, bond_length, localization, ao_labels
        )

    elif system == "NO":
        # Fetching basis
        if "basis" in kwargs:
            basis = kwargs["basis"]
        else:
            basis = "sto6g"

        # Fetching bond_length
        if "bond_length" in kwargs:
            bond_length = kwargs["bond_length"]
        else:
            bond_length = 1.098 # AA
        
        if "localization" in kwargs:
            localization = kwargs["localization"]
        else:
            localization = None

        if localization == "AVAS":
            if "ao_labels" in kwargs:
                ao_labels = kwargs["ao_labels"]
            else:
                print("did not provide ao labels for AVAS")
                exit()
        else:
            ao_labels = None
        
        if "spin_restricted" in kwargs:
            spin_restricted = kwargs["spin_restricted"]
        else:
            spin_restricted = False

        return get_NO_reference_value(
            basis, bond_length, localization, ao_labels, spin_restricted
        )
    elif system == "Cl2":
        # Fetching basis
        if "basis" in kwargs:
            basis = kwargs["basis"]
        else:
            basis = "sto6g"

        # Fetching bond_length
        if "bond_length" in kwargs:
            bond_length = kwargs["bond_length"]
        else:
            bond_length = 1.988 # AA
        
        if "localization" in kwargs:
            localization = kwargs["localization"]
        else:
            localization = None

        if localization == "AVAS" or localization == "AVAS-DMET":
            if "ao_labels" in kwargs:
                ao_labels = kwargs["ao_labels"]
            else:
                print("did not provide ao labels for AVAS")
                exit()
        else:
            ao_labels = None
        
        if "spin_restricted" in kwargs:
            spin_restricted = kwargs["spin_restricted"]
        else:
            spin_restricted = False

        return get_Cl2_reference_value(
            basis, bond_length, localization, ao_labels, spin_restricted
        )



def rotate(angle, vec):

    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rot @ vec

def get_h8_atoms(bond_length, dist, visualize=False):

    xy_pos = []
    start_vec = np.array([0, bond_length / (2 * np.sin(45 * np.pi / 360))])
    start_vec = rotate(-45 * np.pi / 360, start_vec)
    for i in range(8):
        angle = i * 45 * np.pi / 180
        xy_pos.append(rotate(angle, start_vec))

    # applying shift
    shift_vec = np.array([dist, 0])
    xy_pos[2] -= shift_vec
    xy_pos[3] -= shift_vec
    xy_pos[6] += shift_vec
    xy_pos[7] += shift_vec

    if visualize:
        for pos in xy_pos:
            plt.plot(pos[0], pos[1], "b*")

        plt.plot(xy_pos[2][0], xy_pos[2][1], "r*")
        plt.plot(xy_pos[3][0], xy_pos[3][1], "r*")
        plt.plot(xy_pos[6][0], xy_pos[6][1], "r*")
        plt.plot(xy_pos[7][0], xy_pos[7][1], "r*")

        plt.show()

    return [["H", [0, elem[0], elem[1]]] for elem in xy_pos]


def get_localized_orbs(mol, mf, localization, ao_labels, molden_bool=False, name=None):

    if localization == "IAO":

        if isinstance(mf, scf.uhf.UHF):
            c_lo_a = np.zeros((mf.mo_coeff[0].shape), mf.mo_coeff[0].dtype)
            c_lo_b = np.zeros((mf.mo_coeff[0].shape), mf.mo_coeff[0].dtype)

            # localizing occuppied orbitals
            pm_occ_a = lo.PM(mol, mf.mo_coeff[0][:, mf.mo_occ[0] > 0], mf)
            pm_occ_a.pop_method = "iao"
            C = pm_occ_a.kernel()
            c_lo_a[:, mf.mo_occ[0] > 0] = C

            pm_occ_b = lo.PM(mol, mf.mo_coeff[1][:, mf.mo_occ[1] > 0], mf)
            pm_occ_b.pop_method = "iao"
            C = pm_occ_b.kernel()
            c_lo_b[:, mf.mo_occ[1] > 0] = C

            # localizing virtual orbitals
            pm_vir_a = lo.PM(mol, mf.mo_coeff[0][:, mf.mo_occ[0] < 1e-12], mf)
            pm_vir_a.pop_method = "iao"
            C = pm_vir_a.kernel()
            c_lo_a[:, mf.mo_occ[0] < 1e-12] = C

            pm_vir_b = lo.PM(mol, mf.mo_coeff[1][:, mf.mo_occ[1] < 1e-12], mf)
            pm_vir_b.pop_method = "iao"
            C = pm_vir_b.kernel()
            c_lo_b[:, mf.mo_occ[1] < 1e-12] = C

            c_lo = [c_lo_a, c_lo_b]

        else:
            c_lo = np.zeros((mf.mo_coeff.shape), mf.mo_coeff.dtype)

            # localizing occuppied orbitals
            pm_occ = lo.PM(mol, mf.mo_coeff[:, mf.mo_occ > 0], mf)
            pm_occ.pop_method = "iao"
            C = pm_occ.kernel()
            c_lo[:, mf.mo_occ > 0] = C

            # localizing virtual orbitals
            pm_vir = lo.PM(mol, mf.mo_coeff[:, mf.mo_occ < 1e-12], mf)
            pm_vir.pop_method = "iao"
            C = pm_vir.kernel()
            c_lo[:, mf.mo_occ < 1e-12] = C

    elif localization == "AVAS":

        avas_obj = avas.AVAS(mf, ao_labels)
        avas_obj.kernel()

        c_lo = avas_obj.mo_coeff

    elif localization == "meta-lowdin":

        c_lo = np.zeros((mf.mo_coeff.shape), mf.mo_coeff.dtype)

        # localizing occuppied orbitals
        pm_occ = lo.PM(mol, mf.mo_coeff[:, mf.mo_occ > 0], mf)
        pm_occ.pop_method = "meta-lowdin"
        C = pm_occ.kernel()
        c_lo[:, mf.mo_occ > 0] = C

        # localizing virtual orbitals
        pm_vir = lo.PM(mol, mf.mo_coeff[:, mf.mo_occ < 1e-12], mf)
        pm_vir.pop_method = "meta-lowdin"
        C = pm_vir.kernel()
        c_lo[:, mf.mo_occ < 1e-12] = C

    elif localization == "MO":
        c_lo = mf.mo_coeff

    elif localization == "AVAS-DMET":
        '''
        AVAS localization + DMET bath
        NOTE only works for spin restricted formulation atm
        '''

        avas_obj = avas.AVAS(mf, ao_labels, canonicalize = False)
        avas_obj.kernel()

        c_lo = avas_obj.mo_coeff

        # Constructing DMET bath 
        ovlp = mf.mol.get_ovlp()
        shalf = fractional_matrix_power(ovlp, 0.5)
        shalf_inv = fractional_matrix_power(ovlp, -0.5)

        dm = mf.make_rdm1()
        dm_lo = c_lo.T @ dm @ c_lo        
    
        idx_core = np.where(avas_obj.occ_weights < 1e-10)[0]
        idx_act_occ = np.where(avas_obj.occ_weights > 1e-10)[0]
        idx_frag = np.append(idx_act_occ, np.where(avas_obj.vir_weights >avas_obj. threshold)[0] + mf.mol.nelectron//2 )
        
        env_idx = np.delete(np.arange(len(mf.mo_occ)), idx_frag) 
        
        d21 = dm[np.ix_(env_idx, idx_frag)]

        U_mat, s_val, _ = np.linalg.svd(d21, full_matrices=False)
        bath = np.where(s_val > 1e-6)[0]
        U_bath = U_mat[:,bath]
        U_bath = np.vstack((U_bath[idx_core, :], np.eye(len(bath)), U_bath[idx_core[-1]+1:, :]))

        c_lo_bath = c_lo @ U_bath

        breakpoint()


    else:
        print("Localization not defined")
        exit()

    if molden_bool:
        if name is None:
            print("Did not provide name for molden file")
            exit()
        
        if isinstance(mf, scf.uhf.UHF):
            with open(name + "_mo.molden", "w") as f1:
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, mf.mo_coeff[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0])
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, mf.mo_coeff[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1])
                
                print("Wrote molden file: \n", f1)

            with open(name + f"_{localization}.molden", "w") as f1:
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, c_lo[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0])
                
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, c_lo[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1])
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
        

        return c_lo


def flattening(t2, nocc, nvirt):

    return t2.reshape((nocc ** 2, nvirt ** 2))

def get_Cl2_reference_value(basis, bond_length, localization, ao_labels, spin_restricted):

    mol = gto.M()
    mol.basis = basis
    mol.atom = [['Cl', [0,0,0]] , ['Cl', [bond_length,0,0]]]
    mol.verbose = 3
    mol.unit = "angstrom"
    mol.build()

    if spin_restricted:
        mf = mol.RHF().run()
    else:
        mf = mol.UHF().run()

    # localization
    name = f"molden/molden_Cl2_{basis}_{bond_length}"
    c_lo = get_localized_orbs(
        mol, mf, localization, ao_labels, molden_bool=True, name=name
    )

    if spin_restricted:
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mycc = cc.CCSD(mf).run(verbose=0)

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)
    else:
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()
        mycc = cc.UCCSD(mf).run()

        breakpoint()
        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

    breakpoint()
    return mf, c_lo, mp_lo[2], mf.e_tot, mymp.e_corr, mycc.e_corr





def get_NO_reference_value(basis, bond_length, localization, ao_labels, spin_restricted):

    mol = gto.M()
    mol.basis = basis
    mol.atom = [['N', [0,0,0]] , ['O', [bond_length,0,0]]]
    mol.verbose = 3
    mol.spin = 1
    mol.unit = "angstrom"
    mol.build()

    if spin_restricted:
        mf = mol.RHF().run()
    else:
        mf = mol.UHF().run()

    # localization
    name = f"molden/molden_NO_{basis}_{bond_length}"
    c_lo = get_localized_orbs(
        mol, mf, localization, ao_labels, molden_bool=True, name=name
    )

    if spin_restricted:
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mycc = cc.CCSD(mf).run(verbose=0)

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)
    else:
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()
        mycc = cc.UCCSD(mf).run()

        breakpoint()
        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

    breakpoint()
    return mf, c_lo, mp_lo[2], mf.e_tot, mymp.e_corr, mycc.e_corr








def get_Be_ring_reference_value(num, basis, bond_length, localization, ao_labels):

    structure = ring.make(num, bond_length)
    atom = [("Be %f %f %f" % xyz) for xyz in structure]

    mol = gto.M()
    mol.basis = basis
    mol.atom = atom
    mol.verbose = 3
    mol.unit = "angstrom"
    mol.build()

    mf = mol.RHF().run()

    # localization
    name = f"molden/molden_Be_ring_{num}_{basis}_{bond_length}"
    c_lo = get_localized_orbs(
        mol, mf, localization, ao_labels, molden_bool=True, name=name
    )

    # Running regular MP2 and CCSD
    mymp = mp.MP2(mf).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)

    # Running localized MP2
    eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
    mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

    return mf, c_lo, mp_lo[2], mf.e_tot, mymp.e_corr, mycc.e_corr


def check_compressibility(mycc, basis, title):

    t2 = mycc.t2
    t2_mat = t2.reshape((t2.shape[0] ** 2, t2.shape[-1] ** 2))

    u, s, v = np.linalg.svd(t2_mat, full_matrices=False)

    vec = np.zeros_like(s)
    energies = []
    differences = []

    for i in range(len(s)):
        vec[: i + 1] = s[: i + 1]

        t2_approx = u @ np.diag(vec) @ v
        t2_approx = t2_approx.reshape(t2.shape)

        energies.append(mycc.energy(mycc.t1, t2_approx) / mycc.e_corr)
        differences.append(np.linalg.norm(t2_approx - mycc.t2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(1 + np.arange(len(s)), energies, "b-", label="Corr. energy %")
    ax1.plot(0.9 * np.ones(len(s)), "--", label="90%")
    ax1.plot(0.8 * np.ones(len(s)), "--", label="80%")
    ax1.plot(0.7 * np.ones(len(s)), "--", label="70%")
    ax1.plot(0.6 * np.ones(len(s)), "--", label="60%")
    ax1.set_title(f"Correlation energy recovery ({basis})")
    ax1.set_xlabel("T2 approximation rank")
    ax1.legend()

    ax2.semilogy(1 + np.arange(len(s)), differences)
    ax2.set_title("Error in T2 tensor")
    ax2.set_ylim((1e-4, 1))

    fig.suptitle(title)
    plt.show()

    exit()


def get_h8_reference_value(basis, bond_length, dist, localization, ao_labels):

    atoms = get_h8_atoms(bond_length, dist, visualize=False)

    mol = gto.M()
    mol.basis = basis
    mol.atom = atoms
    mol.verbose = 3
    mol.unit = "angstrom"
    mol.build()

    mf = mol.RHF().run()

    # cisolver = fci.FCI(mf)
    # fci_res = cisolver.kernel()

    # localization
    name = f"molden/molden_H8_{basis}_{bond_length}_{dist}"
    c_lo = get_localized_orbs(
        mol, mf, localization, ao_labels, molden_bool=True, name=name
    )

    # Running regular MP2 and CCSD
    mymp = mp.MP2(mf).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)

    # check_compressibility(mycc, basis, f'Testing H-8 system ({bond_length}, {dist})')

    # Running localized MP2
    eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
    mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

    return mf, c_lo, mp_lo[2], mf.e_tot, mymp.e_corr, mycc.e_corr


def get_t_int(t2, eps=0.9):

    max_val = np.max(np.abs(t2))
    idx = np.where(np.abs(t2) > eps * max_val)
    idx = np.asarray(idx)
    return [idx[:, i] for i in range(len(idx[0, :]))]


def get_hring_references_PNO_avas(natom, basis, bd):

    structure = ring.make(natom, bd)
    atom = [("H %f %f %f" % xyz) for xyz in structure]
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.verbose = 5
    mol.build()

    mf = mol.RHF().run()

    ao_labels = ["H 1s"]

    avas_obj = avas.AVAS(mf, ao_labels, with_iao=True)
    avas_obj.kernel()
    name = f"molden/molden_h10_{basis}_{bd}"

    write_molden = True
    if write_molden:
        with open(name + "mo.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
            print("Wrote molden file: \n", f1)

        with open(name + "avas.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(
                mol, f1, avas_obj.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ
            )
            print("Wrote molden file: \n", f1)

    breakpoint()


def get_N2_references_PNO_avas(basis, bd):

    mol = gto.M()
    mol.basis = basis
    # mol.atom  = [['N', [0,0,0]] , ['N', [bd,0,0]]]
    # mol.atom = [['F', [0,0,0]],['F', [bd,0,0]]]
    mol.atom = [["Cl", [0, 0, 0]], ["Cl", [bd, 0, 0]]]

    mol.verbose = 0
    mol.build()

    mf = mol.RHF().run()

    # ao_labels = ['N 2p', 'N 2s']
    # ao_labels = ['F 2p', 'F 2s']
    ao_labels = ["Cl 3p"]

    avas_obj = avas.AVAS(mf, ao_labels)
    avas_obj.kernel()
    name = f"molden/molden_{basis}_{bd}"

    write_molden = False
    if write_molden:
        with open(name + "mo.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
            print("Wrote molden file: \n", f1)

        with open(name + "avas.molden", "w") as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(
                mol, f1, avas_obj.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ
            )
            print("Wrote molden file: \n", f1)

    # Building MP2 objects for the _make_eris routine
    mymp = mp.MP2(mf)  # , mo_coeff=c_mo)
    mymp.max_cycle = 30
    mymp.diis_space = 8
    mymp.verbose = 0

    eris = mp.mp2._make_eris(mymp, mo_coeff=avas_obj.mo_coeff)
    lo_mp = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    # Running regular Mp2 and CCSD
    mymp = mp.MP2(mf).run()
    mycc_reg = cc.CCSD(mf).run(verbose=0)

    return (
        3,
        3,
        mf,
        avas_obj.mo_coeff,
        mymp,
        lo_mp[2],
        mf.e_tot,
        lo_mp[1],
        mycc_reg.e_corr,
    )


def get_N2_references_PNO(basis, bd):

    mol = gto.M()
    mol.basis = basis
    # mol.atom  = [['N', [0,0,0]] , ['N', [bd,0,0]]]
    # mol.atom = [['F', [0,0,0]],['F', [bd,0,0]]]
    mol.atom = [["Cl", [0, 0, 0]], ["Cl", [bd, 0, 0]]]
    mol.verbose = 3
    mol.build()

    mf = mol.RHF().run()

    # NOTE orthogonalize these!

    ovlp = mol.get_ovlp()
    shalf = fractional_matrix_power(ovlp, 0.5)
    shalf_inv = fractional_matrix_power(ovlp, -0.5)
    orbitals = mf.mo_coeff

    nocc = mol.nelectron // 2
    nvirt = len(mf.mo_occ) - mol.nelectron // 2
    # nao x nocc+nvirt
    # NOTE this is hard-coded for now
    nactiv = 5

    # NOTE rename c_mo ...
    c_mo = np.zeros((len(mf.mo_occ), nactiv + nvirt), orbitals.dtype)

    pm_occ = lo.PM(mol, mf.mo_coeff[:, :nocc], mf)
    pm_occ.pop_method = "iao"
    C = pm_occ.kernel()
    c_mo[:, :nactiv] = C[:, nocc - nactiv : nocc]
    c_mo[:, nactiv:] = mf.mo_coeff[:, nocc:]

    occupation = mf.mo_occ[nocc - nactiv :]
    mymp = mp.MP2(mf, mo_coeff=c_mo, mo_occ=occupation)
    mymp.max_cycle = 30
    mymp.diis_space = 8
    mymp.verbose = 0

    eris = mp.mp2._make_eris(mymp, mo_coeff=c_mo)
    eris_ovov = np.asarray(eris.ovov).reshape(nactiv, nvirt, nactiv, nvirt)

    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    vhf = mf.get_veff(mol, dm)
    fockao = mf.get_fock(vhf=vhf, dm=dm)
    fock = c_mo.conj().T.dot(fockao).dot(c_mo)
    mo_e_o = np.diag(fock)[:nactiv]
    mo_e_v = np.diag(fock)[nactiv:]
    eia = mo_e_o[:, None] - mo_e_v

    t2 = eris_ovov.conj().transpose(0, 2, 1, 3)
    t2 /= lib.direct_sum("ia,jb->ijab", eia, eia)

    dm1vir = np.zeros((nvirt, nvirt), dtype=t2.dtype)
    for i in range(nactiv):
        t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += lib.einsum("jca,jcb->ba", l2i, t2i) * 2 - lib.einsum(
            "jca,jbc->ba", l2i, t2i
        )

    # gamma_ao = c_mo[:,nactiv:] @ dm1vir @ c_mo[:,nactiv:].T
    # u, s, _ = np.linalg.svd(gamma_ao)

    Uu, Ss, _ = np.linalg.svd(dm1vir)
    tol = 1e-3

    # NOTE rename orbs to c_act_virt

    orbs = shalf @ c_mo[:, nactiv:] @ Uu[:, Ss > tol]

    n_act_virt = np.sum(Ss > tol)
    print("N active virtual: ", n_act_virt)
    P = np.eye(len(mf.mo_occ)) - orbs @ orbs.T
    proj = P @ shalf @ c_mo[:, nactiv:]
    u_mat, val, _ = np.linalg.svd(proj)
    idx = np.where(val > 1e-7)[0]
    proj_virt = u_mat[:, idx]

    coeffs = np.hstack((C, shalf_inv @ orbs, shalf_inv @ proj_virt))

    mymp = mp.MP2(mf, mo_coeff=coeffs)
    mymp.max_cycle = 30
    mymp.diis_space = 8
    mymp.verbose = 0

    eris = mp.mp2._make_eris(mymp, mo_coeff=coeffs)
    pno_mp = mp.mp2._iterative_kernel(mymp, eris)

    mycc_reg = cc.CCSD(mf).run(verbose=5)

    return (
        nactiv,
        n_act_virt,
        mf,
        coeffs,
        mymp,
        pno_mp[2],
        mf.e_tot,
        pno_mp[1],
        mycc_reg.e_corr,
    )


def get_N2_references(basis, bd):

    mol = gto.M()
    mol.basis = basis
    # mol.atom  = [['N', [0,0,0]] , ['N', [bd,0,0]]]
    # mol.atom  = [['F', [0,0,0]] , ['F', [bd,0,0]]]
    mol.atom = [["Cl", [0, 0, 0]], ["Cl", [bd, 0, 0]]]
    mol.verbose = 3
    mol.build()

    mf = mol.RHF().run()

    orbitals = mf.mo_coeff
    nocc = mol.nelectron // 2
    nvirt = len(mf.mo_occ) - nocc
    c_lo = np.zeros((orbitals.shape), orbitals.dtype)

    # orbitals = mf.mo_coeff[:,mf.mo_occ>0]
    pm_occ = lo.PM(mol, mf.mo_coeff[:, :nocc], mf)
    pm_occ.pop_method = "meta-lowdin"
    C = pm_occ.kernel()
    c_lo[:, :nocc] = C

    pm_vir = lo.PM(mol, mf.mo_coeff[:, nocc:], mf)
    pm_vir.pop_method = "meta-lowdin"
    C = pm_vir.kernel()
    c_lo[:, nocc:] = C

    # Building MP2 objects for the _make_eris routine
    mymp = mp.MP2(mf)  # , mo_coeff=c_mo)
    mymp.max_cycle = 30
    mymp.diis_space = 8
    mymp.verbose = 0
    # Running local MP2 should be conceptually very easy!

    eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
    lo_mp = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    # Running regular Mp2 and CCSD
    mymp = mp.MP2(mf).run()

    # constructing MP2 NOs
    dm_mp = mymp.make_rdm1()
    vals, nos = np.linalg.eigh(dm_mp)
    vals = vals[::-1]
    nos = nos[:, ::-1]
    no_coeff = mf.mo_coeff.dot(nos)

    # localize the MP2-NOs
    c_mo = np.zeros((orbitals.shape), orbitals.dtype)
    c_mo = no_coeff
    pm_occ = lo.PM(mol, no_coeff[:, :nocc], mf)
    pm_occ.pop_method = "meta-lowdin"
    C = pm_occ.kernel()
    c_mo[:, :nocc] = C

    pm_vir = lo.PM(mol, no_coeff[:, nocc : nocc + 3], mf)
    pm_vir.pop_method = "meta-lowdin"
    C = pm_vir.kernel()
    # c_mo[:, nocc:nocc+3] = C

    # Running MP2 in NO basis
    eris = mp.mp2._make_eris(mymp, mo_coeff=c_mo)
    no_mp = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    mycc_reg = cc.CCSD(mf).run(verbose=0)
    # cisolver = fci.FCI(mf)
    # fci_res = cisolver.kernel()

    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {lo_mp[1]}")
    print(f"CCSD corr. energy            : {mycc_reg.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0]-mf.e_tot}")

    orbs = "MP-NO"  # 'localize' # 'localized'
    if orbs == "localized":
        return 3, 3, mf, c_lo, mymp, lo_mp[2], mf.e_tot, mymp.e_corr, mycc_reg.e_corr
    elif orbs == "MP-NO":
        return (
            3,
            3,
            mf,
            no_coeff,
            mymp,
            no_mp[2],
            mf.e_tot,
            mymp.e_corr,
            mycc_reg.e_corr,
        )
    else:
        return (
            3,
            3,
            mf,
            mf.mo_coeff,
            mymp,
            mymp.t2,
            mf.e_tot,
            mymp.e_corr,
            mycc_reg.e_corr,
        )


def ex(n_act_occ, n_act_virt, mf, mo_coeff, mp_t2, idx_s, idx_d):

    nocc = np.sum(mf.mo_occ > 0)
    nvirt = len(mf.mo_occ) - nocc

    act_hole = range(nocc - n_act_occ, nocc)
    act_particle = range(n_act_virt)

    # mycc = cc.rmpccsd_slow.RMPCCSD(mf, mo_coeff=c_mo)
    # res = mycc.kernel(act_hole , act_particle, idx_s, idx_d, t1 = np.zeros((nocc, nvirt)), t2 = lo_mp[2])

    mycc = cc.rmpccsd_slow.RMPCCSD(mf, mo_coeff=mo_coeff)
    mycc.verbose = 5
    res = mycc.kernel(
        act_hole, act_particle, idx_s, idx_d, t1=np.zeros((nocc, nvirt)), t2=mp_t2
    )

    print(f"MPCCSD corr. energy {mycc.e_corr}")

    return mycc.e_corr


def fragmented_mpcc(frags, mf, mo_coeff, mp_t2, idx_s, idx_d):

    nocc = np.sum(mf.mo_occ > 0)
    nvirt = len(mf.mo_occ) - nocc

    t1 = np.zeros((nocc, nvirt))
    t2 = np.copy(mp_t2)

    for frag in frags:
        act_hole, act_particle = frag[0], frag[1] - nocc
        mycc = cc.rmpccsd_slow.RMPCCSD(mf, mo_coeff=mo_coeff)
        mycc.verbose = 5
        res = mycc.kernel(act_hole, act_particle, idx_s, idx_d, t1=t1, t2=t2)

        t2 = mycc.t2
        t1 = mycc.t1

    # mymp = mp.MP2(mf).run(verbose = 0)
    # eris = mp.mp2._make_eris(mymp, mo_coeff=mo_coeff)
    # mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    # mp2_ref = mp.MP2(mf, mo_coeff = mo_coeff)
    cc_ref = cc.CCSD(mf, mo_coeff=mo_coeff)
    cc_ref.kernel()

    t2_new = np.copy(mp_t2)
    t2_new[
        np.ix_(frags[0][0], frags[0][0], frags[0][1] - nocc, frags[0][1] - nocc)
    ] = cc_ref.t2[
        np.ix_(frags[0][0], frags[0][0], frags[0][1] - nocc, frags[0][1] - nocc)
    ]
    t2_new[
            np.ix_(frags[1][0], frags[1][0], frags[1][1] - nocc, frags[1][1] - nocc)] = cc_ref.t2[
        np.ix_(frags[1][0], frags[1][0], frags[1][1] - nocc, frags[1][1] - nocc)
    ]

    t1_new = np.zeros((nocc, nvirt))
    t1_new[np.ix_(frags[0][0], frags[0][1] - nocc)] = cc_ref.t1[
        np.ix_(frags[0][0], frags[0][1] - nocc)
    ]
    t1_new[np.ix_(frags[1][0], frags[1][1] - nocc)] = cc_ref.t1[
        np.ix_(frags[1][0], frags[1][1] - nocc)
    ]

    print(cc_ref.energy(t1_new, t2_new))
    breakpoint()


def test_ring(N, bd, atom, basis, idx_s, idx_d):

    structure = ring.make(N, bd)
    atom = [(atom + " %f %f %f" % xyz) for xyz in structure]

    mol = gto.M()
    mol.basis = basis
    mol.atom = atom
    mol.verbose = 3
    mol.build()

    mf = mol.RHF().run()

    orbitals = mf.mo_coeff
    nocc = mol.nelectron // 2
    c_mo = np.zeros((orbitals.shape), orbitals.dtype)

    # orbitals = mf.mo_coeff[:,mf.mo_occ>0]
    pm_occ = lo.PM(mol, mf.mo_coeff[:, :nocc], mf)
    pm_occ.pop_method = "meta-lowdin"
    C = pm_occ.kernel()
    c_mo[:, :nocc] = C

    pm_vir = lo.PM(mol, mf.mo_coeff[:, nocc:], mf)
    pm_vir.pop_method = "meta-lowdin"
    C = pm_vir.kernel()
    c_mo[:, nocc:] = C

    # Building MP2 objects for the _make_eris routine
    mymp = mp.MP2(mf)  # , mo_coeff=c_mo)
    mymp.max_cycle = 30

    mymp.diis_space = 8
    # Running local MP2 should be conceptually very easy!

    eris = mp.mp2._make_eris(mymp, mo_coeff=c_mo)
    # breakpoint()
    lo_mp = mp.mp2._iterative_kernel(mymp, eris, verbose=5)

    # Running regular Mp2 and CCSD
    mymp = mp.MP2(mf).run()
    mycc_reg = cc.CCSD(mf).run()

    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"MP2 corr. energy (localized) : {lo_mp[1]}")
    print(f"Localized MP2 corr. energy   : {lo_mp[1]}")
    print(f"CCSD corr. energy            : {mycc_reg.e_corr}")

    # Visualizing T2
    viz = False
    if viz:
        nocc = mymp.t2.shape[0]
        nvirt = mymp.t2.shape[-1]

        mp_amp_oo_vv = mymp.t2.reshape((nocc ** 2, nvirt ** 2))
        cc_amp_oo_vv = mycc_reg.t2.reshape((nocc ** 2, nvirt ** 2))

        amp_diff = cc_amp_oo_vv - mp_amp_oo_vv

        breakpoint()

    mycc = cc.rmpccsd_slow.RMPCCSD(mf)

    # NOTE this is for H10
    act_hole = np.array([3, 4])
    # act_hole = np.array([1, 2, 3, 4])
    act_particle = np.array([0, 1])
    # act_particle = np.array([0, 1, 2, 3])

    # res = mycc.kernel(act_hole , act_particle, idx_s, idx_d, t1 = np.zeros((nocc, nocc)), t2 = lo_mp[2])
    res = mycc.kernel(
        act_hole, act_particle, idx_s, idx_d, t1=np.zeros((nocc, nocc)), t2=mymp.t2
    )

    print(f"MPCCSD corr. energy {mycc.e_corr}")

    breakpoint()


if __name__ == "__main__":

    np.set_printoptions(linewidth=280, suppress=True)
    idx_s = [[0, 1, 2], [2], []]
    idx_d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14],
        [3, 7, 9, 10, 11],
        [11],
    ]

    # (4,2)
    idx_s = [[0, 1, 2]]
    idx_d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

    # (2,1)
    # idx_s = [[2]]
    # idx_d = [[3, 7, 9, 10, 11]]

    # for elem_d in idx_d:
    #        for elem_s in idx_s:
    #            test_ring(10, 1.4, 'H', 'sto6g', elem_s, elem_d)
    #            print()
    # exit()

    # bds = [1.098, 1.2, 1.3, 1.4]
    # bds = np.linspace(1.0, 2.5, num=16)
    bds = np.linspace(1.0, 2.7, num=18)
    bds = [1.8]
    # bds = [2]

    res_hf = []
    res_mp = []
    res_cc = []
    res_mpcc = []
    for bd in bds:
        res_mpcc_d = []
        res_hf_d = []
        res_mp_d = []
        res_cc_d = []

        print("Bond length :", bd)

        # mf, lo_coeff, mp_t2, e_mf, e_mp, e_cc = get_reference_values("Be-ring", num_atoms = 10, basis="sto6g", bond_length=2.9, localization = 'AVAS', ao_labels = ['Be 2s'])

        # H8 model
        '''
        mf, lo_coeff, mp_t2, e_mf, e_mp, e_cc = get_reference_values(
            "H8",
            basis="ccpvdz",
            bond_length=0.741,
            dist=3,
            localization="AVAS",
            ao_labels=["H 1s"],
        )
        '''

        # N2 spin unrestricted
        mf, lo_coeff, mp_t2, e_mf, e_mp, e_cc = get_reference_values(
            "Cl2",
            basis="ccpvtz",
            bond_length=1.988,
            localization="AVAS-DMET",
            ao_labels = ["Cl 3p"],
            spin_restricted = True
        )

        # define fragmentation
        # fragments for MOs
        """
        act_hole_1 = [0,3]
        act_hole_2 = [1,2]
        act_part_1 = [4,7]
        act_part_2 = [5,6]
        """

        # fragments for IAO
        """
        act_hole_1 = [0,2]
        act_hole_2 = [1,3]
        act_part_1 = [5,7]
        act_part_2 = [4,6]
        """

        # fragments for AVAS ['H 1s']
        act_hole_1 = [0, 3]
        act_hole_2 = [1, 2]
        act_part_1 = [4, 7]
        act_part_2 = [5, 6]

        frags = [[act_hole_1, act_part_1], [act_hole_2, act_part_2]]

        # frags = [[act_hole_1 + act_hole_2 , act_part_1 + act_part_2]]

        # get_hring_references_PNO_avas(10, 'ccpvdz', bd)

        # n_act_occ, n_act_virt, mf, mo_coeff, mymp, mp_t2, e_mf, e_mp, e_cc = get_N2_references_PNO_avas('ccpvdz', bd)

        # n_act_occ, n_act_virt, mf, mo_coeff, mymp, mp_t2, e_mf, e_mp, e_cc = get_N2_references_PNO('ccpvdz', bd)

        # n_act_occ, n_act_virt, mf, mo_coeff, mymp, mp_t2, e_mf, e_mp, e_cc = get_N2_references('ccpvdz', bd)

        for elem_b in idx_d:
            res_mpcc_s = []
            res_hf_s = []
            res_mp_s = []
            res_cc_s = []

            for elem_s in idx_s:
                # output = ex(n_act_occ, n_act_virt, mf, mo_coeff, mp_t2, elem_s, elem_b)
                output = fragmented_mpcc(frags, mf, lo_coeff, mp_t2, elem_s, elem_b)
                res_mpcc_s.append(output)
                res_hf_s.append(e_mf)
                res_mp_s.append(e_mp)
                res_cc_s.append(e_cc)

            res_mpcc_d.append(res_mpcc_s)
            res_hf_d.append(res_hf_s)
            res_mp_d.append(res_mp_s)
            res_cc_d.append(res_cc_s)

        res_mpcc.append(res_mpcc_d)
        res_hf.append(res_hf_d)
        res_mp.append(res_mp_d)
        res_cc.append(res_cc_d)

    res_hf = np.array(res_hf)
    res_mp = np.array(res_mp)
    res_cc = np.array(res_cc)
    res_mpcc = np.array(res_mpcc)

    # plot results
    plt.plot(bds, res_hf[:, 0, 0], label="HF")
    plt.plot(bds, res_hf[:, 0, 0] + res_mp[:, 0, 0], label="MP2")
    plt.plot(bds, res_hf[:, 0, 0] + res_cc[:, 0, 0], label="CCSD")
    plt.plot(bds, res_hf[:, 0, 0] + res_mpcc[:, 0, 0], label="(4,2)")
    # plt.plot(bds, res_hf[:,0,0] + res_mpcc[:,2,1],label = '(2,1)' )
    # plt.plot(bds, res_hf[:,0,0] + res_mpcc[:,3,1],label = '(1,1)' )
    plt.legend()
    plt.show()

    breakpoint()
