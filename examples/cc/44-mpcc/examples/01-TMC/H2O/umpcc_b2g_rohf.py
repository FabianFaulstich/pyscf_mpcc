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
        avas_obj = avas.AVAS(mf, ao_labels, openshell_option=3)
#        avas_obj = avas.AVAS(mf, ao_labels, minao="6-31g")
#        avas_obj = avas.AVAS(mf, ao_labels)
#        avas_obj = avas.AVAS(mf, ao_labels, minao="6-31g", openshell_option=4)
        avas_obj.with_iao = True
        ncas, nelecas, mocas = avas_obj.kernel()

        act_hole = (np.where(avas_obj.occ_weights[0] > avas_obj.threshold)[0], np.where(avas_obj.occ_weights[1] > avas_obj.threshold)[0])
        act_part = (np.where(avas_obj.vir_weights[0] >avas_obj.threshold)[0] + mf.mol.nelec[0], np.where(avas_obj.vir_weights[1] >avas_obj.threshold)[0] + mf.mol.nelec[1])
       
#       c_lo = avas_obj.mo_coeff
        c_lo = mocas

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
        act_hole = 3
        act_part = 3

    elif localization == "MO":
        c_lo = mf.mo_coeff
        act_hole = 3
        act_part = 3

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
        

    return c_lo, act_hole, act_part


def get_reference_values(basis, localization, ao_labels, spin_restricted, dm0=None):

    Mol1 = [
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
    mol.atom      = Mol1
    mol.unit      = 'Bohr'
    mol.verbose   = 3
    mol.nelectron = 67 # 69 = 29 + 4*10  
    mol.spin      = 1
    mol.charge    = 2
    mol.symmetry  = True
    mol.build()
 
    if spin_restricted:
        mf = mol.ROHF().newton()
    else:
        mf = mol.UHF().newton()

    mf.irrep_nelec = {'Ag': (11, 10), 'B1g': (3, 3), 'B2g': (2, 2), 'B3g': (2, 2), 'B1u': (4, 4), 'B2u': (6, 6), 'B3u': (6, 6)}

    mf.run()
    mf.stability()
    mf.analyze()
    mfe = mf.e_tot
    print(f'Energy: {mfe}')

    # localization
    name = f"molden/molden_Cu_H2O_4_2+_{basis}"
    c_lo, act_hole, act_part = get_localized_orbs(
        mol, mf, localization, ao_labels, molden_bool=False, name=name
    )

    print("active hole", act_hole)
    print("active particle", act_part)

    if spin_restricted:
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()
        mycc = cc.UCCSD(mf).run()
        mycc.diis = 12
        mycc.max_cycle = 80

        # Running localized MP2
        eris = mp.ump2._make_eris(mymp, mo_coeff=(c_lo, c_lo))
#        eris = mp.ump2._make_eris(mymp)
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, verbose=4)
 

    else:
        mult, sz = mf.spin_square()
        print("spin multiplicity", sz)
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()
        mycc = cc.UCCSD(mf).run()
        mymp.diis = 12
        mycc.diis = 12
        mycc.max_cycle = 80
        mymp.max_cycle = 80

        # Running localized MP2
        eris = mp.ump2._make_eris(mymp, mo_coeff=c_lo)
#       eris = mp.ump2._make_eris(mymp)
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, verbose=4)
       
    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

#    return mf1, (c_lo,c_lo), act_hole, act_part, mp_lo[2], mp_lo[3], mf.e_tot, mymp.e_corr, mycc.e_corr
    return mf, c_lo, act_hole, act_part, mp_lo[2], mp_lo[3], mf.e_tot, mymp.e_corr, mycc.e_corr

def fragmented_mpcc(frags, mf, mo_coeff, mp_t2, idx_s, idx_d):

    nocc = np.sum(mf.mo_occ > 0)
    nvirt = len(mf.mo_occ) - nocc

    t1 = np.zeros((nocc, nvirt))
    t2 = np.copy(mp_t2)
    
    for frag in frags:
        act_hole, act_particle = frag[0], frag[1] - nocc
        mycc = cc.rmpccsd_slow.RMPCCSD(mf, mo_coeff=mo_coeff)
#        mycc = cc.umpccsd.CCSD(mf, mo_coeff=mo_coeff)
        mycc.verbose = 4
        res = mycc.kernel(act_hole, act_particle, idx_s, idx_d, t1=t1, t2=t2)

        t2 = mycc.t2
        t1 = mycc.t1

    # mymp = mp.MP2(mf).run(verbose = 0)
    # eris = mp.mp2._make_eris(mymp, mo_coeff=mo_coeff)
    # mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    # mp2_ref = mp.MP2(mf, mo_coeff = mo_coeff)
    cc_ref = cc.CCSD(mf, mo_coeff=mo_coeff)
    cc_ref.verbose = 0
    cc_ref.kernel()
    print('Global CCSD :', cc_ref.e_corr)

    t2_new = np.copy(mp_t2)
    t2_new[
        np.ix_(frags[0][0], frags[0][0], frags[0][1] - nocc, frags[0][1] - nocc)
    ] = cc_ref.t2[
        np.ix_(frags[0][0], frags[0][0], frags[0][1] - nocc, frags[0][1] - nocc)
    ]

    t1_new = np.zeros((nocc, nvirt))
    t1_new[np.ix_(frags[0][0], frags[0][1] - nocc)] = cc_ref.t1[
        np.ix_(frags[0][0], frags[0][1] - nocc)
    ]

    e_ccsd_lo = cc_ref.energy(t1_new, t2_new)
    print('Exact localized CCSD in MPCCSD (4,2) :', e_ccsd_lo)

    return res[0], cc_ref.e_corr, e_ccsd_lo 

def fragmented_mpcc_unrestricted(frags, mf, mo_coeff, mp_t2, mp_t1, idx_s, idx_d):

    t2aa, t2ab, t2bb = mp_t2

    nocca, noccb, nvira, nvirb = t2ab.shape

    t1a, t1b = mp_t1 
    t2 = np.copy(mp_t2)
    t1 = t1a, t1b 

    for frag in frags:
        act_hole = [frag[0][0],  frag[0][1]]
        act_particle = [frag[1][0] - nocca, frag[1][1] - noccb]
        mycc = cc.umpccsd.CCSD(mf, mo_coeff=mo_coeff)
        mycc.verbose = 5
        mycc.max_cycle = 80
        mycc.diis_space = 14
#        mycc.level_shift = .5
        res = mycc.kernel(act_hole, act_particle, idx_s, idx_d, t1=t1, t2=t2)
        print('Exact Exact localized CCSD in MPCCSD (4,2) :', res[0])

        t2 = mycc.t2
        t1 = mycc.t1

    # mymp = mp.MP2(mf).run(verbose = 0)
    # eris = mp.mp2._make_eris(mymp, mo_coeff=mo_coeff)
    # mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    # mp2_ref = mp.MP2(mf, mo_coeff = mo_coeff)

#   t2_new = np.copy(mp_t2)
#   t2_new[
#       np.ix_(frags[0][0], frags[0][0], frags[0][1] - nocc, frags[0][1] - nocc)
#   ] = cc_ref.t2[
#       np.ix_(frags[0][0], frags[0][0], frags[0][1] - nocc, frags[0][1] - nocc)
#   ]
#   t1_new = np.zeros((nocc, nvirt))
#   t1_new[np.ix_(frags[0][0], frags[0][1] - nocc)] = cc_ref.t1[
#       np.ix_(frags[0][0], frags[0][1] - nocc)
#   ]
    return res[0] 


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
    # this shold be just one entry
    idx_s = [[0, 1, 2],[0, 1, 2]]
    idx_d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

    # (2,1)
    # idx_s = [[2]]
    # idx_d = [[3, 7, 9, 10, 11]]

    basis = "631g"
    ao_labels = ["Cu 3d"]

    mf, lo_coeff, act_hole, act_part, mp_t2, mp_t1, e_mf, e_mp, e_cc = get_reference_values(
     basis=basis,
     localization="AVAS",
#    localization="meta-lowdin",
#    localization="MO",
     ao_labels = ao_labels,
     spin_restricted = True,
    )

    frag = [[act_hole, act_part]]

    e_mpcc = fragmented_mpcc_unrestricted(frag, mf, lo_coeff, mp_t2, mp_t1, idx_s, idx_d)

    print("Total Energy: ", (e_mf+e_mpcc))
