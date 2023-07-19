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
#        avas_obj = avas.AVAS(mf, ao_labels, minao="dz", openshell_option=2)
        avas_obj = avas.AVAS(mf, ao_labels, minao="dz")
#       avas_obj = avas.AVAS(mf, ao_labels)
        avas_obj.with_iao = True
        ncas, nelecas, mocas = avas_obj.kernel()
 
#        act_hole = np.where(avas_obj.occ_weights > 1e-10)[0]
#        act_part = np.where(avas_obj.vir_weights >avas_obj. threshold)[0] + mf.mol.nelectron//2

        print(ncas, nelecas)
        print(avas_obj.occ_weights) 
        print(avas_obj.vir_weights) 

        act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        act_part = np.where(avas_obj.vir_weights >avas_obj.threshold)[0] + mf.mol.nelec[0]
       
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


def get_reference_values(basis, bond_length, localization, ao_labels, spin_restricted):

    mol = gto.M()
    mol.basis = basis
    mol.atom = [['N', [0,0,0]] , ['N', [bond_length,0,0]]]
    mol.verbose = 3
    mol.unit = "angstrom"
    mol.spin = 0
    mol.build()

    if spin_restricted:
        mf = mol.RHF().newton()
        mf = mf.run()
        mo1 = mf.stability()[0]

        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1 = mf.stability()[0]
        mf = mf.newton().run(mo1, mf.mo_occ)
        mf.stability()
    else:
#       mf = mol.UHF().run()
        mf = mol.ROHF().run()

        nocc = np.sum(mf.mo_occ > 0)
        ndocc = np.sum(mf.mo_occ > 1)
        nsocc = nocc - ndocc

    # localization
    name = f"molden/molden_N2_{basis}_{bond_length}"
    c_lo, act_hole, act_part = get_localized_orbs(
        mol, mf, localization, ao_labels, molden_bool=True, name=name
    )

    print("active hole", act_hole)
    print("active particle", act_part)

#   c_lo = mf.mo_coeff
#   act_hole = 3
#   act_part = 3

    if spin_restricted:
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mycc = cc.CCSD(mf).run(verbose=0)

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    else:
        mf1 = scf.addons.convert_to_uhf(mf)
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf1).run()
        mycc = cc.UCCSD(mf1).run()

        # Running localized MP2
        eris = mp.ump2._make_eris(mymp, mo_coeff=(c_lo,c_lo))
#       eris = mp.ump2._make_eris(mymp)
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, verbose=0)

       
    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

    return mf, c_lo, act_hole, act_part, mp_lo[2], mf.e_tot, mymp.e_corr, mycc.e_corr


def fragmented_mpcc(frags, mf, mo_coeff, mp_t2, idx_s, idx_d):

    nocc = np.sum(mf.mo_occ > 0)
    nvirt = len(mf.mo_occ) - nocc

    t1 = np.zeros((nocc, nvirt))
    t2 = np.copy(mp_t2)
    
    for frag in frags:
        act_hole, act_particle = frag[0], frag[1] - nocc

        print("active hole")
        print(act_hole)
        print("=============")
        print(frag)

        mycc = cc.rmpccsd_slow.RMPCCSD(mf, mo_coeff=mo_coeff)
#        mycc = cc.umpccsd.CCSD(mf, mo_coeff=mo_coeff)
        mycc.verbose = 5
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


def fragmented_mpcc_unrestricted(frags, mf, mo_coeff, mp_t2, idx_s, idx_d):

    t2aa, t2ab, t2bb = mp_t2

    nocca, noccb, nvira, nvirb = t2ab.shape

    t1a = np.zeros((nocca, nvira))
    t1b = np.zeros((noccb, nvirb))
    t2 = np.copy(mp_t2)
    t1 = t1a, t1b 
    
    for frag in frags:
        act_hole = [frag[0][0],  frag[0][1]]
        act_particle = [frag[1][0] - nocca, frag[1][1] - noccb]
        mycc = cc.umpccsd.CCSD(mf, mo_coeff=(mo_coeff, mo_coeff))
        mycc.verbose = 5
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
#    idx_s = [[0, 1, 2],[0, 1, 2]]
#    idx_d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

    idx_s = [0, 1, 2]
    idx_d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # (2,1)
    # idx_s = [[2]]
    # idx_d = [[3, 7, 9, 10, 11]]

#    bds = np.linspace(1.2, 3.5, num=16)
    bds = np.arange(1.2, 4.2, 0.2)
    
    res_hf = []
    res_mp = []
    res_cc = []
    res_mpcc = []
    res_cclo = []
    for bd in bds:
        print("Bond length :", bd)
        basis = "aug-ccpvdz"
        ao_labels = ["N 2p", "N 2s", "N 3p", "N 3s"]
#        ao_labels = ["N 2p"]

        mf, lo_coeff, act_hole, act_part, mp_t2, e_mf, e_mp, e_cc = get_reference_values(
            basis=basis,
            bond_length=bd,
            localization="AVAS",
#            localization="meta-lowdin",
#            localization="MO",
            ao_labels = ao_labels,
            spin_restricted = True
        )

        frag = [[act_hole, act_part]]
#        frag = [act_hole, act_part]

#        e_mpcc, e_ccsd, e_ccsd_lo = fragmented_mpcc_unrestricted(frag, mf, lo_coeff, mp_t2, idx_s, idx_d)
        e_mpcc, e_ccsd, e_ccsd_lo = fragmented_mpcc(frag, mf, lo_coeff, mp_t2, idx_s, idx_d)

        res_hf.append(e_mf)
        res_mp.append(e_mp)
        res_cc.append(e_ccsd)
        res_mpcc.append(e_mpcc)
        res_cclo.append(e_ccsd_lo)


    res_hf = np.array(res_hf)
    res_mp  = np.array(res_mp)
    res_cc = np.array(res_cc)
    res_mpcc = np.array(res_mpcc)
    res_cclo =  np.array(res_cclo)

#    breakpoint()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(bds, res_hf, label= 'RHF')
    ax1.plot(bds, res_hf + res_mp, label= 'MP2')
    ax1.plot(bds, res_hf + res_cc, label= 'CCSD')
    ax1.plot(bds, res_hf + res_mpcc, label= 'MP-CCSD (4,2)')
    ax1.plot(bds, res_hf + res_cclo, label= 'Glob. CCSD in (4,2)')

    ax1.legend()
    ax1.set_title(f'Absolute energies ({basis}, {ao_labels})', pad=20)
    ax1.set_xlabel('Bond length')

    ax2.plot(bds, res_hf- np.min(res_hf), label= 'RHF')
    #ax2.plot(bds, res_hf + res_mp - np.min(res_hf + res_mp), label= 'MP2')
    ax2.plot(bds, res_hf + res_cc - np.min(res_hf + res_cc), label= 'CCSD')
    ax2.plot(bds, res_hf + res_mpcc - np.min(res_hf + res_mpcc ), label= 'MP-CCSD (4,2)')
    ax2.plot(bds, res_hf + res_cclo - np.min(res_hf + res_cclo), label= 'Glob. CCSD in (4,2)')

    ax2.legend()
    ax2.set_title(f'Relative energies ({basis}, {ao_labels})', pad=20)
    ax2.set_xlabel('Bond length')


#    name = f"plots/N2/abs_energies_{basis}"
    name = f"abs_energies_{basis}"
    for elem in ao_labels:
        name += f"_{elem}"
    name += ".png"
    fig.savefig(name,
                bbox_inches="tight",
                dpi= 150,
                transparent= True,
                facecolor= 'w',
                edgecolor= 'w',
                orientation ='landscape')

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(bds, res_cc - res_mp, label= 'MP2')
    ax1.plot(bds, res_cc - res_mpcc, label= 'MP-CCSD (4,2)')
    ax1.plot(bds, res_cc - res_cclo, label= 'Glob. CCSD in (4,2)')

    ax1.legend()
    ax1.set_title(f'Corr. energy diff. ({basis}, {ao_labels})', pad=20)
    ax1.set_xlabel('Bond length')


#    name = f"plots/N2/corr_energy_diff_{basis}"
    name = f"corr_energy_diff_{basis}"
    for elem in ao_labels:
        name += f"_{elem}"
    name += ".png"

    fig.savefig(name,
                bbox_inches="tight",
                dpi= 150,
                transparent= True,
                facecolor= 'w',
                edgecolor= 'w',
                orientation ='landscape')

    plt.show()


    breakpoint()


