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

        avas_obj = avas.AVAS(mf, ao_labels, minao="dz",openshell_option=3)
 #       avas_obj = avas.AVAS(mf, ao_labels, openshell_option=3)
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


def get_reference_values(basis, bond_length, localization, ao_labels, spin_restricted, spin, dm0=None):

    mol = gto.M()
    mol.basis = basis
    mol.atom = [['N', [0,0,0]] , ['N', [bond_length,0,0]]]
    mol.verbose = 3
    mol.unit = "angstrom"
    mol.spin = spin
    mol.build()

    if spin_restricted:
        mf = mol.RHF().run()
    else:
#        mf = mol.UHF(dm0)
        mf = mol.UHF().newton()

     #  mf = mol.UHF().newton()
        mf = mf.run()
        mo1 = mf.stability()[0]

        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1 = mf.stability()[0]

        mf = mf.newton().run(mo1, mf.mo_occ)
        mf.stability()

#       mf.verbose = 5
#       mf = mf.run(dm0)

    if spin_restricted:
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mycc = cc.CCSD(mf).run(verbose=0)

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, verbose=0)

    else:
#        mf1 = scf.addons.convert_to_uhf(mf)
        mult, sz = mf.spin_square()
        print("spin multiplicity", sz)
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()
        mycc = cc.UCCSD(mf).run()
        mycc.diis = 12
        mycc.level_shift = .05

#        eris = mp.ump2._make_eris(mymp, mo_coeff=(c_lo,c_lo))
        eris = mp.ump2._make_eris(mymp)
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, verbose=0)
       
    print(f"MP2 corr. energy             : {mymp.e_corr}")
    print(f"Localized MP2 corr. energy   : {mp_lo[1]}")
    print(f"CCSD corr. energy            : {mycc.e_corr}")
    # print(f"FCI corr. energy             : {fci_res[0] - mf.e_tot}")

    return mf, mf.e_tot, mp_lo[1], mycc.e_corr, sz

if __name__ == "__main__":

    np.set_printoptions(linewidth=280, suppress=True)

#   bds = np.linspace(1.2, 3.5, num=10)
    bds = np.arange(1.2, 4.0, 0.1)
    spins = [0, 2, 4, 6]
    basis = "aug-ccpvdz"
   
    nr_spin_sectors = len(spins) 
    nr_bond_lengths = len(bds)
 
    res_hf = np.zeros((nr_spin_sectors,nr_bond_lengths), float)
    res_hf_sz = np.zeros((nr_spin_sectors,nr_bond_lengths), float)
    res_mp = np.zeros((nr_spin_sectors,nr_bond_lengths), float)
    res_cc = np.zeros((nr_spin_sectors,nr_bond_lengths), float)
    i = 0
    dm0 = None 
    for spin in spins:
        i += 1
        j = 0 
        for bd in bds:
            print("Bond length, spin :", bd, spin)
            j += 1
            if bd == bds[0]:
                mf, e_mf, e_mp, e_cc, sz = get_reference_values(
                basis=basis,
                bond_length=bd,
                localization="AVAS",
#               localization="meta-lowdin",
#               localization="MO",
                ao_labels = ao_labels,
                spin_restricted = False,
                spin = spin
                )
            else:
                mf, e_mf, e_mp, e_cc, sz = get_reference_values(
                basis=basis,
                bond_length=bd,
                localization="AVAS",
#               localization="meta-lowdin",
#               localization="MO",
                ao_labels = ao_labels,
                spin_restricted = False,
                spin = spin,
                dm0 = dm0 
                )
         
                dm0 = mf.make_rdm1()

            res_hf[i-1,j-1]  = e_mf
            res_mp[i-1,j-1]  = e_mp
            res_cc[i-1,j-1]  = e_cc
            res_hf_sz[i-1,j-1]  = sz

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    for s in range(len(spins)): 
        spin = spins[s]//2
        ax1.plot(bds, res_hf[s,:], label= f'UHF_spin={spin}')

    ax1.legend()
    ax1.set_title(f'Absolute energies UHF ({basis})', pad=20)
    ax1.set_xlabel('Bond length')

    for s in range(len(spins)): 
        spin = spins[s]//2
        ax2.plot(bds, res_hf[s,:] + res_mp[s,:], label= f'UMP2_spin={spin}')

    ax2.legend()
    ax2.set_title(f'Absolute energies UMP2 ({basis})', pad=20)
    ax2.set_xlabel('Bond length')

    for s in range(len(spins)): 
        spin = spins[s]//2
        ax3.plot(bds, res_hf[s,:] + res_cc[s,:], label= f'UCCSD_spin={spin}')

    ax3.legend()
    ax3.set_title(f'Absolute energies UCCSD ({basis})', pad=20)
    ax3.set_xlabel('Bond length')

    name = f"abs_energies_{basis}"
#    for elem in ao_labels:
#        name += f"_{elem}"
#    name += f"_{spin}"
    name += ".png"

    print(name)

    fig.savefig(name,
                bbox_inches="tight",
                dpi= 150,
                transparent= True,
                facecolor= 'w',
                edgecolor= 'w',
                orientation ='landscape')

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

    for s in range(len(spins)): 
        spin = spins[s]//2
        ax1.plot(bds,res_hf_sz[s,:], label= f'<S2>={spin}')

#    ax1.plot(bds, res_cc - res_mp, label= 'UMP2')
#   ax1.plot(bds, res_cc - res_mpcc, label= 'UMP-CCSD (4,2)')

    ax1.legend()
    ax1.set_title(r'UHF $\langle S^2 \rangle$', pad=20)
    ax1.set_xlabel('Bond length')

    plt.show()

    breakpoint()
