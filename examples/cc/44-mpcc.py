'''
    Vanilla version of mp2-ccsd
'''

import numpy as np
from pyscf import ao2mo, gto, scf, mp, cc, lo
from pyscf.tools import ring
import matplotlib.pyplot as plt

def get_t_int(t2, eps = 0.9):

    max_val = np.max(np.abs(t2))
    idx = np.where(np.abs(t2)> eps * max_val)
    idx = np.asarray(idx)
    return [idx[:,i] for i in range(len(idx[0,:]))]

def ex(basis, bd, idx_s, idx_d):
    
    
    mol = gto.M()
    mol.basis = basis
    mol.atom  = [['N', [0,0,0]] , ['N', [bd,0,0]]]
    mol.verbose = 3
    mol.build()

    mf = mol.RHF().run()
    mymp = mp.MP2(mf).run()
    mycc_reg = cc.CCSD(mf).run() 

    print(f"MP2 corr. energy  {mymp.e_corr}")   
    print(f"CCSD corr. energy {mycc_reg.e_corr}")
    
    #t_int = get_t_int(mymp.t2)
   
    mycc = cc.rmpccsd_slow.RMPCCSD(mf)
    act_hole = np.array([4, 5, 6])
    act_particle = np.array([0, 1, 2])
    
    res = mycc.kernel(act_hole , act_particle, idx_s, idx_d)
    
    print(f"MPCCSD corr. energy {mycc.e_corr}")

    return mf.e_tot, mymp.e_corr, mycc_reg.e_corr,  mycc.e_corr


def test_ring(N, bd, atom, basis, idx_s, idx_d):
    
    structure = ring.make(N, bd)
    atom = [(atom +' %f %f %f' % xyz) for xyz in structure]
    
    mol = gto.M()
    mol.basis = basis
    mol.atom  = atom
    mol.verbose = 3
    mol.build()

    mf = mol.RHF().run()

    orbitals = mf.mo_coeff
    pm = lo.PM(mol, orbitals, mf)
    pm.pop_method = 'iao'  
    C = pm.kernel()
    
    # comparison objects
    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    fock_ao = mf.get_fock(vhf=vhf, dm=dm)
    fock_lo = C.T @ fock_ao @ C
    fock_mo = mf.mo_coeff.T @ fock_ao @ mf.mo_coeff
 
    '''
    # running localized computations
    norb = mol.nao_nr() 
    h1e0 = mol.intor("int1e_nuc_sph") + mol.intor("int1e_kin_sph")
    eri0 = mol.intor("int2e_sph")
 
    mf_lo = scf.RHF(mol)
    h1e = C.T @ h1e0 @ C
    eri = ao2mo.incore.full(eri0, C)
    
    mf_lo.get_hcore = lambda *args: h1e
    mf_lo.get_ovlp = lambda *args: np.eye(norb)
    mf_lo._eri = ao2mo.restore(8, eri, norb)
    mf_lo.kernel()

    dm_lo = mf_lo.make_rdm1()
    vhf_lo = mf_lo.get_veff(mol, dm_lo)
    fock_ao_lo = mf.get_fock(vhf=vhf_lo, dm=dm_lo)
    fock_lo_lo = mf_lo.mo_coeff.T @ fock_ao_lo @ mf_lo.mo_coeff
    '''

    # Building MP2 objects for the _make_eris routine
    mymp = mp.MP2(mf)
    mymp.max_cycle = 200
    
    #Running local MP2 should be conceptually very easy!
    eris = mp.mp2._make_eris(mymp, mo_coeff= C)
    breakpoint()
    lo_mp = mp.mp2._iterative_kernel(mymp, eris, verbose=5) 

    # Running regular Mp2 and CCSD 
    mymp = mp.MP2(mf).run()
    mycc_reg = cc.CCSD(mf).run() 

    print(f"MP2 corr. energy          : {mymp.e_corr}")   
    print(f"Localized MP2 corr. energy: {lo_mp[1]}")
    print(f"CCSD corr. energy         : {mycc_reg.e_corr}")
    breakpoint()  

    # Visualizing T2
    viz = False
    if viz:
        nocc = mymp.t2.shape[0]
        nvirt = mymp.t2.shape[-1]

        mp_amp_oo_vv = mymp.t2.reshape((nocc**2, nvirt**2))
        cc_amp_oo_vv = mycc_reg.t2.reshape((nocc**2, nvirt**2))

        amp_diff = cc_amp_oo_vv - mp_amp_oo_vv

        breakpoint()

    #t_int = get_t_int(mymp.t2)
   
    mycc = cc.rmpccsd_slow.RMPCCSD(mf)
    
    # NOTE this is for H10
    #act_hole = np.array([3, 4])
    act_hole = np.array([1, 2, 3, 4])
    #act_particle = np.array([0, 1])
    act_particle = np.array([0, 1, 2, 3]) 

    res = mycc.kernel(act_hole , act_particle, idx_s, idx_d, t1 = np.zeros((nocc, nocc)), t2 = lo_mp[2])
    
    print(f"MPCCSD corr. energy {mycc.e_corr}")

    breakpoint()

if __name__ == "__main__":

    np.set_printoptions(linewidth = 280, suppress = True)
    idx_s = [[0,1,2], [2], []]
    idx_d = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
             [1,2,3,5,6,7,8,9,10,11,14],
             [3,7,9,10,11],
             [11]
            ]

    for elem_d in idx_d:
            for elem_s in idx_s:
                test_ring(10, 1.4, 'H', 'sto6g', elem_s, elem_d)    
                print()

    exit()

    bds = [1.098, 1.2, 1.3]

    res_hf = []
    res_mp = []
    res_cc = []
    res_mpcc = []
    for bd in bds:
        res_mpcc_d = []
        res_hf_d = []
        res_mp_d = []
        res_cc_d = []
 
        for elem_b in idx_d:
            res_mpcc_s = []
            res_hf_s = []
            res_mp_s = []
            res_cc_s = []
     
            for elem_s in idx_s:
                output = ex('ccpvdz', bd, elem_s, elem_b)
                res_mpcc_s.append(output[3])
                res_hf_s.append(output[0])
                res_mp_s.append(output[1])
                res_cc_s.append(output[2])
     
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
 
    breakpoint()

