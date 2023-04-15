'''
    Vanilla version of mp2-ccsd
'''

import numpy as np
from pyscf import gto, scf, mp, cc

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

if __name__ == "__main__":


    idx_s = [[0,1,2], [2], []]
    idx_d = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
             [1,2,3,5,6,7,8,9,10,11,14],
             [3,7,9,10,11],
             [11]
            ]

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

