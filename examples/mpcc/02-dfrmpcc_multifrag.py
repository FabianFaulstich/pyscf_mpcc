from pyscf import gto, scf, mp, cc
from pyscf.mp.dfmp2_native import DFMP2

from pyscf.mcscf import avas
from pyscf.data.elements import chemcore

import numpy as np

from pyscf import mpcc
import sys

rose_dir = '/Users/avijitshee/rose_corr/python_scripts'
sys.path.append(rose_dir)


if __name__ == "__main__":

    mol = gto.Mole()
    mol.atom = '''
                H 1.00268965422152 0.41260814802418  0.00000615157015
                F 1.88614457314300 0.13546434067777 -0.00000615157071
                H 1.79731030186484 2.38739171120797  0.00000615156678
                F 0.91385547077064 2.66453580009008 -0.00000615156623
               '''

    mol.basis = {'F':"aug-ccpvdz",'H':"aug-ccpvdz"} 
    mol.cart = True
    mol.build()

    mf = scf.RHF(mol).density_fit().run()
    mf.threshold = 1e-7

    # Orbital localization    
    
#    ncore = chemcore(mol)
    ncore = 0
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]


    def get_IFOS(xyz_frags, charge_frags, mult_frags):
       from pyscf_ifo import run_calculations

       large_basis = 'aug-ccpvdz'
       small_basis = 'ccpvdz'
   
       c_lo_energies, c_lo = run_calculations(xyz_frags, charge_frags, mult_frags, large_basis, small_basis)  

       act_hole_0 = (np.arange(5)) 
       act_part_0 = (np.arange(10,25))

       act_hole_1 = (np.arange(5,10)) 
       act_part_1 = (np.arange(25,40))

       return c_lo, act_hole_0, act_part_0, act_hole_1, act_part_1


    xyz_frags = ['hf_dimer', 'hf_frag0', 'hf_frag1']
    charge_frags = [0, 0, 0]
    mult_frags = [1, 1, 1]

    c_lo, act_hole_0, act_part_0, act_hole_1, act_part_1 = get_IFOS(xyz_frags, charge_frags, mult_frags)

    mymp = DFMP2(mf).run()
#   mycc = cc.CCSD(mf).density_fit().run()

    frag_info = {'frag': [[act_hole_0, act_part_0],[act_hole_1, act_part_1]]}

    conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}

    kwargs = frag_info|conv_info  #union operation

    # No computation
    mympcc = mpcc.RMPCC(mf, 'True', c_lo, **kwargs)

    # NOTE setting the following variables explicitly to the default value
    # This can also be passed through mpcc.MPCC(mf, 'arg'= value)
    # mympcc = mpcc.MPCC(mf, ll_con_tol = 1e-6, ll_max_its = 50)
    
    #mympcc.lowlevel.ll_max_its = 50
    #mympcc.lowlevel.ll_con_tol = 1e-6
 

#   mympcc.MPCC.frag = frag

    # localization input
    # 
#   mympcc.kernel(localization = True, )
    mympcc.kernel()

    print("Quit MPCC")
    print(mympcc.lowlevel.e_tot)
    # localization, where?
    # a-a, i-a do this in ERIs
    # 

    # NOTE what we want:
    #mympcc.lowlevel set #its, tol, 
    #   "ll method" set this RPA here 


