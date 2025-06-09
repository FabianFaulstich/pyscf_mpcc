from pyscf import gto, scf, cc
from pyscf.mp.dfmp2_native import DFMP2

from pyscf import mpcc

if __name__ == "__main__":

    mol = gto.Mole()
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.757, 0.587)],
        [1, (0.0, 0.757, 0.587)],
    ]
    mol.basis = "cc-pvdz"
    mol.build()

    mf = scf.RHF(mol).density_fit().run()
    mymp = DFMP2(mf).run()
    #mycc = cc.CCSD(mf).density_fit().run()

    # No computation
    mympcc = mpcc.MPCC(mf)

    # NOTE setting the following variables explicitly to the default value
    # This can also be passed through mpcc.MPCC(mf, 'arg'= value)
    # mympcc = mpcc.MPCC(mf, ll_con_tol = 1e-6, ll_max_its = 50)
    
    mympcc.lowlevel.ll_max_its = 50
    mympcc.lowlevel.ll_con_tol = 1e-6


    mympcc.kernel()

    print("Left MPCC")
    print(mympcc.lowlevel.e_tot)
    breakpoint()
    # localization, where?
    # a-a, i-a do this in ERIs
    # 

    # NOTE what we want:
    #mympcc.lowlevel set #its, tol, 
    #   "ll method" set this RPA here 


