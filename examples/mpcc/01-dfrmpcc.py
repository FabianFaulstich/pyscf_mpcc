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

    mympcc = mpcc.MPCC(mf)
    breakpoint()
