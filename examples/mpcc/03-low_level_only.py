from pyscf import gto, scf, cc
from pyscf import mpcc

import time

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
    c_lo = mf.mo_coeff
 
    frag_info = {'frag': [[[0], [0]]]}
    conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
    kwargs = frag_info|conv_info  #union operation

    mympcc = mpcc.RMPCC(mf, 'True', c_lo, **kwargs)
  
    print('Initializing ...')
    # Initializing the input for low-level solver
    st = time.time()
    mycc = cc.CCSD(mf)
    mycc.max_cycle = 6
    mycc.kernel()
    _, _, Y = mympcc.lowlevel.init_amps()
    print(f'Done! Elapsed time: {time.time() - st} sec')


    # Running low-level solver
    print('Starting Low-Level Solver')
    t1, t2, Y = mympcc.lowlevel.kernel(mycc.t1, [0], Y)
    print('Finished Low-Level solver!')


