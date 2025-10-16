from pyscf import gto, scf, mp, cc
from pyscf.mp.dfmp2_native import DFMP2

from pyscf.mcscf import avas
from pyscf.data.elements import chemcore

import numpy as np

from pyscf import mpcc


####### Matrix free Choleski experiment #######

def MFPC(n, diag_fn, col_fn, max_rank=None, tol=None):
    """
    Matrix-free Pivoted Cholesky.
    """

    if max_rank is None: max_rank = n
    p = np.arange(n)
    L = np.zeros((n, max_rank), dtype=float)
    d = np.maximum(diag_fn(), 0.0).copy()  # residual diagonal
    r = 0

    for k in range(max_rank):
        # choose pivot
        j = k + np.argmax(d[k:])
        if j != k:
            p[[k, j]] = p[[j, k]]
            d[[k, j]] = d[[j, k]]
            L[[k, j], :k] = L[[j, k], :k]

        alpha2 = max(d[k], 0.0)
        if (tol is not None and np.sqrt(alpha2) <= tol) or alpha2 == 0.0:
            break

        alpha = np.sqrt(alpha2)
        L[k, k] = alpha

        if k < n - 1:
            # full column in original indexing, then take permuted tail
            col_full = col_fn(p[k])          # shape (n,)
            a_tail = col_full[p[k+1:]]       # shape (n-k-1,)
            # Schur update
            if k > 0:
                schur = L[k+1:, :k] @ L[k, :k]
                w = (a_tail - schur) / alpha
            else:
                w = a_tail / alpha
            L[k+1:, k] = w
            d[k+1:] = np.maximum(d[k+1:] - w**2, 0.0)

        r += 1

    return L[:, :r], p, r


def PC(A, max_rank=None, tol=None):

    n = A.shape[0]
    if max_rank is None:
        max_rank = n

    p = np.arange(n)
    L = np.zeros((n, max_rank), dtype=A.dtype)

    # residual diagonal d_k = diag( A_{pp} - L_k L_k^T )
    d = np.diag(A).copy()
    d = np.maximum(d, 0.0)

    r = 0
    for k in range(max_rank):
        # choose pivot
        j = k + np.argmax(d[k:])
        if d[j] < 0:
            d[j] = 0.0

        # swap
        if j != k:
            p[[k, j]] = p[[j, k]]
            d[[k, j]] = d[[j, k]]
            L[[k, j], :k] = L[[j, k], :k]

        alpha = np.sqrt(max(d[k], 0.0))
        if tol is not None and alpha <= tol: # below tol
            break
        if alpha == 0.0:  # nothing more to add
            break

        L[k, k] = alpha
        if k < n - 1:
            # column of A in the permuted indexing: A[p[k+1:], p[k]]
            a_col = A[p[k+1:], p[k]]
            # compute Schur complement piece: w = (a_col - L_{k+1:, :k} @ L[k, :k]) / alpha
            if k > 0:
                w = (a_col - L[k+1:, :k] @ L[k, :k]) / alpha
            else:
                w = a_col / alpha

            # write into L
            L[k+1:, k] = w
            # update residual diagonal
            d[k+1:] -= w**2
            d[k+1:] = np.maximum(d[k+1:], 0.0)  # guard against roundoff

        r += 1

    return L[:, :r], p, r


def approx_D(eri, D):

    ####
    # Generating reference data, remove later
    w = eri.eia.reshape(-1)
    n = w.shape[0]
    Dmat = 1./(w[:, None] + w[None, :])
    print(f'Check symmetry of D (||D - D.T||): {np.linalg.norm(Dmat - Dmat.T)}')
    print(f'Check PSD of D:\n{np.linalg.eigh(Dmat)[0]}')

    # Computing approximation to D
    vals, vecs = np.linalg.eigh(Dmat)
    eps = 1e-8  # or whatever floor you want
    idx = np.where(vals > eps)[0]
    M = np.diag(np.sqrt(vals[idx])) @ vecs[:,idx].T

    print(f'\nLow rank via SVD')
    
    print(f'Reconstructing D: {np.linalg.norm(Dmat - M.T @ M)} requiring {len(idx)} vecs')
    print("relative Frobenius error:", np.linalg.norm(Dmat - M.T @ M, 'fro') / np.linalg.norm(Dmat, 'fro'))
    ####
    
    
    L, p, r = PC(Dmat, len(idx) )

    Aprx = L @ L.T
    Aperm = Dmat[np.ix_(p, p)]
    print('\nFull matrix')
    print(f'rank used: {r}')
    
    print(f'Reconstructing D: {np.linalg.norm(Aperm - Aprx)}')
    print("relative Frobenius error:", np.linalg.norm(Aperm - Aprx, 'fro') / np.linalg.norm(Aperm, 'fro'))

    # Trying matrix free 
    def diag_fn():
        return 1.0 / (2.0 * w)  # A_ii

    def col_fn(k):
        return 1.0 / (w + w[k])

    n = w.size

    L, p, r = MFPC(n, diag_fn, col_fn, max_rank=len(idx))

    Aprx = L @ L.T
    Aperm = Dmat[np.ix_(p, p)]
    print('\nMatrix free version')
    print("rank used:", r)

    print(f'Reconstructing D: {np.linalg.norm(Aperm - Aprx)}')
    print("relative Frobenius error:", np.linalg.norm(Aperm - Aprx, 'fro') / np.linalg.norm(Aperm, 'fro'))
    breakpoint()

#####################################


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
    mf.threshold = 1e-6

    mycc = cc.CCSD(mf)
    mycc.kernel()

    # Orbital localization    
    
#    ncore = chemcore(mol)
    ncore = 0
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]

    ao_labels = ["O 2p", "O 2s","H 1s"]
    minao="sto-3g"


#    ao_labels = ["O 1s", "O 2p", "O 2s", "O 3s", "O 3p", "O 3d", "H 1s", "H 2s", "H 2p"]
#    minao="cc-pvdz"

    openshell_option = 3
    
    avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
    avas_obj.with_iao = True
    avas_obj.threshold = 1e-7 
    
    _, _, mocas = avas_obj.kernel()
    act_hole = (
        np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
    )

#    act_part = (
#        np.where(avas_obj.vir_weights > avas_obj.threshold)[0] + mf.mol.nelec[0])

    act_part = (
        np.where(avas_obj.vir_weights > avas_obj.threshold)[0])

    print(act_part)

    c_lo = mocas
    #c_lo = mf.mo_coeff

    print ("dimension of active hole", len(act_hole)) 
    print ("dimension of active part", len(act_part)) 

    frag = [[act_hole, act_part]]

#    mymp = DFMP2(mf).run()
    mycc = cc.CCSD(mf).density_fit().run()

    frag_info = {'frag': [[act_hole, act_part]]}

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


    print(f'CCSD:\n Total energy: {mycc.e_tot} Correlation energ: {mycc.e_corr}')
    print(f'DF-MPCCSD:\n Total energy: {mympcc.lowlevel.e_tot} Correlation energ: {mympcc.lowlevel.e_corr}')
    print(f'Difference:\n Total energy: {float(mympcc.lowlevel.e_tot - mycc.e_tot)} Correlation energ: {mympcc.lowlevel.e_corr - mycc.e_corr}')
    breakpoint()
    # localization, where?
    # a-a, i-a do this in ERIs
    # 

    # NOTE what we want:
    #mympcc.lowlevel set #its, tol, 
    #   "ll method" set this RPA here 


