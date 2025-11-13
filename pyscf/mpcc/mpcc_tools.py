import numpy as np

import numpy as np
#from cpd import cp_als, init_from_pool_generalized, reconstruct_cp_tensor
import opt_einsum as oe
from numpy.linalg import svd, norm

def cp_d(X,rank=25):
    max_iter = 100
    tol = 1e-3
    cOption = 1
    kOption = 0
    
    results = {
        "X": {"ranks": [], "err_init3": []}, 
    }
# === CPD on X===
    multiples = list(range(1,2))   
    rank_list = [m * rank for m in multiples]
    for r in rank_list:
        weights, factors, _ = cp_als(X, r, max_iter, tol, init=3,cOption=cOption, kOption=kOption)
        X_hat = reconstruct_cp_tensor(weights, factors)
        rel_error = (np.linalg.norm(X - X_hat) / np.linalg.norm(X))*100
        print(f"[init=3] X rank {r}: error = {rel_error:.3e}")
    return X_hat, factors, weights

def cp_d1(X,factors, ran=100, lOption=1):
    def adjust_factor(factor_matrix, rank_new):
        """Trim or pad factor matrix to new rank."""
        old_rank = factor_matrix.shape[1]
        if rank_new == old_rank:
            return factor_matrix
        elif rank_new < old_rank:
            return factor_matrix[:, :rank_new]
        else:
            pad = init_from_pool_generalized(factor_matrix.shape[0], rank_new - old_rank)
            return np.hstack([factor_matrix, pad])

    max_iter = 100
    tol = 1e-3
    cOption = 1
    kOption = 0
    
    results = {
        "Lvv": {"ranks": [], "err_initfactors": []}
    }
 #=== CPD on X ===
    multiples = list(range(1,2))   
    rank_list = [m * ran for m in multiples]
    for r in rank_list:
        # reuse factors
        if lOption ==1:
            A_adj = adjust_factor(factors[0], r)
            O_adj = adjust_factor(factors[1], r)
            init_factors = [A_adj, O_adj, init_from_pool_generalized(X.shape[2], r)]
        else:
            A_adj = adjust_factor(factors[0],r)
            V_adj = adjust_factor(factors[2], r)
            init_factors = [A_adj, V_adj, V_adj]

        weights2, factors2, _ = cp_als(X, r, max_iter, tol, init=init_factors,cOption=cOption, kOption=kOption)
        X_hat2 = reconstruct_cp_tensor(weights2, factors2)
        rel_err_fact = (np.linalg.norm(X - X_hat2) / np.linalg.norm(X))*100

        print(f"[init=factors]: {rel_err_fact:.3e}")
    return X_hat2,factors2, weights2


def MFPC(n, diag_fn, col_fn, max_rank=22, tol=None):
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

def piv_chol_tensor(eia, check = False):
        
        ni, na = eia.shape
        w = eia.reshape(-1)
        def diag_fn():
            return 1.0 / (2.0 * w)

        def col_fn(k):
            return 1.0 / (w + w[k])

        L, p, r = MFPC(ni * na, diag_fn, col_fn)
        
        Lp = np.zeros_like(L)            
        Lp[p, :] = L

        L4 = Lp.reshape(ni, na, -1)
        if check:
            print("# Testing matrix-free pivoted Cholesky accuracy")
            from pyscf import lib
            A4_approx = np.einsum("iar,jbr->ijab", L4, L4)
            D = lib.direct_sum("ia+jb->ijab", eia, eia)
            A4_true = 1.0 / D
            fro_rel = np.linalg.norm(A4_true - A4_approx) / np.linalg.norm(A4_true)
            max_abs = np.max(np.abs(A4_true - A4_approx))

            print(f"4th-order relative Frobenius error: {fro_rel:.3e}")
            print(f"4th-order max abs entry error:     {max_abs:.3e}")

        return Lp.reshape(ni, na, -1) 
# inputs
#np.random.seed(42)
X    = np.random.rand(4,3,2)
rank = list(range(4,5,1))
tol  = 1e-3 
max_iter = 1 
cOption = 1
kOption = 0
init = 3
def init_from_pool_generalized(dim, rank, pool_multiplier=3):
    pool_size = max(rank, 10) * pool_multiplier  # Ensure a minimum pool size
    pool = np.random.uniform(-1, 1, (dim, pool_size))
    factors = pool[:, np.random.choice(pool_size, rank, replace=False)]
    
    # Normalize each column (factor) to unit length (L2 norm)
    norms = np.linalg.norm(factors, axis=0) 
    norms[norms == 0] = np.finfo(float).eps  # Handle zero norms
    factors_normalized = factors / norms
    
    return factors_normalized


def khatri_rao(matrices, skip_matrix=None):
    """
    Computes the Khatri-Rao product of a list of matrices.
    If skip_matrix is specified, that matrix is skipped.
    """
    matrices = [m for (i, m) in enumerate(matrices) if i != skip_matrix]
    R = matrices[0].shape[1]
    result = matrices[0]
    for m in matrices[1:]:
        result = np.einsum('ir,jr->ijr', result, m).reshape(-1, R)
    return result

def inner_product_cpd(rank,weights1,factors1,weights2,factors2):
    inner_prod= 0.0
    for r in range(rank):
        for rp in range(rank):
            prod = weights1[r] * weights2[rp]
            for A1, A2 in zip(factors1, factors2):
                prod *= np.dot(A1[:,r], A2[:,rp])
            inner_prod += prod
    return inner_prod
def normalize_col(A):
    norms = np.linalg.norm(A, axis=0)
    norms[norms ==0] = 1.0
    return A / norms, norms
def cp_als(X, rank, max_iter, tol, init, cOption,kOption):
    """
    CP-ALS algorithm for tensor decomposition.
    
    Returns:
        A list of factor matrices [A^(1), ..., A^(N)]
    """
    #np.random.seed(42)
    N = X.ndim
    shape = X.shape
    if isinstance(init, int):
        if init == 1:
            factors = [np.random.rand(s, rank) for s in shape]
        else:
            factors = [init_from_pool_generalized(s, rank) for s in shape]
    elif isinstance(init, list):
        factors = init
    else:
        raise ValueError("Error")
    norm_X = np.linalg.norm(X)

    weights = np.ones(rank)
    norm_factors = [A.copy() for A in factors]
    prev_weights = None
    prev_norm_factors = None
    prev_rel_resd = np.inf
    
    for iteration in range(max_iter):
        total_change = 0.0
        norm_Xhat_sq = 0.0 # Initialize scalar for norm squared of approximation
        modes = range(N)
        for n in modes:
            if kOption == 0:
                # Compute Gramian Matrix and product of Gramian
                G = np.ones((rank,rank))    
                for i, f in enumerate(norm_factors):
                    if i != n:
                        G *= f.T @ f

            # Building einsum string
                indices = [chr(ord('a') + i) for i in range(N)]
                einsum_str = f"{''.join(indices)},"
                einsum_str += ','.join([f"{indices[i]}r" for i in range(N) if i !=n])
                einsum_str += f"->{indices[n]}r"

            # contraction operation
                operands = [X] + [norm_factors[i] for i in range(N) if i !=n]
                M = oe.contract(einsum_str,*operands)
                reg = 1e-10
                A_new = np.linalg.solve(G + reg*np.eye(G.shape[0]), M.T).T

            else:
                Xns = [ np.reshape(np.moveaxis(X, n, 0), (shape[n], -1)) for n in range(N)] 
                # Compute M^{(n)} = X_{(n)} @ Khatri-Rao product of all factors except n
                Xn = Xns[n]
                kr = khatri_rao(norm_factors, skip_matrix=n)
                A_new = Xn @ (np.linalg.pinv(kr)).T
            # Solve least squares
            norms = norm(A_new,axis=0)
            norms[norms == 0] =1.0

            if n==N-1:
                weights = norms
                F_M = np.sum(M * A_new)  
                norm_Xhat_sq = np.sum(G * ( A_new.T @ A_new))
            A_new /= norms

            # Track change
            total_change += np.linalg.norm(norm_factors[n] - A_new) ** 2
            # Update
            norm_factors[n] = A_new

        if cOption == 1:
            residual = np.sqrt(abs( norm_X *norm_X - 2 * F_M + norm_Xhat_sq))
            rel_residual = residual / norm_X 
            delta = prev_rel_resd - rel_residual
            if delta < tol:
                print(f"Converged(CPD) in {iteration+1} iterations.") 
                break
            prev_rel_resd = rel_residual
        elif cOption == 2:
            if np.sqrt(total_change) < tol * norm(X):
                print(f"Converged(CPD) in {iteration + 1} iterations.") 
                break
        else:
            if iteration > 0:
                inner_prod = inner_product_cpd(rank,weights,norm_factors,prev_weights,prev_norm_factors)
                prev_norm_sq = norm_approx(rank,prev_weights,prev_norm_factors)
                curr_norm_sq = norm_Xhat_sq
                change = curr_norm_sq-2*inner_prod+prev_norm_sq
                rel_change = np.sqrt(max(change,0)) / np.sqrt(prev_norm_sq)
                if rel_change < tol:
                    print(f'Converged(CPD) in {iteration+1} iterations.')
                    break
            prev_weights = weights.copy()
            prev_norm_factors = [A.copy() for A in norm_factors]

    return weights,norm_factors,iteration +1


def reconstruct_cp_tensor(weights, norm_factors):
    shape = [A.shape[0] for A in norm_factors]
    rank = norm_factors[0].shape[1]
    reconstructed = np.zeros(shape)

    for r in range(rank):
        outer = weights[r] * norm_factors[0][:, r]
        for factor in norm_factors[1:]:
            outer = np.multiply.outer(outer, factor[:, r])
        reconstructed += outer
    return reconstructed
