import numpy as np

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

