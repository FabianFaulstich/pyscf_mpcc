from pyscf import lib, df
from pyscf.lib import logger


import numpy as np


class MPCC_LL:
    def __init__(self, mf, eris, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        if 'll_max_its' in kwargs:
            self.ll_max_its = kwargs['ll_max_its']
        else:    
            self.ll_max_its = 50

        if 'll_con_tol' in kwargs:
            self.ll_con_tol = kwargs['ll_con_tol']
        else:
            self.ll_con_tol = 1e-6
        
        # NOTE can be potentially initialized
        self.t1 = None
        self.t2 = None

        # NOTE use DIIS as default
        self.diis = True

        self._eris = eris
        self._e_corr = None

    @property
    def e_tot(self):
        if self._e_corr is None:
            print('MPCC did not run, return mean field solution:')
            return float(self.mf.e_tot)
        else:
            return float(self.mf.e_tot + self._e_corr)

    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc
    
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    def kernel(self):

        # NOTE Do we want to initialize t1 and t2?
        self.t1 = np.zeros((self.nocc, self.nvir))
        self.t2 = np.zeros((self.nocc, self.nocc, self.nvir, self.nvir))

        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        e_corr = None
        while err > self.ll_con_tol and count < self.ll_max_its:

            res, e_corr, t1_new, t2_new = self.updated_amps()
            if self.diis:
                t1, t2 = self.run_diis(t1_new, t2_new, adiis)
            else:
                t1, t2 = t1_new, t2_new

            self.t1 = t1
            self.t2 = t2

            count += 1
            err = res
            # NOTE change this to logger!
            print(f"It {count}; correlation energy {e_corr:.6e}; residual {res:.6e}")

        self._e_corr = e_corr
        self._e_tot = self.mf.e_tot + self._e_corr


    def updated_amps(self):
        """
        Following Table XXX in Future Paper
        """

        # NOTE do we want the local copy?
        t1 = self.t1

        # Contractions
        X, Xoo, Xvo = self.get_X(t1)

        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        
        Ω = self.get_Ω(X, Xoo, Xvo, Joo, Jvo, t1)
        
        Fov = self.get_F(X, Xoo, Jvo)
        
        t2, Yvo = self.get_t2_Yvo(Jvo)
        
        Ω = self.update_Ω(Ω, Yvo, Fov, t2, t1, Joo)

        # NOTE check the sign on the first term
        e1 = np.einsum("Lij,ja->Lai", Xoo, t1) + np.einsum("L,ia->Lai", X, t1) + Jvo
        ΔE = np.einsum("Lai,Lai", e1, Yvo)

        res = np.linalg.norm(self.t1 + Ω.T / self._eris.eia)

        t1 = -Ω.T / self._eris.eia
        return res, ΔE, t1, t2

    def run_diis(self, t1, t2, adiis):

        vec = self.amplitudes_to_vector(t1, t2)
        t1, t2 = self.vector_to_amplitudes(adiis.update(vec))

        return t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        nov = self.nocc * self.nvir
        size = nov + nov * (nov + 1) // 2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(t2.transpose(0, 2, 1, 3).reshape(nov, nov), out=vector[nov:])
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nocc * self.nvir
        t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
        # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
        t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
        t2 = t2.reshape(self.nocc, self.nvir, self.nocc, self.nvir).transpose(
            0, 2, 1, 3
        )
        return t1, np.asarray(t2, order="C")

    def get_X(self, t1):

        Xvo = np.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = np.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = np.einsum("Lia,ia->L", self._eris.Lov, t1)

        return X, Xoo, Xvo

    def get_J(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + np.transpose(self._eris.Lov, (0, 2, 1)) - np.einsum("Lij,ja->Lai", Joo, t1)
        )

        return Joo, Jvo

    def get_Ω(self, X, Xoo, Xvo, Joo, Jvo, t1):

        Ω = -1 * np.einsum("Laj,Lji->ai", Xvo, Joo)
        Ω += np.einsum("Ljk,ka,Lji->ai", Xoo, t1, Joo)
        Ω += np.einsum("Lai,L->ai", Jvo, X)

        Ω += np.einsum("ib,ba -> ai", t1, self._eris.fvv)
        Ω -= np.einsum("ka,ik -> ai", t1, self._eris.foo)

        return Ω

    def get_F(self, X, Xoo, Jvo):

        return np.einsum("Lbj,L->jb", Jvo, X) - np.einsum("Lij,Lib->jb", Xoo, self._eris.Lov)

    def get_t2_Yvo(self, Jvo):

        _eris = np.einsum("Lai,Ljb->aijb", Jvo, Jvo)
        t2 = (2 * _eris - np.transpose(_eris, (0, 3, 2, 1))) / self._eris.D
        Yvo = np.einsum("aibj,Ljb->Lai", t2, self._eris.Lov)

        return t2, Yvo

    def update_Ω(self, Ω, Yvo, Fov, t2, t1, Joo):

        Ω += np.einsum("aijb,bj->ai", t2, Fov)
        Jvv = np.einsum("Ljb,ja->Lba", self._eris.Lov, t1) + self._eris.Lvv
        Ω += np.einsum("Lba,Lbi->ai", Jvv, Yvo)
        Ω -= np.einsum("Lji,Laj->ai", Joo, Yvo)

        return Ω

