from pyscf import lib

import numpy as np


class MPCC_LL:
    def __init__(self, mf, eris):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        # NOTE can be potentially initialized
        self.t1 = None
        self.t2 = None

        # NOTE use DIIS as default
        self.diis = True

        self.nao = mf.mol.nao
        self.nocc = mf.mol.nelec[0]
        self.nvir = self.nao - self.nocc
        self.naux = self.with_df.get_naoaux()
        self.eris = eris

    def kernel(self):

        # NOTE Do we want to initialize t1 and t2?
        self.t1 = np.zeros((self.nocc, self.nvir))
        self.t2 = np.zeros((self.nocc, self.nocc, self.nvir, self.nvir))

        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        while err > self.con_tol and count < self.max_its:

            res, DE, t1_new, t2_new = self.updated_amps()
            if self.diis:
                t1, t2 = self.run_diis(t1_new, t2_new, adiis)
            else:
                t1, t2 = t1_new, t2_new

            self.t1 = t1
            self.t2 = t2

            count += 1
            err = res
            print(f"It {count} Energy progress {DE:.6e} residual progress {res:.6e}")

        self.e_corr = DE
        self.e_tot = self.mf.e_tot + DE
        breakpoint()
        return DE

    def updated_amps(self):
        """
        Following Table XXX in Future Paper
        """

        # NOTE do we want the local copies?
        t1 = self.t1
        Loo = self.eris.Loo
        Lov = self.eris.Lov
        Lvv = self.eris.Lvv

        fock = self.eris.eia
        foo = self.eris.foo
        fvv = self.eris.fvv

        # Contractions
        X, Xoo, Xvo = self.eris.get_X(t1)

        Joo, Jvo = self.eris.get_J(Xoo, Xvo, t1)
        
        Ω = self.eris.get_Ω(X, Xoo, Xvo, Joo, Jvo, t1)
        
        Fov = self.eris.get_F(X, Xoo, Jvo)
        
        t2, Yvo = self.eris.get_t2_Yvo(Jvo)
        
        Ω = self.eris.update_Ω(Ω, Yvo, Fov, t2, t1, Joo)

        # NOTE check the sign on the first term
        e1 = np.einsum("Lij,ja->Lai", Xoo, t1) + np.einsum("L,ia->Lai", X, t1) + Jvo
        ΔE = np.einsum("Lai,Lai", e1, Yvo)

        res = np.linalg.norm(self.t1 + Ω.T / fock)

        t1 = -Ω.T / fock
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
