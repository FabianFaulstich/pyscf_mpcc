import numpy as np
from pyscf import lib, df


class ERIs:

    def __init__(self, mf):

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.nao = mf.mol.nao
        self.nocc = mf.mol.nelec[0]
        self.nvir = mf.mol.nao - self.nocc
        self.naux = self.with_df.get_naoaux()

        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy

        fock_mo = self.mo_coeff.T @ mf.get_fock() @ self.mo_coeff
        self.foo = fock_mo[: self.nocc, : self.nocc]
        self.fvv = fock_mo[self.nocc : self.nao, self.nocc : self.nao]
        self.fov = fock_mo[: self.nao, self.nocc : self.nao]

        self.eia = lib.direct_sum(
            "a-i->ia", self.mo_energy[self.nocc :], self.mo_energy[: self.nocc]
        )
        self.D = lib.direct_sum("-ia-jb->aibj", self.eia, self.eia)

    def make_eri(self):

        Loo = np.empty((self.naux, self.nocc, self.nocc))
        Lov = np.empty((self.naux, self.nocc, self.nvir))
        Lvo = np.empty((self.naux, self.nvir, self.nocc))
        Lvv = np.empty((self.naux, self.nvir, self.nvir))

        p1 = 0
        occ, vir = np.s_[: self.nocc], np.s_[self.nocc :]

        for eri1 in self.with_df.loop():
            eri1 = lib.unpack_tril(eri1).reshape(-1, self.nao, self.nao)
            Lpq = lib.einsum("Lab,ap,bq->Lpq", eri1, self.mo_coeff, self.mo_coeff)
            p0, p1 = p1, p1 + Lpq.shape[0]
            blk = np.s_[p0:p1]
            Loo[blk] = Lpq[:, occ, occ]
            Lov[blk] = Lpq[:, occ, vir]
            Lvo[blk] = Lpq[:, vir, occ]
            Lvv[blk] = Lpq[:, vir, vir]

        self.Loo = Loo
        self.Lov = Lov
        self.Lvo = Lvo
        self.Lvv = Lvv

        # NOTE 06/04 debate about cache 

    # hold all a-a, a-i, i-i tensor segments here 
    # compute Lvv on the fly *lazy?

    # NOTE Remove after discussion on 06/11
    # NOTE move contractions into the "class"
    # NOTE Now comes the second set of equations
    def get_Aia(self, u):

        tmp_eri = np.einsum("Lkc,Lad->kcad", self.Lov, self.Lvv)

        return np.einsum("kicd, kcad -> ia", u, tmp_eri)

    def get_Bia(self, u):

        tmp_eri = np.einsum("Lki,Llc->kilc", self.Loo, self.Lov)

        return np.einsum("klac, -kilc -> ia", u, tmp_eri)

    def get_Cia(self, u):

        return np.einsum("ikac, kc-> ia", u, self.fov)

    def get_Aijab(self, t):

        tmp_eri = np.einsum("Lac, Lbd->acbd", self.Lvv, self.Lvv)

        return np.einsum("ijcd, acbd -> ijab", t, tmp_eri)

    def get_Bijab(self, t):

        tmp_eri = np.einsum("Lkc, Lld -> kcld", self.self.Lov, self.self.Lov)
        B = np.einsum("ijcd, kcld -> ijkl", t, tmp_eri)
        tmp_eri = np.einsum("Lki, Ljl -> ijkl", self.Loo, self.Loo)
        B = tmp_eri + B

        return np.einsum("klab, ijkl -> ijab", t, B)

    def get_Cijab(self, t):

        # NOTE tmp_eri is the same as in get_Bijab
        tmp_eri = np.einsum("Lkc, Lld -> kcld", self.Lov, self.Lov)
        C = np.einsum("liad, kcld -> kiac", t, tmp_eri)
        tmp_eri = np.einsum("Lki, Lac -> kiac", self.Loo, self.Lvv)
        C = tmp_eri - 0.5 * C

        return np.einsum("kjbc, kiac -> ijab", t, C)

    def get_Dijab(self, t):

        L = self.make_L()
        u = make_u(t)
        B = L + 0.5 * np.eisum("ilad, dclk -> icak", u, L)

        return 0.5 * np.einsum("jkbc, icak -> ijab", u, B)

    def get_Eijab(self, t):

        u = make_u(t)
        tmp_eri = np.einsum("Lld, Lkc -> ldkc", self.Lov, self.Lov)
        E = self.fvv - np.einsum("klbd, ldkc -> bc", u, tmp_eri)

        return np.einsum("ijac, bc -> ijab", t, E)

    def get_Gijab(self, t):

        u = make_u(t)
        tmp_eri = np.einsum("Lkd, Llc -> kdlc", self.Lov, self.Lov)
        G = self.foo + np.einsum("ljcd, kdlc -> jk", u, tmp_eri)

        return np.einsum("-ikab, jk -> ijab", t, G)

    def make_u(t):

        return 2 * t - np.transpose(t, (0, 1, 3, 2))

    def make_L(self):

        tmp_eri1 = np.einsum("Lpr ,Lqs -> rspq", self.fov, self.fov)
        tmp_eri2 = np.einsum("Lps ,Lqr -> rspq", self.fov, self.fov)

        return 2 * tmp_eri1 - tmp_eri2

    #####
