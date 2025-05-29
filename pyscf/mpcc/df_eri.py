import numpy as np
from pyscf import lib 

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
        self.compute_three_center_ints()


    def compute_three_center_ints(self):

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

