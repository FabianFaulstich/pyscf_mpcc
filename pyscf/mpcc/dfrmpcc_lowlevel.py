from pyscf import lib

import numpy as np


class MPCC_LL:
    def __init__(self, mf):
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

        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff

        # NOTE Keeping this just for now as look-up table, remove later!
        # self.moa = self.eris.mo_coeff
        # self.nocca = self.eris.nocc
        # self.nao = self.moa.shape[0]
        # self.nmoa = self.moa.shape[1]

        self.nvir = self.nao - self.nocc
        self.nvir_pair = self.nvir * (self.nvir + 1) // 2
        self.compute_three_center_ints()

        fock_mo = self.mo_coeff.T @ self.mf.get_fock() @ self.mo_coeff
        self.foo = fock_mo[: self.nocc, : self.nocc]
        self.fvv = fock_mo[self.nocc : self.nao, self.nocc : self.nao]

        # NOTE eia was formerly called "fock", change in full code!
        self.eia = lib.direct_sum(
            "a-i->ia", self.mo_energy[self.nocc :], self.mo_energy[: self.nocc]
        )
        self.D = lib.direct_sum("-ia-jb->aibj", self.eia, self.eia)


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

            
    def kernel(self):

        # NOTE Do we want to initialize t1 and t2?
        self.t1 = np.zeros((self.nocc,self.nvir))
        self.t2 = np.zeros((self.nocc,self.nocc,self.nvir,self.nvir))

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
  
        print("In Kernel!")
        breakpoint()
 
        return DE


    def updated_amps(self):
        """
        Following Table XXX in Future Paper
        """
       
        # NOTE do we want the local copies?
        t1 = self.t1
        Loo = self.Loo
        Lov = self.Lov
        Lvv = self.Lvv

        fock = self.eia
        foo = self.foo
        fvv = self.fvv

        # Contractions
        Xvo = np.einsum("Lab,ib->Lai", Lvv, t1)
        Xoo = np.einsum("Lia,ja->Lij", Lov, t1)
        X = np.einsum("Lia,ia->L", Lov, t1)
        
        Joo = Xoo + Loo
        Jvo = Xvo + np.transpose(Lov, (0, 2, 1)) - np.einsum("Lij,ja->Lai", Joo, t1)
        
        Ω = -1 * np.einsum("Laj,Lji->ai", Xvo, Joo)
        Ω += np.einsum("Ljk,ka,Lji->ai", Xoo, t1, Joo)
        Ω += np.einsum("Lai,L->ai", Jvo, X)

        # NOTE singles contraction with F for non-diagonal basis
        #Ωvo += np.einsum('ia,ia -> ai',t1, fock)
        # NOTE include fov contractrion here 
        Ω += np.einsum('ib,ba -> ai',t1,fvv)
        Ω -= np.einsum('ka,ik -> ai',t1,foo)

        Fov = np.einsum("Lbj,L->jb", Jvo, X) - np.einsum("Lij,Lib->jb", Xoo, Lov)

        eris = np.einsum("Lai,Ljb->aijb", Jvo, Jvo)

        t2 = (2 * eris - np.transpose(eris, (0, 3, 2, 1))) / self.D
        Yvo = np.einsum("aibj,Ljb->Lai", t2, Lov)
        Ω += np.einsum("aijb,bj->ai", t2, Fov)

        Jvv = np.einsum("Ljb,ja->Lba", Lov, t1) + Lvv
        Ω += np.einsum("Lba,Lbi->ai", Jvv, Yvo)
        Ω -= np.einsum("Lji,Laj->ai", Joo, Yvo)

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

    def amplitudes_to_vector(self, t1, t2, out = None):
        nov = self.nocc * self.nvir
        size = nov + nov*(nov+1)//2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov), out=vector[nov:])
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nocc * self.nvir
        t1 = vector[:nov].copy().reshape((self.nocc,self.nvir))
        # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
        t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
        t2 = t2.reshape(self.nocc,self.nvir,self.nocc,self.nvir).transpose(0,2,1,3)
        return t1, np.asarray(t2, order='C')


