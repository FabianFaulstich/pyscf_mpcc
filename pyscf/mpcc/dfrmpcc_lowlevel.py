from numpy.linalg import qr
from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass

from pyscf.mpcc import mpcc_tools
from functools import partial

import time 

class MPCC_LL:
    def __init__(self, mf, eris, frags, **kwargs):
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
 
        if 'll_kernel_type' in kwargs:
            self.kernel_type = kwargs['ll_kernel_type']
        else:
            self.kernel_type = 'factorized'

        if 'll_method' in kwargs:
            self.ll_method = kwargs['ll_method']
        else:
            self.ll_method = 'T1_transform'

        if 'll_low_rank_tol' in kwargs:
            self.ll_low_rank_tol = kwargs['ll_low_rank_tol']
        else:
            self.ll_low_rank_tol = None

        self._kernels = {
                'factorized': self._factorized_kernel,
                'unfactorized': self._unfactorized_kernel, 
                }

        self.frags = frags

        #NOTE can be potentially initialized
        #self.t1 = None
        self.t2 = None
        self._Y = None

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
    def e_corr(self):
        return float(self._e_corr)

    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc
    
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    def kernel(self, t1, t2_act ,**kwargs):

        print('Starting low-level MPCC iteration...')
        try:
            func = self._kernels[self.kernel_type]
        except KeyError:
            raise ValueError(f'Unknown low-level kernel type: {self.kernel_type}')
        
        return func(t1, t2_act ,**kwargs)

    def _unfactorized_kernel(self, t1=None, t2=None, **kwargs):
        print('In unfactorized Kernel')

        #ll_method = kwargs.get('ll_method', self.ll_method)
        if self.ll_method == 'rpax':
            update_amps = self.update_amps_unfactorized_RPA
        elif self.ll_method == 'T1_transform':
            update_amps = self.update_amps_unfactorized
        else:
            raise ValueError(
                f"Unknown ll_method: {self.ll_method}. "
                "Use 'rpax', or 'T1_transform'."
            )
        
        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        e_corr = None

        while err > self.ll_con_tol and count < self.ll_max_its:

            res, e_corr, t1_new, t2_new = update_amps(t1, t2)
            if self.diis:
                t1_new, t2_new = self.run_diis_full(t1_new, t2_new, adiis)
            else:
                t1_new, t2_new = t1_new, t2_new

            t1, t2 = t1_new, t2_new
            t1_new, t2_new = None, None  # free memory
            
            count += 1
            err = res
            # NOTE change this to logger!
            print(f"It {count}; correlation energy {e_corr:.6e}; residual {res:.6e}")

        self._e_corr = e_corr
        self._e_tot = self.mf.e_tot + self._e_corr

        return t1, t2

    def update_amps_unfactorized(self, t1, t2):

        # Contractions
        Xoo, Xvo, X = self.get_X(t1)

        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Ω = self.get_Ω_slow(X, Xvo, Foo, Fvv, Fov, t1, t2)

        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
    
        ΔE = self.get_energy(t1, t2)

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0
            res2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0

        res1 = Ω.T / self._eris.eia
        res2 = res2 / self._eris.D
        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 -= res1
        t2 -= res2

        return res, ΔE, t1, t2


    def update_amps_unfactorized_RPA(self, t1, t2):

        # Contractions
        Xoo, Xvo, X = self.get_X(t1)

        Xvo_t2 = self.get_Xvo_t2(t2)
        Joo, Jvo, Jvv = self.get_J_RPA(Xoo, Xvo, Xvo_t2, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)    

        Ω = self.get_Ω_slow_RPA(X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2)

        res2 = self.update_t2_RPA(t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv, Xvo_t2)
    
        ΔE = self.get_energy(t1, t2)

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0
            res2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0

        res1 = Ω.T / self._eris.eia
        res2 = res2 / self._eris.D
        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 -= res1
        t2 -= res2

        return res, ΔE, t1, t2


    def _factorized_kernel(self, t1=None, t2=None, **kwargs):

        res = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            t2_act = t2[np.ix_(act_hole, act_hole, act_particle, act_particle)]

        e_corr = None
        while res > self.ll_con_tol and count < self.ll_max_its:

            res, t1_it, Δt2s_o, Δt2s_v, Y = self.update_amps_factorized(t1, t2_act, self._Y, **kwargs)
            if self.diis:
                t1_it  = self.run_diis(t1_it, adiis)

            t1 = t1_it
            self._Y = Y
            
            count += 1
            # NOTE change this to logger!
            print(f"It {count}; residual {res:.6e}")


        # NOTE non-iterative N^5 cost!
        t2 = self.get_t2(Y, t2_act, Δt2s_o, Δt2s_v)
        self._e_corr = self.get_energy(t1, t2) 

        return t1, t2

    def update_amps_factorized(self, t1, t2_act, Y, **kwargs):
        """
        Following Table XXX in Future Paper
        """
        # Setp 1 
        Xoo, Xvo, X = self.get_X(t1)

        # Step 2
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
       
        # Step 3
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        # Step 4 
        Ω = self.get_Ω(X, Xvo, Foo, Fvv, Fov, t1, Y)

        # NOTE discuss the grouping of actions, 5-7/8 seems to be fit for one step
        # Step 5 
        Foo, Fvv = self.update_F(Foo, Fvv, Fov, t1)

        # Step 6 & 7
        D, Uoo, Uvv = self.get_D(Foo, Fvv, **kwargs)

        # NOTE Can we transform D insead of J, that way we don't have to transform Y back!
        

        # Step 8
        Jvo = self.update_J(Jvo, Uoo, Uvv)

        # Step 9
        Y, Yt = self.update_Y_backup(Joo, Jvo, D, Uvv, Uoo, Y)

        # Step 10 & 11
        Δt2s_o, Δt2s_v, Ω = self.include_t2_active(Foo, Fvv, Fov, t2_act, Y, Ω)
       
        res = Ω.T / self._eris.eia
        t1 -= res

        # NOTE Checking energy convergence, remove energy computation later!
        t2 = self.get_t2(Y, t2_act, Δt2s_o, Δt2s_v)
        e_corr = self.get_energy(t1, t2) 
        print(f'    CC2 correlation energy: {e_corr}')


        return np.linalg.norm(res), t1, Δt2s_o, Δt2s_v, Y
    
    @dataclass
    class _IMDS_T1:
       oo: np.ndarray
       Jvv: np.ndarray
       Jov: np.ndarray
       Foo: np.ndarray
       Fvv: np.ndarray
       Fov: np.ndarray
       Xoo: np.ndarray
       Xvo: np.ndarray
    
    def get_t1_imds(self, t1):

        X, Xoo, Xvo = self.get_X(t1)

        #Joo, Jvo, Jvv = self.get_J_all(Xoo, Xvo, t1)
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) 

        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Jvo)

        #store Joo, Jvo, Foo etc..
        imds_t1 = self._IMDS_T1(Joo=Joo, Jvv=Jvv, Jov=Jvo, Foo=Foo, Fvv=Fvv, Fov=Fov, Xoo=Xoo, Xvo=Xvo)
        return imds_t1
    
    def run_diis_full(self, t1, t2, adiis):

        def amplitudes_to_vector_full(t1, t2, out=None):
            nov = self.nocc * self.nvir
            size = nov + nov * (nov + 1) // 2
            vector = np.ndarray(size, t1.dtype, buffer=out)
            vector[:nov] = t1.ravel()
            lib.pack_tril(t2.transpose(0, 2, 1, 3).reshape(nov, nov), out=vector[nov:])
            return vector

        def vector_to_amplitudes_full(vector):
            nov = self.nocc * self.nvir
            t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
            # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
            t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
            t2 = t2.reshape(self.nocc, self.nvir, self.nocc, self.nvir).transpose(
                0, 2, 1, 3
            )
            return t1, np.asarray(t2, order="C")

        vec = amplitudes_to_vector_full(t1, t2)
        t1, t2 = vector_to_amplitudes_full(adiis.update(vec))

        return t1, t2

    def run_diis(self, t1, adiis):

        def amplitudes_to_vector(t1):
            nov = t1.shape[0] * t1.shape[1]
            vector = t1.ravel()
            return vector

        def vector_to_amplitudes(vector):
            nov = self.nocc * self.nvir
            t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
            return t1

        vec = amplitudes_to_vector(t1)
        t1 = vector_to_amplitudes(adiis.update(vec))

        return t1

    def run_diis_Δt2(self, t2_in, adiis):

        def amplitudes_to_vector(t2):
            return t2.ravel()

        def vector_to_amplitudes_shapes(vector, shapes):
            return vector.reshape(shapes)

        vec2amp = partial(vector_to_amplitudes_shapes, shapes = t2_in.shape)
        vec = amplitudes_to_vector(t2_in)

        return vec2amp(adiis.update(vec))

    def get_X(self, t1):

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return Xoo, Xvo, X

    def add_t2_to_fock(self, Fvv, Foo, Xvo_t2):    

        Foo += lib.einsum("Lie,Lej->ij", self._eris.Lov,Xvo_t2)
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo_t2)

        return Foo, Fvv


    def get_Xvo_t2(self, t2):

        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        Xvo_t2 = lib.einsum("Lkc,ikac ->Lai", self._eris.Lov, t2_antisym)

        return Xvo_t2


    def get_J(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        return Joo, Jvo


    def get_J_RPA(self, Xoo, Xvo, Xvo_t2, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + Xvo_t2 + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) #we don't need this here  

        return Joo, Jvo, Jvv



    def get_F(self, t1, X, Xoo, Xvo):

        Foo = self._eris.foo.copy()
        Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
        Foo -= lib.einsum("Lmj,Lim->ij", self._eris.Loo,Xoo)

        Fvv = self._eris.fvv.copy()
        Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X) 
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo)

        Fov  = self._eris.fov.copy()
        Fov += lib.einsum("Ljb,L->jb", self._eris.Lov, X)
        Fov -= lib.einsum("Lji,Lib->jb", Xoo, self._eris.Lov)

        return Foo, Fvv, Fov

    # FIXME There is substantial code overlap with get_Ω
    def get_Ω_slow(self, X, Xvo, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)*0.5 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)


        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)

        del Foo_tmp, Fvv_tmp
        
        return Ω 


    def get_Ω_slow_RPA(self, X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)*0.5 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo) #new
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        Ω += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2) #new


        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)

        del Foo_tmp, Fvv_tmp
        
        return Ω 


    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)
        res2 = tmp + tmp.transpose(1,0,3,2)
        res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        return res2

    def _svd_factorize_slice(self, mat, tol):

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        if s.size == 0:
            return u[:, :0], vh[:0]

        cutoff = tol * s[0]
        rank = np.count_nonzero(s > cutoff)
        if rank == 0:
            rank = 1

        return u[:, :rank] * s[:rank], vh[:rank]

    def _contract_n3v3(self, Joo, Jvv, t2):

        tol = self.ll_low_rank_tol
        if tol is None:
            return lib.einsum("Lmj,Lbe,imae->ijab", Joo, Jvv, t2, optimize=True)

        naux = Joo.shape[0]
        nocc = Joo.shape[1]
        nvir = Jvv.shape[1]

        contracted = np.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        rank_oo = []
        rank_vv = []

        print(f"Performing low-rank factorization of Joo and Jvv with tol={tol:.2e} for N3V3 contraction...")

        for aux_idx in range(naux):
            left_oo, right_oo = self._svd_factorize_slice(Joo[aux_idx], tol)
            left_vv, right_vv = self._svd_factorize_slice(Jvv[aux_idx], tol)

            rank_oo.append(left_oo.shape[1])
            rank_vv.append(left_vv.shape[1])

            tmp_lr = lib.einsum("mp,qe,imae->iapq", left_oo, right_vv, t2, optimize=True)
            contracted += lib.einsum("pj,bq,iapq->ijab", right_oo, left_vv, tmp_lr, optimize=True)

        avg_rank_oo = float(np.mean(rank_oo))
        avg_rank_vv = float(np.mean(rank_vv))

        print(f"rank-reduced N3V3 contraction enabled: avg rank(Joo)={avg_rank_oo:.2f} avg rank(Jvv)={avg_rank_vv:.2f}")

        #logger.debug1(
        #    self,
        #    "rank-reduced N3V3 contraction enabled: avg rank(Joo)=%.2f avg rank(Jvv)=%.2f",
        #    avg_rank_oo,
        #    avg_rank_vv,
        #)
        return contracted


    def update_t2_RPA(self, t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv, Xvo_t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)

## N3V3

        #W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv)
        #tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2)

        tmp -= self._contract_n3v3(Joo, Jvv, t2)


#       W_jema = lib.einsum("Lmj, Lae -> maje", Joo, Jvv)
#       tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2) 

        res2 = tmp + tmp.transpose(1,0,3,2)
        res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

    #    Waebf = lib.einsum("Lae, Lbf -> abef", Jvv, Jvv)
    #    res2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)

    #    Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo, Joo)
    #    res2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2) 

#       res2 += lib.einsum("Lai,Lbj->ijab", Xvo_t2, Jvo)
#       res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Xvo_t2)

        return res2


    def get_Ω(self, X, Xvo, Foo, Fvv, Fov, t1, Y):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)*0.5 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)
      
        # Here adjust to only update the environment part 
        Ω_temp = lib.einsum("LRjb, bj -> LR", Y, Fov)
        Ω += 2* lib.einsum("LR, LRai -> ai", Ω_temp, Y) 

        Ω_temp = lib.einsum("LRbi, jb -> LRij", Y, Fov)
        Ω -= lib.einsum("LRij, LRaj", Ω_temp, Y)

        del Foo_tmp, Fvv_tmp, Ω_temp

        return Ω


    def t2_transform_quadratic(self,t2):

        Vnemf = lib.einsum("Lne, Lmf -> nmef", self._eris.Lov, self._eris.Lov)
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2) # #nf should be ii, ia, ai types

        return Imbje


    def update_F(self, Foo, Fvv, Fov, t1):

        Fvv = Fvv - lib.einsum("ic,ib -> bc", Fov, t1) 
        Foo = Foo + lib.einsum("kc,ic -> ki", Fov, t1)
 
        return Foo, Fvv

    def get_D(self, Foo, Fvv, **kwargs):

        Foo = 0.5 * (Foo + Foo.T)
        Fvv = 0.5 * (Fvv + Fvv.T)

        #print(f"Symmetry in Foo: {np.linalg.norm(Foo - Foo.T)}")
        #print(f"Symmetry in Fvv: {np.linalg.norm(Fvv - Fvv.T)}")

        e_oo, Uoo = np.linalg.eigh(Foo)
        e_vv, Uvv = np.linalg.eigh(Fvv)
        eia = lib.direct_sum("-i+a->ia", e_oo, e_vv)
      
        if 'chol_tol' in kwargs:
            chol_tol = kwargs['chol_tol']
        else:
            chol_tol = None
       
        if 'chol_rank' in kwargs:
            chol_rank = kwargs['chol_rank']
        else:
            chol_rank = None
        
        D = mpcc_tools.piv_chol_tensor(eia, tol = chol_tol, rank = chol_rank)

        return D, Uoo, Uvv

    def update_J(self, Jvo, Uoo, Uvv):

        Jvo = np.einsum("ab, ij, Lai -> Lbj", Uvv, Uoo, Jvo)

        return Jvo

    def update_Y(self, Jvo, D, Uvv, Uoo):

        dt = np.einsum('jbr, ij, ab', D, Uoo, Uvv)
        return Jvo[:, None, :, :] *D.transpose(2, 1, 0)[None, :, :, :]

    def update_Y_backup(self, Joo, Jvo, D, Uvv, Uoo, Y):

        st = time.time()
        Yt = Jvo[:, None, :, :] *D.transpose(2, 1, 0)[None, :, :, :]
        print(f'HP elapsed time: {time.time() - st}')

        # NOTE REMOVE THE CHECK IF WE ARE HAPPY!!!
        if False:
            t2_app = lib.einsum("LRai, LRbj -> ijab", Yt, Yt)    
            
            fro_rel = np.linalg.norm(t2 - t2_app) / np.linalg.norm(t2)
            max_abs = np.max(np.abs(t2 - t2_app))

            print(f'Test in update_Y:')
            print(f"4th-order relative Frobenius error: {fro_rel:.3e}")
            print(f"4th-order max abs entry error:     {max_abs:.3e}")

        st = time.time()
        Y = np.einsum("LRbj, ab, ij -> LRai", Yt, Uvv, Uoo, optimize = True)
        print(f'EINSUM elapsed time: {time.time() - st}')
        return Y, Yt

    def include_t2_active(self, Foo, Fvv, Fov, t2_act, Y, Ω, tol = 1e-6, count_tol = 1000):
       
        print(f'Computing active t2-correction ...')   
        Δt2s_o = [] 
        Δt2s_v = [] 

        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):

            # FIXME once fragmentation is assigned, compute these once!!!
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)
 
            eia_o = lib.direct_sum("Ia+jb->Ijab", 
                                   self._eris.eia[np.ix_(inact_hole, act_particle)], 
                                   self._eris.eia[np.ix_(act_hole, act_particle)])
            eia_v = lib.direct_sum("iA+jb->ijAb", 
                                   self._eris.eia[np.ix_(act_hole, inact_particle)], 
                                   self._eris.eia[np.ix_(act_hole, act_particle)])

            # NOTE This is the truly iterative part
            # Step 10  
            Ω[np.ix_(act_particle, act_hole)] = 0.0

            δt2 = -lib.einsum("LRai, LRbj -> ijab", 
                               Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)], 
                               Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)])    
            Δt2 = t2_act[k] - δt2 
  
            Δt2_o = -lib.einsum('kI, kjab -> Ijab ',Foo[np.ix_(act_hole, inact_hole)], Δt2)
        #   Δt2_o -=  lib.einsum('kJ, ikab -> Jiab ',Foo[np.ix_(act_hole, inact_hole)], Δt2)
            Δt2_o -=  lib.einsum('kJ, kiba -> Jiba ',Foo[np.ix_(act_hole, inact_hole)], Δt2) 

            Δt2_v = lib.einsum('Ac, ijcb -> ijAb ',Fvv[np.ix_(inact_particle, act_particle)], Δt2)
            Δt2_v += lib.einsum('Bc, ijac -> ijBa ',Fvv[np.ix_(inact_particle, act_particle)], Δt2)
          # Δt2_v += lib.einsum('Bc, jica -> jiBa ',Fvv[np.ix_(inact_particle, act_particle)], Δt2)

            #calculate the norm of Δt2_o and Δt2_v 
            res_o = np.linalg.norm(Δt2_o)
            res_v = np.linalg.norm(Δt2_v)
            res = np.sqrt(res_o**2 + res_v**2)
            #res = res_o + res_v
            print(f'    Initial residual for t2 correction: {res:.3e}')
            #unit test this residual calculation, to match it with a precalculated residual value  1.13262e-01
            if abs(res - 1.13262e-01) > 1e-3:
                print(f'    WARNING: Residual calculation does not match expected value! Computed: {abs(res - 1.13262e-01):.5e}, Expected: 1.13262e-01')

            Δt2_o_it_save = np.copy(Δt2_o)
            Δt2_v_it_save = np.copy(Δt2_v)

  #          Δt2_o_it = np.copy(Δt2_o)
  #          Δt2_v_it = np.copy(Δt2_v)

            Δt2_o = Δt2_o / eia_o
            Δt2_v = Δt2_v / eia_v

           #set active part to zero
            #Δt2_o[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
            #Δt2_v[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0

            count = 0 
            acc = np.inf 
            acc_prev = np.inf
            
            # DIIS acceleration parameters
          #  adiis_o = lib.diis.DIIS()
            adiis_v = lib.diis.DIIS()
          #  adiis_o.min_space = 2
           # adiis_o.space = 15
            adiis_v.min_space = 2
            adiis_v.space = 15
            
            diis_start = 3
            
            # Adaptive damping: aggressive early, conservative later
            damp_init = 0.3
            damp_final = 0.8
            
            # Preconditioning: normalize by expected magnitude scales
            scale_o = np.sqrt(np.mean(eia_o**2))
            scale_v = np.sqrt(np.mean(eia_v**2))

            while (acc > tol and count < count_tol):
                Δt2_o_it = Δt2_o_it_save.copy()
                Δt2_v_it = Δt2_v_it_save.copy()

                # Compute residuals with Fock contributions
                Δt2_o_it -= lib.einsum('jk,Ikab -> Ijab', Foo[np.ix_(act_hole, act_hole)], Δt2_o)
                Δt2_o_it -= lib.einsum('IK,Kjab -> Ijab', Foo[np.ix_(inact_hole, inact_hole)], Δt2_o)
                Δt2_o_it /= eia_o
                
                Δt2_v_it += lib.einsum('bc,ijAc -> ijAb', Fvv[np.ix_(act_particle, act_particle)], Δt2_v)
                Δt2_v_it += lib.einsum('AC,ijCb -> ijAb', Fvv[np.ix_(inact_particle, inact_particle)], Δt2_v)
                Δt2_v_it /= eia_v

            #set active part to zero
                #Δt2_o[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
                #Δt2_v[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
                


                # Compute convergence criteria before update
                acc_o = np.linalg.norm(Δt2_o_it)
                acc_v = np.linalg.norm(Δt2_v_it)
                acc = np.sqrt(acc_o**2 + acc_v**2)

                # Adaptive damping based on convergence rate
                if count == 0:
                    damp = damp_init
                else:
                    # Progress-based damping: stronger damping when progress stalls
                    rel_improvement = (acc_prev - acc) / (acc_prev + 1e-12)
                    if rel_improvement < 0.05:  # Stalled convergence
                        damp = min(damp_final, damp + 0.05)
                    elif rel_improvement > 0.3:  # Good progress
                        damp = max(damp_init, damp - 0.02)
                    else:
                        damp = damp_init + (damp_final - damp_init) * (count / max(count_tol, 1)) ** 1.5


                damp = damp_init + (damp_final - damp_init) * (count / count_tol) ** 1.5

                # Apply DIIS acceleration if converging well
                if count >= diis_start:
                    try:
                        # Combine amplitude updates for DIIS
                        #Δt2_o_vec = self.run_diis_Δt2(Δt2_o - (1.0 - damp) * Δt2_o_it, adiis_o)
                        Δt2_v_vec = self.run_diis_Δt2(Δt2_v - (1.0 - damp) * Δt2_v_it, adiis_v)
                        
                        Δt2_o = Δt2_o - Δt2_o_it
                        Δt2_v = Δt2_v_vec
                    except:
                        # Fall back to damped update if DIIS fails
                        Δt2_o = Δt2_o - (1.0 - damp) * Δt2_o_it
                        Δt2_v = Δt2_v - (1.0 - damp) * Δt2_v_it
                else:
                    # Standard damped update in early iterations
                    Δt2_o = Δt2_o - (1.0 - damp) * Δt2_o_it
                    Δt2_v = Δt2_v - (1.0 - damp) * Δt2_v_it

                count += 1
                
                # Diagnostic output
                if acc_prev != np.inf:
                    rel_improvement = (acc_prev - acc) / (acc_prev + 1e-12) * 100
                    print(f'    It: {count},  acc occ: {acc_o:.2e},  acc vir: {acc_v:.2e},  '
                          f'total: {acc:.2e},  rel_impr: {rel_improvement:+.1f}%,  damp: {damp:.2f}')
                else:
                    print(f'    It: {count},  acc occ: {acc_o:.2e},  acc vir: {acc_v:.2e},  total: {acc:.2e}')
                
                acc_prev = acc
            
            print(f'    Iter. T2 correction converged in {count}/{count_tol} steps with final accuracy {acc:.2e}')

            Δt2s_o.append(Δt2_o)
            Δt2s_v.append(Δt2_v)

            # Step 11 use t2 active correction to improve Ω

            t2_antisym = 2.0*Δt2_o - np.transpose(Δt2_o, (0, 1, 3, 2))
            Ω[np.ix_(act_particle, inact_hole)] += np.einsum("Ijab,jb -> aI", t2_antisym, Fov[np.ix_(act_hole, act_particle)])
  
            t2_antisym = 2.0*Δt2_v - np.transpose(Δt2_v, (1, 0, 2, 3))
            Ω[np.ix_(inact_particle, act_hole)] += np.einsum("ijAb,jb -> Ai", t2_antisym, Fov[np.ix_(act_hole, act_particle)])

        return Δt2s_o, Δt2s_v , Ω


    def include_t2_active_stupid(self, Foo, Fvv, Fov, t2_act, Y, Ω, tol = 1e-6, count_tol = 1000):
       
        print(f'Computing active t2-correction ...')   
        Δt2s_o = [] 
        Δt2s_v = [] 

        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):

            # FIXME once fragmentation is assigned, compute these once!!!
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)
 
            eia_o = lib.direct_sum("Ia+jb->Ijab", 
                                   self._eris.eia[np.ix_(inact_hole, act_particle)], 
                                   self._eris.eia[np.ix_(act_hole, act_particle)])
            eia_v = lib.direct_sum("iA+jb->ijAb", 
                                   self._eris.eia[np.ix_(act_hole, inact_particle)], 
                                   self._eris.eia[np.ix_(act_hole, act_particle)])

            # NOTE This is the truly iterative part
            # Step 10  
            Ω[np.ix_(act_particle, act_hole)] = 0.0

            δt2 = -lib.einsum("LRai, LRbj -> ijab", 
                               Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)], 
                               Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)])    
            Δt2 = t2_act[k] - δt2 

            dt2_all = np.zeros((n_occ, n_occ, n_vir, n_vir)) 

            dt2_all[np.ix_(act_hole, act_hole, act_particle, act_particle)] = Δt2

            tmp  = lib.einsum("bc,ijac->ijab", Fvv, dt2_all)
            tmp -= lib.einsum("mi,mjab->ijab", Foo, dt2_all)

            res2 = tmp + tmp.transpose(1,0,3,2)


   # set active part to zero before iteration, we will add it back after convergence
            res2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0


            print(f'Initial residual norm before iteration: {np.linalg.norm(res2):.5e}')   

            Δt2_it_save = np.copy(res2)

  #         Δt2_o_it = np.copy(Δt2_o)
  #         Δt2_v_it = np.copy(Δt2_v)


            #dt2_all = 0.0*dt2_all
            dt2_all = res2/self._eris.D

           #set active part to zero
            dt2_all[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
    #       Δt2_o[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
   #        Δt2_v[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0


            count = 0 
            acc = np.inf 
            acc_prev = np.inf

            adiis = lib.diis.DIIS()
            adiis.min_space = 2
            adiis.space = 15
            
            diis_start = 4
            damp_init = 0.1
            damp_final = 0.8
            use_diis = True

            while (acc > tol and count< count_tol):
                Δt2_it = Δt2_it_save.copy()

                res2 = -lib.einsum("mi,mjab->ijab", Foo, dt2_all)
                res2 += lib.einsum("bc,ijac->ijab", Fvv, dt2_all)

                Δt2_it += res2 + res2.transpose(1,0,3,2)

                Δt2_it /= self._eris.D

                # Zero out active-active part before computing residual
                Δt2_it[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
                acc = np.linalg.norm(Δt2_it)

                # Adaptive damping: stronger early, weaker later
                damp = damp_init + (damp_final - damp_init) * (count / count_tol) ** 1.5

                if use_diis and count >= diis_start:
                    # Apply DIIS to the full residual
                    #dt2_all = self.run_diis_Δt2(dt2_all - (1.0 - damp) * Δt2_it, adiis)
                    dt2_all = self.run_diis_Δt2(dt2_all - Δt2_it, adiis)
                else:
                    # Standard damped update without DIIS
                    dt2_all = dt2_all - (1.0 - damp) * Δt2_it
                    dt2_all = dt2_all - Δt2_it

                count += 1

                # Check relative convergence
                if acc_prev != 0:
                    rel_improvement = (acc_prev - acc) / acc_prev
                else:
                    rel_improvement = 0.0

                print(f'    It: {count},  acc: {acc:.2e},  rel_impr: {rel_improvement:.2e},  damp: {damp:.2f}')
                
                acc_prev = acc
            
            print(f'    Iter. T2 correction finished in {count}/{count_tol} steps at {acc:.2e} accuracy.')


            Δt2_o = Δt2_it[np.ix_(inact_hole, act_hole, act_particle, act_particle)] 
            Δt2_v = Δt2_it[np.ix_(act_hole, act_hole, inact_particle, act_particle)]


            Δt2s_o.append(Δt2_o)
            Δt2s_v.append(Δt2_v)

            # Step 11 use t2 active correction to improve Ω

            t2_antisym = 2.0*Δt2_o - np.transpose(Δt2_o, (0, 1, 3, 2))
            Ω[np.ix_(act_particle, inact_hole)] += np.einsum("Ijab,jb -> aI", t2_antisym, Fov[np.ix_(act_hole, act_particle)])
  
            t2_antisym = 2.0*Δt2_v - np.transpose(Δt2_v, (1, 0, 2, 3))
            Ω[np.ix_(inact_particle, act_hole)] += np.einsum("ijAb,jb -> Ai", t2_antisym, Fov[np.ix_(act_hole, act_particle)])

        return Δt2s_o, Δt2s_v , Ω


            
    def init_amps_fact(self):
       
        Y = self._eris.Lov[:, None, :, :] *self._eris.dD.transpose(2, 0, 1)[None, :, :, :]
        t1 = self._eris.fov/self._eris.eia

        return t1, Y

    def init_amps(self):

        # NOTE Check the initialization here!
        t2 = -lib.einsum("Lia,Ljb->ijab", self._eris.Lov, self._eris.Lov)
        t2 /= self._eris.D     
        t1 = -self._eris.fov/self._eris.eia
        Y = -self._eris.Lov.transpose(0,2,1)[:, None, :, :] *self._eris.dD.transpose(2, 1, 0)[None, :, :, :]
        
        self._Y = Y
        self._t2 = t2
        return t1, t2

    def get_t2(self, Y, t2_act, Δt2s_o, Δt2s_v):

        t2 = -lib.einsum("LRai, LRbj -> ijab", Y, Y)
        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[k]

            t2[np.ix_(inact_hole, act_hole, act_particle, act_particle)] += Δt2s_o[k]
            t2[np.ix_(act_hole, act_hole, inact_particle, act_particle)] += Δt2s_v[k]

        return t2

    def get_energy(self, t1, t2):                                                     
       '''RCCSD correlation energy'''                                                               
       fock = self._eris.fov.copy()
       e = 2*lib.einsum('ia,ia', fock, t1)    
       tau = lib.einsum('ia,jb->ijab',t1,t1)                                                         
       tau += t2 
       eris_ovov = lib.einsum("Lia,Ljb->iajb", self._eris.Lov, self._eris.Lov)
       e += 2*lib.einsum('ijab,iajb', tau, eris_ovov)                                                
       e -=  lib.einsum('ijab,ibja', tau, eris_ovov)                                                
       #if abs(e.imag) > 1e-4:
       #    logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)                   
       return e.real                       

    #Note: For the time being, we will not use the most optmal way to update the amplitudes. It can be taken care later on..
