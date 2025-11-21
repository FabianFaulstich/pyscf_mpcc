from numpy.lib.index_tricks import ix_
from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass

from pyscf.mpcc import mpcc_tools

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
 
        if 'll_update' in kwargs:
            self.kernel_type = kwargs['kernel_type']
        else:
            self.kernel_type = 'factorized'

        self._kernels = {
                'factorized': self._factorized_kernel,
                'unfactorized': self._unfactorized_kernel, 
                }

        self.frags = frags

        # NOTE can be potentially initialized
       # self.t1 = None
       # self.t2 = None

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

        try:
            func = self._kernels[self.kernel_type]
        except KeyError:
            raise ValueError(f'Unknown low-level kernel theory: {kind}')
        
        return func(t1, t2_act ,**kwargs)

    def _factorized_kernel(self, t1=None, t2_act=None, Y=None):

        # NOTE Do we want to initialize t1 and t2?
        # WARNING t2 is incorrectly initialized here!
        '''
        if t1 is None and t2 is None and Y is None:
            
            # NOTE So far this is an N^5 step. 
            # So potentially initilaization should also be done in the factorized manner.
            t1, t2, Y = self.init_amps() 
            
            t2_new = []
            for frag in self.frags:
                act_hole = frag[0]
                act_particle = frag[1]
                t2_new.append(t2[np.ix_(act_hole, act_hole, act_particle, act_particle)])
            t2 = t2_new

            # NOTE remove this once df it running
            if False:
                t2_app = lib.einsum("LRai, LRbj -> ijab", Y, Y)    

                fro_rel = np.linalg.norm(t2 - t2_app) / np.linalg.norm(t2)
                max_abs = np.max(np.abs(t2 - t2_app))

                print(f'Test for init:')
                print(f"4th-order relative Frobenius error: {fro_rel:.3e}")
                print(f"4th-order max abs entry error:     {max_abs:.3e}")
        '''

        res = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        e_corr = None
        while res > self.ll_con_tol and count < self.ll_max_its:

            res, t1_it, Δt2s_o, Δt2s_v, Y = self.update_amps(t1, t2_act, Y)
            if self.diis:
                t1_it  = self.run_diis(t1_it, adiis)

            t1 = t1_it
            
            count += 1
            # NOTE change this to logger!
            print(f"It {count}; residual {res:.6e}")


        # NOTE non-iterative N^5 cost!
        t2 = self.get_t2(Y, t2_act, Δt2s_o, Δt2s_v)
        self._e_corr = self.get_energy(t1, t2) 

        return t1, t2, Y

    def update_amps(self, t1, t2_act, Y):
        """
        Following Table XXX in Future Paper
        """
        # Setp 1 
        Xoo, Xvo, X = self.get_X(t1)

        # Step 2
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
       
        # Step 3
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        # NOTE remove this once t2 is correctly fatorized
        #t2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
        #t2 += t2/self._eris.D

        # Step 4 
        Ω = self.get_Ω(X, Xvo, Foo, Fvv, Fov, t1, Y)

        # NOTE discuss the grouping of actions, 5-7/8 seems to be fit for one step
        # Step 5 
        Foo, Fvv = self.update_F(Foo, Fvv, Fov, t1)

        # Step 6 & 7
        D, Uoo, Uvv = self.get_D(Foo, Fvv)

        # NOTE Can we transform D insead of J, that way we don't have to transform Y back!
        # Step 8
        Jvo = self.update_J(Jvo, Uoo, Uvv)

        # Step 9 
        Y, Yt = self.update_Y(Joo, Jvo, D, Uvv, Uoo, Y)

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
    
    def run_diis(self, t1, adiis):

        vec = self.amplitudes_to_vector(t1)
        t1 = self.vector_to_amplitudes(adiis.update(vec))

        return t1

    def amplitudes_to_vector(self, t1):
        nov = self.nocc * self.nvir
        vector = t1.ravel()
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nocc * self.nvir
        t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
        return t1


    def get_X(self, t1):

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return Xoo, Xvo, X

    def get_J(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        return Joo, Jvo

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

    def update_F(self, Foo, Fvv, Fov, t1):

        Fvv = Fvv - lib.einsum("ic,ib -> bc", Fov, t1) 
        Foo = Foo + lib.einsum("kc,ic -> ki", Fov, t1)
 
        return Foo, Fvv

    def get_D(self, Foo, Fvv):

        Foo = 0.5 * (Foo + Foo.T)
        Fvv = 0.5 * (Fvv + Fvv.T)

        #print(f"Symmetry in Foo: {np.linalg.norm(Foo - Foo.T)}")
        #print(f"Symmetry in Fvv: {np.linalg.norm(Fvv - Fvv.T)}")

        e_oo, Uoo = np.linalg.eigh(Foo)
        e_vv, Uvv = np.linalg.eigh(Fvv)
        eia = lib.direct_sum("-i+a->ia", e_oo, e_vv)
       
        D = mpcc_tools.piv_chol_tensor(eia)

        return D, Uoo, Uvv

    def update_J(self, Jvo, Uoo, Uvv):

        Jvo = lib.einsum("ab, ij, Lai -> Lbj", Uvv, Uoo, Jvo)

        return Jvo

    def update_Y(self, Joo, Jvo, D, Uvv, Uoo, Y):

        Yt = Jvo[:, None, :, :] *D.transpose(2, 1, 0)[None, :, :, :]

        # NOTE REMOVE THE CHECK IF WE ARE HAPPY!!!
        if False:
            t2_app = lib.einsum("LRai, LRbj -> ijab", Yt, Yt)    
            
            fro_rel = np.linalg.norm(t2 - t2_app) / np.linalg.norm(t2)
            max_abs = np.max(np.abs(t2 - t2_app))

            print(f'Test in update_Y:')
            print(f"4th-order relative Frobenius error: {fro_rel:.3e}")
            print(f"4th-order max abs entry error:     {max_abs:.3e}")

        Y = lib.einsum("LRbj, ab, ij -> LRai", Yt, Uvv, Uoo)
        return Y, Yt

    def include_t2_active(self, Foo, Fvv, Fov, t2_act, Y, Ω, tol = 1e-6, count_tol = 100):
        
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
  
            Δt2_o = -np.einsum('kI, kjab -> Ijab ',Foo[np.ix_(act_hole, inact_hole)], Δt2)
            Δt2_o -=  np.einsum('kJ, ikab -> Jiab ',Foo[np.ix_(act_hole, inact_hole)], Δt2)
  
            Δt2_v = np.einsum('cA, ijcb -> ijAb ',Fvv[np.ix_(act_particle, inact_particle)], Δt2)
            Δt2_v += np.einsum('cB, ijac -> ijBa ',Fvv[np.ix_(act_particle, inact_particle)], Δt2)
                 
            Δt2_o_it = np.copy(Δt2_o)
            Δt2_v_it = np.copy(Δt2_v)

            count = 0 
            acc = np.infty
            while (acc > tol and count< count_tol):
                Δt2_o_it -= np.einsum('jk,ikab -> ijab',Foo[np.ix_(act_hole, act_hole)], Δt2_o)
                Δt2_o_it -= np.einsum('ik,kjab -> ijab',Foo[np.ix_(inact_hole, inact_hole)], Δt2_o)
                Δt2_o_it = Δt2_o_it / eia_o
                
                Δt2_v_it += np.einsum('bc,ijac -> ijab', Fvv[np.ix_(act_particle, act_particle)], Δt2_v)
                Δt2_v_it += np.einsum('ac,ijcb -> ijab', Fvv[np.ix_(inact_particle, inact_particle)], Δt2_v)
                Δt2_v_it = Δt2_v_it / eia_v

                acc_o = np.linalg.norm(Δt2_o_it - Δt2_o)
                acc_v = np.linalg.norm(Δt2_v_it - Δt2_v)
                acc = acc_o + acc_v
                Δt2_o -= Δt2_o_it 
                Δt2_v -= Δt2_v_it 

                count += 1
            
            print(f'    Iter. T2 correction finished in {count}/{count_tol} steps at {acc:.2e} accuracy.')

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
        return t1, t2, Y

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
