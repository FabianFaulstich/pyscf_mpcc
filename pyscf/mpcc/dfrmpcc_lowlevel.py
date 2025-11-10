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

    def kernel(self, t1=None, t2=None, Y=None):

        # NOTE Do we want to initialize t1 and t2?
        # WARNING t2 is incorrectly initialized here!
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

        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        e_corr = None
        while err > self.ll_con_tol and count < self.ll_max_its:

            res, t1_new, dt2_ll_o, dt2_ll_v, Y_new, Yt = self.update_amps(t1, t2, Y)
            if self.diis:
                # NOTE no t2 diis
                t1_new  = self.run_diis(t1_new, adiis)
            else:
                t1_new = t1_new

            t1, Y = t1_new, Y_new
            t1_new, Y_new = None, None  # free memory
            
            count += 1
            err = res
            # NOTE change this to logger!
            print(f"It {count}; residual {res:.6e}")


        # NOTE computing energy
        t2_new = -lib.einsum("LRai, LRbj -> ijab", Y, Y)
        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            t2_new[np.ix_(inact_hole, act_hole, act_particle, act_particle)] = dt2_ll_o[k]
            t2_new[np.ix_(act_hole, act_hole, inact_particle, act_particle)] = dt2_ll_v[k]

        
        e_corr = self.get_energy(t1, t2_new) 

        print(f"Correlation Energy: {e_corr}")
        self._e_corr = e_corr
        self._e_tot = self.mf.e_tot + self._e_corr
        #Run update amplitudes to get the intermediate quantities
        #imds_t1 = self.get_t1_imds(t1)

        return t1, t2_new, Y

    def update_amps(self, t1, t2, Y):
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

        # Step 8
        Jvo = self.update_J(Jvo, Uoo, Uvv)

        # Step 9 
        Y, Yt = self.update_Y(Joo, Jvo, D, Uvv, Uoo, Y)

        dt2_ll_o = []
        dt2_ll_v = []
        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            
            energy_den_o = lib.direct_sum("ia+jb->ijab", self._eris.eia[np.ix_(inact_hole, act_particle)], self._eris.eia[np.ix_(act_hole, act_particle)])
            energy_den_v = lib.direct_sum("ia+jb->ijab", self._eris.eia[np.ix_(act_hole, inact_particle)], self._eris.eia[np.ix_(act_hole, act_particle)])

            Ω[np.ix_(act_particle, act_hole)] = 0.0
              
            # Step 10  
            #NOTE input t2_act and correct Y
            dt2 = -lib.einsum("LRai, LRbj -> ijab", Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)], Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)])    
            Dt2 = t2[k] - dt2  
  
            diff_t2_o = -np.einsum('kI, kjab -> Ijab ',Foo[np.ix_(act_hole, range(n_occ))], Dt2)
            diff_t2_o -=  np.einsum('kJ, ikab -> Jiab ',Foo[np.ix_(act_hole, range(n_occ))], Dt2)
  
            diff_t2_v = np.einsum('cA, ijcb -> ijAb ',Fvv[np.ix_(act_particle, range(n_vir))], Dt2)
            diff_t2_v += np.einsum('cB, ijac -> ijBa ',Fvv[np.ix_(act_particle, range(n_vir))], Dt2)
                 
            t2_corr_o = np.copy(diff_t2_o[np.ix_(inact_hole, act_hole, act_particle, act_particle)])
            # t2_corr_o += diff_t2_o[np.ix_(act_hole, inact_hole, act_particle, act_particle)].transpose((1,0,2,3))
 
            t2_corr_v = np.copy(diff_t2_v[np.ix_(act_hole, act_hole, inact_particle, act_particle)])
            # t2_corr_v += diff_t2_v[np.ix_(act_hole, act_hole, act_particle, inact_particle)].transpose((0,1,3,2))
           
            acc = np.infty
            t2_corr_o_new = np.copy(t2_corr_o)
            t2_corr_v_new = np.copy(t2_corr_v)

            count = 0 
            tol = 1e-6
            while acc > tol:
                t2_corr_o_new -= np.einsum('jk,ikab -> ijab',Foo[np.ix_(act_hole, act_hole)], t2_corr_o)
                t2_corr_o_new -= np.einsum('ik,kjab -> ijab',Foo[np.ix_(inact_hole, inact_hole)], t2_corr_o)
                t2_corr_o_new = t2_corr_o_new / energy_den_o
                
                t2_corr_v_new += np.einsum('bc,ijac -> ijab', Fvv[np.ix_(act_particle, act_particle)], t2_corr_v)
                t2_corr_v_new += np.einsum('ac,ijcb -> ijab', Fvv[np.ix_(inact_particle, inact_particle)], t2_corr_v)
                t2_corr_v_new = t2_corr_v_new / energy_den_v

                acc_o = np.linalg.norm(t2_corr_o_new - t2_corr_o)
                acc_v = np.linalg.norm(t2_corr_v_new - t2_corr_v)
                acc = acc_o + acc_v
                t2_corr_o -= t2_corr_o_new
                t2_corr_v -= t2_corr_v_new

           
            dt2_ll_o.append(t2_corr_o)
            dt2_ll_v.append(t2_corr_v)

        # Step 11 use corrected t2 active to correct Omega

        res = Ω.T / self._eris.eia
        t1 -= res

        return np.linalg.norm(res), t1, dt2_ll_o, dt2_ll_v, Y, Yt
    
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

#    def get_J_all(self, Xoo, Xvo, t1):

#        Joo = Xoo + self._eris.Loo
#        Jvo = (
#            Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
#        )

#        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) 

#        return Joo, Jvo, Jvv

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
