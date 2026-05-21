from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass


class MPCC_HL:
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
    def nvir(self):
        return self.mf.mol.nao - self.nocc
    
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    @property
    def act_hole(self):
        return self.frags[0]

    @property
    def nact_particle(self):
        return len(self.act_particle)

    @property
    def nact_hole(self):
        return len(self.act_hole)

    @property
    def act_particle(self):
        return self.frags[1]


    def kernel(self, t1=None, t2=None):

        # NOTE Do we want to initialize t1 and t2?

        if t1 is None and t2 is None:
            t1, t2 = self.init_amps()
           
        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()
        adiis.space = 8

        e_corr = None
        while err > self.ll_con_tol and count < self.ll_max_its:

            res, t1_new, t2_new = self.updated_amps(t1, t2)
            if self.diis:
                t1_new, t2_new = self.run_diis(t1_new, t2_new, adiis)
            else:
                t1_new, t2_new = t1_new, t2_new

            t1, t2 = t1_new, t2_new
            t1_new, t2_new = None, None  # free memory
            
            count += 1
            err = res
            # NOTE change this to logger!
            print(f"It {count}; residual {res:.6e}")

#extract the active part of the t1 and t2 amplitudes
        t1_active = t1[np.ix_(self.act_hole, self.act_particle)]
        t2_active = t2[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]

        return t1_active, t2_active

    def updated_amps(self, t1, t2):
        """
        Following Table XXX in Future Paper
        """

        t1_renorm, L1_renorm = self.get_renormalized_t1(t1, -t1)

        
        # Contractions
        X, Xoo, Xvo = self.get_X_t1(t1_renorm)
        

        Joo, Jvo, Jvv = self.get_J_t1(Xoo, Xvo, t1_renorm)

        Xt, Xoo, Xov = self.get_X_L1(Jvv, Xoo, Jvo, L1_renorm)
        
        Joo, Jov, Jvv = self.get_J_L1(Joo, Xoo, Xov, Xvo, Jvv, L1_renorm)

        Xvo_t2 = self.get_X_t2(t2)
    
        Jvo = self.get_J_t2(Jvo, Xvo_t2)

      # Foo, Fvv, Fov = self.get_F(t1_renorm, X, Xoo, Xvo, Jvo)

        Foo, Fvv, Fov = self.get_F(t1_renorm, X, Xt, Xoo, Xvo, Jvo)

        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)    

        Ω = self.get_Ω(X, Xt, Xvo, Xvo_t2, Foo, Fvv, Fov, t1_renorm, t2)
        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1_renorm, Joo, Jvv)

        #make the the inactive residuals zero
        
        Ω_dummy = np.zeros_like(Ω)
        res2_dummy = np.zeros_like(res2) 
         
        for frag in self.frags:
           act_hole = frag[0]
           act_particle = frag[1]
           #make a slice based on active_hole and active_particle
           Ω_dummy[np.ix_(self.act_particle, self.act_hole)] = Ω[np.ix_(self.act_particle, self.act_hole)].copy()
           res2_dummy[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)] = res2[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)].copy()

        res1 = Ω_dummy.T / self._eris.eia
        res2 = res2_dummy / self._eris.D

        print("norm of the residuals: res1, res2", np.linalg.norm(res1), np.linalg.norm(res2))

    #    res = np.linalg.norm(res1)/(self.nocc * self.nvir) + np.linalg.norm(res2)/(self.nocc**2 * self.nvir**2)
        res = np.linalg.norm(res1) + np.linalg.norm(res2)
        
        t1 -= res1
        t2 -= res2
        return res, t1, t2
    
    @dataclass
    class _IMDS_T1:
       Joo: np.ndarray
       Jvv: np.ndarray
       Jov: np.ndarray
       Foo: np.ndarray
       Fvv: np.ndarray
       Fov: np.ndarray
       Xoo: np.ndarray
       Xvo: np.ndarray
    
    def get_t1_imds(self, t1, t2):

        X, Xoo, Xvo, Xvo_t2 = self.get_X(t1, t2)

        Joo, Jvo, Jvv = self.get_J_all(Xoo, Xvo, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Xvo_t2, Jvo)

        #store Joo, Jvo, Foo etc..
        imds_t1 = self._IMDS_T1(Joo=Joo, Jvv=Jvv, Jov=Jvo, Foo=Foo, Fvv=Fvv, Fov=Fov, Xoo=Xoo, Xvo=Xvo)
        return imds_t1
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

    @staticmethod
    def get_renormalized_t1(t1, L1, conv_tol=1e-8, max_cycle=50):

        #initialize the renormalized t1 and L1
        t1_renorm = t1.copy()
        L1_renorm = L1.copy()

        for niter in range(max_cycle):
            t1_prev = t1_renorm.copy()
            L1_prev = L1_renorm.copy()

            den_oo = lib.einsum("ic, jc ->ij", L1_prev, t1_prev)
            den_vv = -lib.einsum("kb, ka ->ab", L1_prev, t1_prev)

            #update t1 and L1 until the renormalized amplitudes converge
            t1_renorm = t1 - lib.einsum("ia,ij->ja", t1, den_oo)
            L1_renorm = L1 + lib.einsum("ia,ab->ib", t1, den_vv)

            delta = np.max([np.linalg.norm(t1_renorm - t1_prev),
                             np.linalg.norm(L1_renorm - L1_prev)])
            
            if delta < conv_tol:
                break

        print (f"Renormalization converged in {niter+1} iterations")

        return t1_renorm, L1_renorm

    def get_X_t1(self, t1):

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return X, Xoo, Xvo

    def get_X_t2(self, t2):

        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
        Xvo_t2 = lib.einsum("Lkc,ikac ->Lai", self._eris.Lov, t2_antisym)

        return Xvo_t2

    def get_J_t1(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) 

        return Joo, Jvo, Jvv


    def get_J_t2(self, Jvo, Xvo_t2):
        Jvo += Xvo_t2
        return Jvo

    def get_X_L1(self, Jvv, Xoo, Jvo, L1):
        Xov  = lib.einsum("Lba,ib->Lia", Jvv, L1)
        Xoo += lib.einsum("Laj,ia->Lij", Jvo, L1) #think about the sign here
        Xt   = lib.einsum("Lai,ia->L", self._eris.Lvo, L1)*2.0
        return Xt, Xoo, Xov


    def get_J_L1(self, Joo, Xoo, Xov, Xvo, Jvv, L1):

        Joo += Xoo
        Jov  = (
            Xov + self._eris.Lov - lib.einsum("Lij,ja->Lia", Joo, L1)
        )

        Jvv = Jvv - lib.einsum("Lbk,ka->Lab", Xvo, L1) 
        return Joo, Jov, Jvv 

    def get_F(self, t1, X, Xt, Xoo, Xvo, Jvo):

        nocc, nvir = t1.shape
        Foo = self._eris.foo.copy()
#       Foo[np.diag_indices(nocc)] -= np.diag(self._eris.foo)
        Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X+Xt)
        Foo -= lib.einsum("Lmj,Lim->ij", self._eris.Loo,Xoo)

        Fvv = self._eris.fvv.copy()
#       Fvv[np.diag_indices(nvir)] -= np.diag(self._eris.fvv)
        Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X+Xt) 
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo)
              
        Fov  = self._eris.fov.copy()
        Fov += lib.einsum("Ljb,L->jb", self._eris.Lov, X+Xt)
#       Fov += lib.einsum("Lbj,L->jb", Jvo, X)
        Fov -= lib.einsum("Lji,Lib->jb", Xoo, self._eris.Lov)

        return Foo, Fvv, Fov

    def add_t2_to_fock(self, Fvv, Foo, Xvo_t2):    

        Foo += lib.einsum("Lie,Lej->ij", self._eris.Lov,Xvo_t2)
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo_t2)

        return Foo, Fvv

    def get_Ω_kallay(self, X, Xvo, Foo, Fvv, Fov, Joo, Jvv, t1, t2):

        Ω = self._eris.fov.copy()

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1) 

        Ω += lib.einsum("ib,ab -> ia", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ia", t1, Foo_tmp)

        Ω += lib.einsum("Lai,L->ia", self._eris.Lvo, X)
        Ω -= lib.einsum("Laj,Lji->ia", Xvo, self._eris.Loo)

        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        Ω += lib.einsum("ijab,jb->ia", t2_antisym, Fov) # term C

        Mvo_t2 = lib.einsum("Lkc,ikac ->Lai", self._eris.Lov, t2_antisym)

        Ω -= lib.einsum("Laj,Lji->ia", Mvo_t2, Joo) #term B
        Ω += lib.einsum("Lae,Lei->ia", Jvv, Mvo_t2) #term A

        return Ω

    def get_Ω(self, X, Xt, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += 0.5*lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= 0.5*lib.einsum("lb,la->ab",Fov,t1) 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X+Xt)

        Ω += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2)

        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)
   
        del Foo_tmp, Fvv_tmp

        return Ω

    def get_L(self, X, Xov, Xvo_t2, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += 0.5*lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= 0.5*lib.einsum("lb,la->ab",Fov,t1) 

        L = self._eris.fov.copy()

        L -= lib.einsum("Lja,Lji->ia", Xov, self._eris.Loo)
        L -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo)
        L += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        L += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2)

        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)
   
        del Foo_tmp, Fvv_tmp

        return L




    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv):

        Imbje, Imbej, Imnij = self.t2_transform_quadratic(t2)  


        res2 = lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        #PPL
        Waebf = lib.einsum("Lae, Lbf -> abef", Jvv, Jvv)
        res2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)

        #HHL
        #get the active part of Joo

        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo, Joo) + Imnij
        res2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2) 
        
        #norm of the active part of the HHL contribution to the R2 residue
        res2_active = res2[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
        #print("norm of the active part of the HHL contribution to the R2 residue", np.linalg.norm(res2_active)) 


        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)

        #extract the active part of the tmp matrix
        tmp_active = tmp[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
        #print("norm of the active part of the R2 residue after Foo and Fvv contractions", np.linalg.norm(tmp_active)) 

## N3V3
        W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv) - Imbje 
        tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2) # em should be ii, ia, ai types

        tmp_active = tmp[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
        #print("norm of the active part of the R2 residue after W_jebm contribution", np.linalg.norm(tmp_active))


        W_jema = lib.einsum("Lmj, Lae -> maje", Joo, Jvv) - Imbje*0.5
        tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2) # em should be ii, ia, ai types

        tmp_active = tmp[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
        #print("norm of the active part of the R2 residue after W_jema contribution", np.linalg.norm(tmp_active))
## more DCA like terms:
        tmp -= lib.einsum("mbej, imae -> ijab", Imbej, t2) #should it not be antisym

        tmp_active = tmp[np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
        #print("norm of the active part of the R2 residue after Imbej contribution", np.linalg.norm(tmp_active))

        res2 += tmp + tmp.transpose(1,0,3,2) 


        del Foo_tmp, Fvv_tmp 
       
        return res2

    def t2_transform_quadratic(self,t2):

        #I_mn^ij 
        Vnemf = lib.einsum("Lne, Lmf -> nmef", self._eris.Lov, self._eris.Lov)
        Imnij = lib.einsum("mnef, ijef -> mnij", Vnemf, t2) #ef should be ii, ia, ai types
        #I^je_bm
        Imbej = lib.einsum("nmef, jnbf -> mbej", Vnemf, t2)
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2) # #nf should be ii, ia, ai types

        return Imbje, Imbej, Imnij

    def init_amps(self):

        t2 = lib.einsum("Lai,Lbj->ijab", self._eris.Lvo, self._eris.Lvo)
        t2 /= self._eris.D          
        t1 = self._eris.fov/self._eris.eia

#       energy = self.energy(t1, t2) 
#       print("Initial energy:", energy)

        return t1, t2
    def get_energy(self, t1, t2):
        """
        Calculate the MPCC energy using the current amplitudes.
        """
        X, Xoo, Xvo = self.get_X(t1)
        Joo, Jvo, Jvv = self.get_J(Xoo, Xvo, t1)
        Yvo = self.get_t2_Yvo(t2)
        e1 = lib.einsum("Lij,ja->Lai", Xoo, t1) + lib.einsum("L,ia->Lai", X, t1) + Jvo
        e_corr = lib.einsum("Lai,Lai", e1, Yvo)
        return e_corr
    
    def energy(self, t1, t2):                                                     
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


