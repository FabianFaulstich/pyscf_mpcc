from pyscf import lib, df
from pyscf.lib import logger
import numpy as np


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

    def kernel(self, t1=None, t2=None):

        # NOTE Do we want to initialize t1 and t2?

        if t1 is None and t2 is None:
            t1, t2 = self.init_amps()
           
        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        e_corr = None
        while err > self.ll_con_tol and count < self.ll_max_its:

            res, e_corr, t1_new, t2_new = self.updated_amps(t1, t2)
            if self.diis:
                t1_new, t2_new = self.run_diis(t1_new, t2_new, adiis)
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
        #Run update amplitudes to get the intermediate quantities

        return t1, t2

    def updated_amps(self, t1, t2):
        """
        Following Table XXX in Future Paper
        """

        # Setp 1 
        Xoo, Xvo, X = self.get_X(t1)

        # Step 2
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        
        # Step 3
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Jvo)

        # Step 4
        Ω = self.get_Ω(X, Xvo, Foo, Fvv, Fov, t1)

        # Step 5
        # NOTE rewrite in terms of Y
        Ω = self.update_Ω(Ω, Fov, t2)

        # Step 6
        # NOTE remove when T2 is factorized
        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)

        # Step 7 
        # NOTE remove when T2 is factorized
        ΔE = self.energy(t1, t2)  # use the low-level energy calculation

        #make the active residue to go to zero

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            #print("active hole array", act_hole)
            #make a slice based on active_hole and active_particle
            Ω[np.ix_(act_particle, act_hole)] = 0.0
            res2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0


        res1 = Ω.T / self._eris.eia
        # NOTE remove res2 when T2 is factorized
        res2 = res2 / self._eris.D
        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 += res1
        # NOTE Remove this and just return Y
        t2 += res2

        return res, ΔE, t1, t2

    def get_t1_imds(self, t1, t2):

        X, Xoo, Xvo = self.get_X(t1)

        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Jvo)

        Ω = self.get_Ω(X, Xvo, Foo, Fvv, Fov, t1)

        Ω = self.update_Ω(Ω, Fov, t2)
        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
        #store Joo, Jvo, Foo, Fvv, Fov, Ω, res2 in a container
        imds = {
            'Joo': Joo,
            'Jvo': Jvo,
            'Foo': Foo,
            'Fvv': Fvv,
            'Fov': Fov,
            'Ω': Ω,
            'res2': res2
        }
        return imds
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

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return Xoo, Xvo, X

    def get_J(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + self._eris.Lvo - lib.einsum("Lij,ja->Lai", Joo, t1)
        )

        return Joo, Jvo

    def get_F(self, t1, X, Xoo, Xvo, Jvo):

        Foo = self._eris.foo.copy()
        Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
        Foo -= lib.einsum("Lki,Lkj->ij", self._eris.Loo,Xoo)

        Fvv = self._eris.fvv.copy()
        Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X) 
        Fvv -= lib.einsum("Lka,Lbk->ab",self._eris.Lov,Xvo)
              
        Fov  = self._eris.fov.copy()
        Fov += lib.einsum("Lbj,L->jb", self._eris.Lvo, X)
        Fov -= lib.einsum("Lij,Lib->jb", Xoo, self._eris.Lov)

        return Foo, Fvv, Fov

    def get_Ω(self, X, Xvo, Foo, Fvv, Fov, t1):

        Ω = -1 * lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω += lib.einsum("Lia,L->ai", self._eris.Lov, X)

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("kb,ka->ab",Fov,t1)*0.5 

        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        return Ω

    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1):

        Fvv_tmp = Fvv.copy() 
        Foo_tmp = Foo.copy()

        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)

        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("ki,kjab->ijab", Foo_tmp, t2)

        res2 = tmp + tmp.transpose(1,0,3,2)
        res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        return res2


    def update_Ω(self, Ω, Fov, t2):

        t2_antisym = 2*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)
        return Ω

    def init_amps(self):

        t2 = lib.einsum("Lia,Ljb->ijab", self._eris.Lov, self._eris.Lov)
        t2 /= self._eris.D          
        t1 = self._eris.fov/self._eris.eia

        return t1, t2
    def get_energy(self, t1, t2):
        """
        Calculate the MPCC energy using the current amplitudes.
        """
        X, Xoo, Xvo = self.get_X(t1)
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        Yvo = self.get_t2_Yvo(t2)
        e1 = lib.einsum("Lij,ja->Lai", Xoo, t1) + lib.einsum("L,ia->Lai", X, t1) + Jvo
        e_corr = lib.einsum("Lai,Lai", e1, Yvo)
        return e_corr
    
    def energy(self, t1, t2):                                                     
       '''RCCSD correlation energy'''                                                               
       
       #nocc, nvir = t1.shape                                                                        
       #fock = eris.fock
       #e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)    
       e = 0.0                                          
       tau = np.einsum('ia,jb->ijab',t1,t1)                                                         
       tau += t2 
       eris_ovov = lib.einsum("Lia,Ljb->iajb", self._eris.Lov, self._eris.Lov)
       e += 2*np.einsum('ijab,iajb', tau, eris_ovov)                                                
       e +=  -np.einsum('ijab,ibja', tau, eris_ovov)                                                
       #if abs(e.imag) > 1e-4:
       #    logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)                   
       return e.real                       

    #Note: For the time being, we will not use the most optmal way to update the amplitudes. It can be taken care later on..


