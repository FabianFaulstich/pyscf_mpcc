from pyscf import df
from pyscf import lib
import numpy

class MPCC_HL:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
        self.frag = frags
    
        self.diis = True

        self.ll_con_tol = kwargs.get("ll_con_tol")
        self.ll_max_its = kwargs.get("ll_max_its")

        self._set_integral_blocks()
    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc
        
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    @property
    def naux(self):
        return self.with_df.get_naoaux()

    @property
    def act_hole(self):
        return self.frag[0]

    @property
    def nact_particle(self):
        return len(self.act_particle)
    
    @property
    def nact_hole(self):
        return len(self.act_hole)

    @property
    def act_particle(self):
        return self.frag[1]
        
    @property
    def inact_hole(self):
        return numpy.setdiff1d(numpy.arange(self.nocc), self.act_hole)
        
    @property
    def inact_particle(self):
        return numpy.setdiff1d(numpy.arange(self.nvir), self.act_particle)

        # at this point we will classify integralsi: 

    def _set_integral_blocks(self):

        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle
        naux_idx = numpy.arange(self.naux)
        
        self.Loo_ia = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, act_hole)]
        self.Loo_aa = self._eris.Loo[numpy.ix_(naux_idx, act_hole, act_hole)]

        self.Lvv_aa = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, act_particle)]
        self.Lvv_ia = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, act_particle)]

        self.Lov_ia = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, act_particle)]
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)]


    def t1_transform(self, imds, t1, M, Moo, Mvo, Mvo_t2):
        #fetch the 3-center integrals in MO basis


        #in this function we will build only those terms where t1_aa contributes. also these terms will be iteratively updated.
        #initialization could be done outside the iterative loop as well. 

# first construct intermediates:
#      
        Moo_aa = Moo[0]
        Moo_ia = Moo[1]
        Mvo_aa = Mvo[0]
#       Mvo_ai = Mvo[1]
#      JOO (active-active)
        Joo_aa = numpy.array(imds.Joo).copy()
        Joo_aa += Moo_aa

#      JVV  (active-active, active-inactive)
        Jvv_aa = numpy.array(imds.Jvv).copy()      
        Jvv_aa -= lib.einsum("Lkb,ka->Lab", self.Lov_aa, t1)

#      JOV (active-active)
        Jvo_aa  = numpy.array(imds.Jvo).copy()

#       Jvo_aa += lib.einsum("Lac,ic->Lia", self.Lvv_aa, t1)
        Jvo_aa += Mvo_aa
        Jvo_aa += Mvo_t2
        Jvo_aa -= lib.einsum("Lji,ja->Lai", Joo_aa, t1)  

#Now construct Fock matrix: (active-active, active-inactive)
        Foo_aa  = lib.einsum("Lij,L->ij", self.Loo_aa, M)
        Foo_aa -= lib.einsum("Lmj,Lim->ij",self.Loo_aa,Moo_aa)
#       Foo_aa -= lib.einsum("Lmj,Lim->ij",self.Loo_ia,Moo_ia)###perhaps not required?
        Foo_aa_t1  = numpy.array(imds.Foo_t1).copy()
       
        Foo_aa_t1 += Foo_aa

        Foo_aa_t2  = numpy.array(imds.Foo_t2).copy()
        
        Foo_aa_t2 += Foo_aa
       #are we missng any term here?
         
#       Fvv_aa  = numpy.array(imds.Fvv).copy()
        Fvv_aa = lib.einsum("Lab,L->ab",self.Lvv_aa,M) 
        
#       Fvv_aa -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mvo_aa)
        Fvv_aa -= lib.einsum("Lmb,Lam->ab",self.Lov_aa,Mvo_aa)


        Fvv_aa_t1  = numpy.array(imds.Fvv_t1).copy()
        
        Fvv_aa_t1 += Fvv_aa

        Fvv_aa_t2  = numpy.array(imds.Fvv_t2).copy()
        
        Fvv_aa_t2 += Fvv_aa

        #Fov (active-active)
        Fov_aa  = numpy.array(imds.Fov).copy()
        
        Fov_aa += lib.einsum("Lia,L->ia",self.Lov_aa,M)
        Fov_aa -= lib.einsum("Lib,Lji->jb",self.Lov_aa,Moo_aa)
#       Fov_aa -= lib.einsum("Lib,Lji->jb",self.Lov_ia,Moo_ia)

        Joo = [Joo_aa]
        Jvv = [Jvv_aa]
        Jvo = [Jvo_aa]

        Foo = [Foo_aa_t1, Foo_aa_t2]
        Fvv = [Fvv_aa_t1, Fvv_aa_t2]
        Fov = Fov_aa

        return Joo, Jvv, Jvo, Foo, Fvv, Fov

    def create_M_intermediates(self, t1, t2):
        #construct the intermediates for the t2 update:
        #M0
        M0 = lib.einsum("Lkc,kc->L", self.Lov_aa, t1)*2.0

        #Moo
        Moo_aa = lib.einsum("Lic,jc->Lij", self.Lov_aa, t1)
        Moo_ia = lib.einsum("Lic,jc->Lij", self.Lov_ia, t1)

        #Mvo
        Mvo_aa = lib.einsum("Lac,ic->Lai", self.Lvv_aa, t1)
        Mvo_ia = lib.einsum("Lac,ic->Lai", self.Lvv_ia, t1)

        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
        Mvo_t2 = lib.einsum("Lme, imae -> Lai", self.Lov_aa, t2_antisym)

        Moo = [Moo_aa, Moo_ia]
        Mvo = [Mvo_aa, Mvo_ia]
        return M0, Moo, Mvo, Mvo_t2


    def add_t2_to_fock(self, Fvv, Foo, Mvo_t2):    

        #construct antisymmetrized t2:

        #generate the intermediate with full arrays 
        Foo_tmp = lib.einsum("Lie,Lej->ij", self.Lov_aa, Mvo_t2)
        Fvv_tmp = -lib.einsum("Lmb,Lam->ab",self.Lov_aa,Mvo_t2)

        Foo_aa_t1, Foo_aa_t2 = Foo
        Fvv_aa_t1, Fvv_aa_t2 = Fvv

        Foo = [Foo_aa_t1+Foo_tmp, Foo_aa_t2+Foo_tmp]
        Fvv = [Fvv_aa_t1+Fvv_tmp, Fvv_aa_t2+Fvv_tmp]

        return Foo, Fvv


    def R1_residue_active(self, imds, t1, t2, Fov, Fvv, Foo, M0, Mvo, Mvo_t2 ):


        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
       
        R1 = imds.R1.copy()
       
        Foo_t1, _ = Foo 
        Fvv_t1, _ = Fvv

        Foo_t1 += lib.einsum("ic,jc->ij", Fov, t1)*0.5

        R1 -= lib.einsum("ki,ka->ia", Foo_t1, t1)       

        Fvv_t1 -= lib.einsum("lb,la->ab", Fov, t1)*0.5
 
        R1 += lib.einsum("ab,ib->ia", Fvv_t1, t1)
        R1 += lib.einsum("me, imae -> ia", Fov, t2_antisym) #many terms
        R1 += lib.einsum("Lia, L -> ia", self.Lov_aa, M0)
        R1 -= lib.einsum("Lji,Laj -> ia", self.Loo_aa, Mvo[0])
        R1 -= lib.einsum("Lji,Laj -> ia", self.Loo_aa, Mvo_t2)
        R1 += lib.einsum("Lae, Lei -> ia", self.Lvv_aa, Mvo_t2)

        return R1

    def R2_residue_active(self, imds, t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo):

        
        Jvo = Jvo[0]
        Joo = Joo[0]
        Jvv = Jvv[0]

        Imbje, Imbej, Imnij = self.t2_transform_quadratic(t2)  

        Imbje += imds.Imbje_active
        Imbej += imds.Imbej_active
        Imnij += imds.Imnij_active

        R2 = imds.R2.copy()

        #factorized part of the residue:
        R2 += lib.einsum("Lai, Lbj -> ijab", Jvo, Jvo)
        #PPL 
        Waebf = lib.einsum("Lae, Lbf -> abef", Jvv, Jvv)
        R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)
        
        #HHL
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo, Joo) + Imnij
        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2) #three possibilities (aa, ai, ia)

        #Fock matrix contribution:

        _, Foo_t2 = Foo 
        _, Fvv_t2 = Fvv

        Foo_t2 += lib.einsum("ic,jc->ij", Fov, t1)
        R2_tmp = -lib.einsum("mi, mjab -> ijab", Foo_t2, t2) #only one possibility m has to be inactive.

        Fvv_t2 -= lib.einsum("lb,la->ab", Fov, t1)
        R2_tmp += lib.einsum("bc, ijac -> ijab", Fvv_t2, t2) # only one possibility e has to be inactive.

        #N3V3 terms:

        W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv) - Imbje 
        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2) # em should be ii, ia, ai types

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo, Jvv) - Imbje*0.5
        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2) # em should be ii, ia, ai types

        R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej, t2) #should it not be antisym?

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        Foo_t2 = Fvv_t2 = None

        return R2

    def t2_transform_quadratic(self,t2):

        #I_mn^ij 
        Vnemf = lib.einsum("Lne, Lmf -> nmef", self.Lov_aa, self.Lov_aa)
        Imnij = lib.einsum("mnef, ijef -> mnij", Vnemf, t2) #ef should be ii, ia, ai types
        #I^je_bm
        Imbej = lib.einsum("nmef, jnbf -> mbej", Vnemf, t2)
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2) # #nf should be ii, ia, ai types

        return Imbje, Imbej, Imnij
    

    def run_diis(self, t1, t2, adiis):

        vec = self.amplitudes_to_vector(t1, t2)
        t1, t2 = self.vector_to_amplitudes(adiis.update(vec))

        return t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        nov = self.nact_hole * self.nact_particle
        size = nov + nov * (nov + 1) // 2
        vector = numpy.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(t2.transpose(0, 2, 1, 3).reshape(nov, nov), out=vector[nov:])
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nact_hole * self.nact_particle
        t1 = vector[:nov].copy().reshape((self.nact_hole, self.nact_particle))
        # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
        t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
        t2 = t2.reshape(self.nact_hole, self.nact_particle, self.nact_hole, self.nact_particle).transpose(
            0, 2, 1, 3
        )
        return t1, numpy.asarray(t2, order="C")


    def updated_amps(self, imds, t1, t2):
        """
        Following Table XXX in Future Paper
        """


        M0, Moo, Mvo, Mvo_t2 = self.create_M_intermediates(t1, t2)
        Joo, Jvv, Jvo, Foo, Fvv, Fov = self.t1_transform(imds, t1, M0, Moo, Mvo, Mvo_t2)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mvo_t2)

        R1 = self.R1_residue_active(imds, t1, t2, Fov, Fvv, Foo, M0, Mvo, Mvo_t2 )
        R2 = self.R2_residue_active(imds, t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo)

        res1 = R1/ self._eris.eia[numpy.ix_(self.act_hole, self.act_particle)]
        res2 = R2/ self._eris.D[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
 
        t1 -= res1
        t2 -= res2
#        res = numpy.linalg.norm(res1) + numpy.linalg.norm(res2)
        res = numpy.linalg.norm(res2)

        return res, t1, t2


    def kernel(self, imds, t1full, t2full):

      #we have to extract the active only amplitudes from the full amplitudes:

      t1 = t1full[numpy.ix_(self.act_hole, self.act_particle)].copy()

      t2 = t2full[numpy.ix_(self.act_hole,self.act_hole, self.act_particle,self.act_particle)].copy()

      e_corr = None
      adiis = lib.diis.DIIS()
      err = numpy.inf
      print("Starting High-level MPCC iteration...")
      count = 0
      while err > self.ll_con_tol and count < self.ll_max_its:

          res, t1_new, t2_new = self.updated_amps(imds, t1, t2)
          if self.diis:
              t1_new, t2_new = self.run_diis(t1_new, t2_new, adiis)
          else:
              t1_new, t2_new = t1_new, t2_new

          t1 = t1_new
          t2 = t2_new       
          t1_new = None
          t2_new = None
          count += 1
          err = res
          # NOTE change this to logger!
          #print(f"It {count}; correlation energy {e_corr:.6e}; residual {res:.6e}")
          print(f"It {count}; residual {res:.6e}")

          # NOTE Checking CCSD correlation energy convergence, remove later
          e_cc = self.get_cc_energy(t1full, t2full, t1, t2)
          print(f'    CCSD correlation energy: {e_cc}')

      #self._e_corr = e_corr
      #self._e_tot = self.mf.e_tot + self._e_corr


      return t1, t2

    def get_cc_energy(self, t1, t2, t1_act, t2_act):

        act_hole = self.frag[0]
        act_particle = self.frag[1]
       
        t1[numpy.ix_(act_hole, act_particle)] = t1_act
        t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act


        fock = self._eris.fov.copy()
        e = 2*lib.einsum('ia,ia', fock, t1)    
        tau = lib.einsum('ia,jb->ijab',t1,t1)                                                         
        tau += t2 
        eris_ovov = lib.einsum("Lia,Ljb->iajb", self._eris.Lov, self._eris.Lov)
        e += 2*lib.einsum('ijab,iajb', tau, eris_ovov)                                                
        e -=  lib.einsum('ijab,ibja', tau, eris_ovov)                                                
        return e.real                       

