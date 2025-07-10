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


    def t1_transform(self, imds, t1, M, Moo, Mov, Mov_t2):
        #fetch the 3-center integrals in MO basis


        #in this function we will build only those terms where t1_aa contributes. also these terms will be iteratively updated. initialization could be done outside the iterative loop as well. 


# first construct intermediates:
#      
        Moo_aa = Moo[0]
        Moo_ia = Moo[1]
        Mov_aa = Mov[0]
#       Mov_ai = Mov[1]
#      JOO (active-active)
        Joo_aa = numpy.array(imds.Joo).copy()
        Joo_aa += lib.einsum("Lic,jc->Lij", self.Lov_aa, t1)

#      JVV  (active-active, active-inactive)
        Jvv_aa = numpy.array(imds.Jvv).copy()          
        Jvv_aa -= lib.einsum("Lkb,ka->Lab", self.Lov_aa, t1)

#      JOV (active-active)
        Jov_aa  = numpy.array(imds.Jov).copy()
        Jov_aa += lib.einsum("Lac,ic->Lia", self.Lvv_aa, t1)
        Jov_aa -= lib.einsum("Lki,ka->Lia", Joo_aa, t1)  

#Now construct Fock matrix: (active-active, active-inactive)
        Foo_aa  = numpy.array(imds.Foo).copy()
        Foo_aa += lib.einsum("Lij,L->ij", self.Loo_aa, M)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",self.Loo_aa,Moo_aa)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",self.Loo_ia,Moo_ia)
        
       #are we missng any term here?
         
        Fvv_aa  = numpy.array(imds.Fvv).copy()
        Fvv_aa += lib.einsum("Lab,L->ab",self.Lvv_aa,M) 
        Fvv_aa += lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_aa)

        #Fov (active-active)
        Fov_aa  = numpy.array(imds.Fov).copy()
        Fov_aa += lib.einsum("Lia,L->ia",self.Lov_aa,M)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",self.Lov_aa,Moo_aa)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",self.Lov_ia,Moo_ia)

        Joo = [Joo_aa]
        Jvv = [Jvv_aa]
        Jov = [Jov_aa]

        Foo = Foo_aa
        Fvv = Fvv_aa
        Fov = Fov_aa

        return Joo, Jvv, Jov, Foo, Fvv, Fov

    def create_M_intermediates(self, t1, t2):
        #construct the intermediates for the t2 update:
        #M0
        M0 = lib.einsum("Lkc,kc->L", self.Lov_aa, t1)*2

        #Moo
        Moo_aa = lib.einsum("Lic,jc->Lij", self.Lov_aa, t1)
        Moo_ia = lib.einsum("Lic,jc->Lij", self.Lov_ia, t1)

        #Mov
        Mov_aa = lib.einsum("Lac,ic->Lia", self.Lvv_aa, t1)
        Mov_ai = lib.einsum("Lac,ic->Lia", self.Lvv_ia, t1)

        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
        Mov_t2 = lib.einsum("Lme, imae -> Lia", self.Lov_aa, t2_antisym)

        Moo = [Moo_aa, Moo_ia]
        Mov = [Mov_aa, Mov_ai]
        return M0, Moo, Mov, Mov_t2


    def add_t2_to_fock(self, Fvv, Foo, Mov_t2):    


        #construct antisymmetrized t2:

        #generate the intermediate with full arrays 
        Foo += lib.einsum("Lie,Lje->ij", self.Lov_aa, Mov_t2)
        Fvv -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_t2)

        return Foo, Fvv


    def R1_residue_active(self, imds, t1, t2, Fov, Fvv, Foo, M0, Mov, Mov_t2 ):


        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
       
        R1 = imds.R1.copy()

        

        Foo += lib.einsum("me,ie->im", Fov, t1)*0.5

        R1 -= lib.einsum("im, ma -> ia", Foo, t1)       

        Fvv -= lib.einsum("me, ma -> ae", Fov, t1)*0.5
 
        R1 += lib.einsum("ae, ie -> ia", Fvv, t1)
        R1 -= lib.einsum("me, imae -> ia", Fov, t2_antisym) #many terms
        R1 += lib.einsum("Lia, L -> ia", self.Lov_aa, M0)
        R1 -= lib.einsum("Lim, Lma -> ia", self.Loo_aa, Mov[0])
        R1 -= lib.einsum("Lim, Lma -> ia", self.Loo_aa, Mov_t2)
        R1 += lib.einsum("Lae, Lie -> ia", self.Lvv_aa, Mov_t2)

        #discard the updated Foo, Fvv contribution:
        Foo -= lib.einsum("me,ie->im", Fov, t1)*0.5
        Fvv += lib.einsum("me, ma -> ae", Fov, t1)*0.5
 

        return R1


    def R2_residue_active(self, t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo):

        
        Jov = Jov[0]
        Joo = Joo[0]
        Jvv = Jvv[0]

        #factorized part of the residue:
        R2 = lib.einsum("Lia, Ljb -> ijab", Jov, Jov)
        #PPL 
        Waebf = lib.einsum("Lae, Lbf -> abef", Jvv, Jvv)
#       R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)#three possibilities (ii, ia, ai)
        R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)

        #HHL
        Wijmn = lib.einsum("Lim, Ljn -> ijmn", Joo, Joo)
        R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2) #three possibilities (aa, ai, ia)

        #Fock matrix contribution:

        Foo += lib.einsum("me,ie->im", Fov, t1)
        R2_tmp = -lib.einsum("im, mjab -> ijab", Foo, t2) #only one possibility m has to be inactive.

        Fvv -= lib.einsum("me, ma -> ae", Fov, t1)

      #it is better to have a 
        R2_tmp += lib.einsum("ae, ijeb -> ijab", Fvv, t2) # only one possibility e has to be inactive.

        #N3V3 terms:

        W_jebm = lib.einsum("Ljm, Lbe -> jebm", Joo, Jvv)
#       R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2) # em should be ii, ia, ai types

        R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2)


        W_jema = lib.einsum("Ljm, Lae -> jema", Joo, Jvv)
        R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2) # em should be ii, ia, ai types


        #Non DCA terms:
        Ijemb, Ijebm, Imnij = self.t2_transform_quadratic(t2)        

        R2_tmp -= lib.einsum("jebm, imae -> ijab", Ijebm, t2)
        R2_tmp += lib.einsum("jebm, imae -> ijab", Ijebm, t2)

        R2_tmp -= lib.einsum("jema, imeb -> ijab", Ijemb, t2)

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        return R2


    def t2_transform_quadratic(self,t2):

        #I_mn^ij 

        Vmenf = lib.einsum("Lme, Lnf -> menf", self.Lov_aa, self.Lov_aa)
        Imnij = lib.einsum("menf, ijef -> ijmn", Vmenf, t2) #ef should be ii, ia, ai types

        #I^je_bm
        Vnemf = lib.einsum("Lne, Lmf -> nemf", self.Lov_aa, self.Lov_aa)
        Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2) #nf should be ii, ia, ai types

        #I^je_mb
        Ijemb = lib.einsum("nemf, jnfb -> jemb", Vnemf, t2) # #nf should be ii, ia, ai types

        return Ijemb, Ijebm, Imnij
    

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


        M0, Moo, Mov, Mov_t2 = self.create_M_intermediates(t1, t2)
        Joo, Jvv, Jov, Foo, Fvv, Fov = self.t1_transform(imds, t1, M0, Moo, Mov, Mov_t2)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mov_t2)

        R1 = self.R1_residue_active(imds, t1, t2, Fov, Fvv, Foo, M0, Mov, Mov_t2 )
        R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo)

        res1 = R1/ self._eris.eia[numpy.ix_(self.act_hole, self.act_particle)]
        res2 = R2/ self._eris.D[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
 
        t1 += res1
        t2 += res2
        res = numpy.linalg.norm(res1) + numpy.linalg.norm(res2)

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

      #self._e_corr = e_corr
      #self._e_tot = self.mf.e_tot + self._e_corr


      return t1, t2


