from pyscf import df
from pyscf import lib
import numpy

class mpcc_HL:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
        self.frag = frags
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
        return self.frag[0][0]
            
    @property
    def act_particle(self):
        return self.frag[0][1]
        
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

        self.Lov_ia = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, act_particle)]
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)]


    def t1_transform(self, imds, t1):
        #fetch the 3-center integrals in MO basis


        #in this function we will build only those terms where t1_aa contributes. also these terms will be iteratively updated. initialization could be done outside the iterative loop as well. 


# first construct intermediates:
#      

#      JOO (active-active)
        Joo_aa = imds.Joo_aa
        Joo_aa += lib.einsum("Lic,jc->Lij", Lov_aa, t1_aa)

#      JVV  (active-active, active-inactive)
        Jvv_aa = imds.Jvv_aa           
        Jvv_aa -= lib.einsum("Lkb,ka->Lab", Lov_aa, t1_aa)

#      JOV (active-active)
        Jov_aa  = imds.Jov_aa
        Jov_aa += lib.einsum("Lac,ic->Lia", Lov_aa, t1_aa)
        Jov_aa -= lib.einsum("Lki,ka->Lia", Joo_aa, t1_aa)  

#Now construct Fock matrix: (active-active, active-inactive)
        Foo_aa  = imds.Foo_aa
        Foo_aa += lib.einsum("Lij,L->ij", Loo_aa, M)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",Loo_aa,Moo_aa)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",Loo_ia,Moo_ia)
        
       #are we missng any term here?
         
        Fvv_aa  = imds.Fvv_aa
        Fvv_aa += lib.einsum("Lab,L->ab",Lvv_aa,M) 
        Fvv_aa += lib.einsum("Lma,Lmb->ab",Lov_aa,Mov_aa)

        #Fov (active-active)
        Fov_aa  = imds.Fov_aa 
        Fov_aa += lib.einsum("Lia,L->ia",Lov_aa,M)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",Lov_aa,Moo_aa)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",Lov_ia,Moo_ia)

        Joo = [Joo_aa]
        Jvv = [Jvv_aa]
        Jov = [Jov_aa]

        Foo = [Foo_aa]
        Fvv = [Fvv_aa]
        Fov = [Fov_aa]

        return Joo, Jvv, Jov, Foo, Fvv, Fov

    def create_M_intermediates(self, t1, t2):
        #construct the intermediates for the t2 update:
        #M0
        M0 = lib.einsum("Lkc,kc->L", Lov_aa, t1_aa)*2

        #Moo
        Moo = lib.einsum("Lic,jc->Lij", Lov_aa, t1_aa)
        Moo_ia = lib.einsum("Lic,jc->Lij", Lov_ia, t1_aa)

        #Mov
        Mov = lib.einsum("Lac,ic->Lia", Lvv_aa, t1_aa)
        Mov_ai = lib.einsum("Lac,ic->Lia", Lvv_ia, t1_aa)


        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
        Mov_t2 += lib.einsum("Lme, imae -> Lia", Lov_aa, t2_antisym[numpy.ix_(act_hole, act_hole, act_particle, act_particle)])


        return M0, Moo, Mov, Mov_t2


    def add_t2_to_fock(self, Fvv, Foo, Mov_t2):    


        #construct antisymmetrized t2:

        Foo_ai, Foo_aa = Foo
        Fvv_ai, Fvv_aa = Fvv

        #generate the intermediate with full arrays 
        Foo_aa += lib.einsum("Lie,Lje->ij", Lov_aa, Mov_t2)
        Fvv_aa -= lib.einsum("Lma,Lmb->ab",Lov_aa,Mov_t2)

        Foo = [Foo_aa]
        Fvv = [Fvv_aa]

        return Foo, Fvv


    def R1_residue_active(self, t1, t2, Fov, Fvv, Foo):


        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
       
        R1 = Fov_aa

        Foo += lib.einsum("me,ie->im", Fov_aa, t1)

        R1 -= lib.einsum("im, ma -> ia", Foo_aa, t1)       

        Fvv_aa -= lib.einsum("me, ma -> ae", Fov_aa, t1)
 
        R1 += lib.einsum("ae, ie -> ia", Fvv_aa, t1)
        R1 -= lib.einsum("me, imae -> ia", Fov_aa, t2_antisym) #many terms
        R1 += lib.einsum("Lia, L -> ia", Lov_aa, M0)
        R1 -= lib.einsum("Lim, Lma -> ia", Loo_aa, Mov_t1_aa)
        R1 -= lib.einsum("Lim, Lma -> ia", Loo_aa, Mov_t2_aa)
        R1 += lib.einsum("Lae, Lie -> ia", Lvv_aa, Mov_t2_aa)

        return R1


    def R2_residue_active(self, t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo):

        Joo_ai = Joo[0]
        Jvv_ai = Jvv[1]
        Jov_aa = Jov[0]

        #factorized part of the residue:
        R2 = lib.einsum("Lia, Ljb -> ijab", Jov_aa, Jov_aa)
        #PPL 
        Waebf = lib.einsum("Lae, Lbf -> abef", Jvv_aa, Jvv_aa)
#       R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)#three possibilities (ii, ia, ai)
        R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)

        #HHL
        Wijmn = lib.einsum("Lim, Ljn -> ijmn", Joo_aa, Joo_aa)
        R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2) #three possibilities (aa, ai, ia)

        #Fock matrix contribution:

        Foo += lib.einsum("me,ie->im", Fov, t1)
        R2_tmp -= lib.einsum("im, mjab -> ijab", Foo_aa, t2) #only one possibility m has to be inactive.

        Fvv -= lib.einsum("me, ma -> ae", Fov, t1)

      #it is better to have a 
        R2_tmp += lib.einsum("ae, ijeb -> ijab", Fvv_aa, t2) # only one possibility e has to be inactive.

        #N3V3 terms:

        W_jebm = lib.einsum("Ljm, Lbe -> jebm", Joo_aa, Jvv_aa)
#        R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2) # em should be ii, ia, ai types

        R2_tmp -= lib.einsum("jebm, imae -> ijab", Wjebm, t2)


        W_jema = lib.einsum("Ljm, Lae -> jema", Joo_aa, Jvv_aa)
        R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2) # em should be ii, ia, ai types


        #Non DCA terms:
        Ijemb, Ijebm, Imnij = self.t2_transform_quadratic(t2)        

        R2_tmp -= lib.einsum("jebm, imae -> ijab", Ijebm, t2)
        R2_tmp += lib.einsum("jebm, imae -> ijab", Ijebm, t2)

        R2_tmp -= lib.einsum("jema, imeb -> ijab", Ijemb, t2)

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        return R2


    def t2_transform_quadratic(t2):

        # We have already built the DCA terms by assembling the factorized terms.  
        # these terms shall be ignored with the DCA approximation.

        #I_mn^ij 

        Vmenf = lib.einsum("Lme, Lnf -> menf", Lov, Lov)
        Imnij = lib.einsum("menf, ijef -> ijmn", Vmenf, t2) #ef should be ii, ia, ai types

        #I^je_bm
        Vnemf = lib.einsum("Lne, Lmf -> nemf", Lov, Lov)
        Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2) #nf should be ii, ia, ai types

        #I^je_mb
        Ijemb = lib.einsum("nemf, jnfb -> jemb", Vnemf, t2) # #nf should be ii, ia, ai types

        return Ijemb, Ijebm, Imnij
    

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


    def updated_amps(imds, self):
        """
        Following Table XXX in Future Paper
        """

        # NOTE do we want the local copy?
        t1 = self.t1
        t2 = self.t2

        M0, Moo, Mov, Mov_t2 = self.create_M_intermediates(t1, t2)
        Joo, Jvv, Jov, Foo, Fvv, Fov = self.t1_transform(t1, imds)
        Foo, Fvv = self.add_t2_to_fock(t2, M0, Fvv, Foo)

        R1 = self.R1_residue_active(t1, t2, Fov, Fvv, Foo)
        R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo)

        res  = np.linalg.norm(self.t1 + R1 / self._eris.eia)
        res += np.linalg.norm(self.t2 + R2 / self._eris.D)

        t1 = -R1/ self._eris.eia
        t2 = -R2/ self._eris.D
        return res, t1, t2


    def kernel(self, imds, t1full, t2full):

      #we have to extract the active only amplitudes from the full amplitudes:
      t1 = t1full[self.act_hole, :][:, self.act_particle]
      t2 = t2full[self.act_hole, self.act_hole][:, :, self.act_particle, self.act_particle]

      e_corr = None
      adiis = lib.diis.DIIS()
      while err > self.ll_con_tol and count < self.ll_max_its:

          res, t1_new, t2_new = self.updated_amps(imds, t1, t2)
          if self.diis:
              t1, t2 = self.run_diis(t1_new, t2_new, adiis)
          else:
              t1, t2 = t1_new, t2_new

          self.t1 = t1
          self.t2 = t2

          count += 1
          err = res
          # NOTE change this to logger!
          print(f"It {count}; correlation energy {e_corr:.6e}; residual {res:.6e}")

      self._e_corr = e_corr
      self._e_tot = self.mf.e_tot + self._e_corr


      return t1, t2


