from pyscf import df
from pyscf import lib
import numpy
from dataclasses import dataclass

class screened:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
        self.frag = frags
#       self.DCA = kwargs.get('DCA')
#       self.add_DCA = True 
        self.add_DCA = kwargs.get('DCA', True)
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
        
        self.Loo_ii = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, inact_hole)].copy()
        self.Loo_ia = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, act_hole)].copy()
        self.Loo_ai = self._eris.Loo[numpy.ix_(naux_idx, act_hole, inact_hole)].copy()
        self.Loo_aa = self._eris.Loo[numpy.ix_(naux_idx, act_hole, act_hole)].copy()

        self.Lvv_ii = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, inact_particle)].copy()
        self.Lvv_ia = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, act_particle)].copy()
        self.Lvv_ai = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, inact_particle)].copy()
        self.Lvv_aa = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, act_particle)].copy()

        self.Lov_ii = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, inact_particle)].copy()
        self.Lov_ia = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, act_particle)].copy()
        self.Lov_ai = self._eris.Lov[numpy.ix_(naux_idx, act_hole, inact_particle)].copy()
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)].copy()


    def _set_t1_blocks(self, t1):

        t1_ii = t1[numpy.ix_(self.inact_hole, self.inact_particle)]
        t1_ia = t1[numpy.ix_(self.inact_hole, self.act_particle)]
        t1_ai = t1[numpy.ix_(self.act_hole, self.inact_particle)]
        t1_aa = t1[numpy.ix_(self.act_hole, self.act_particle)]

        return t1_ii, t1_ia, t1_ai, t1_aa

    def t1_transform(self, t1, M, Moo, Mov, Mov_t2):
        #fetch the 3-center integrals in MO basis

        t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)

        Lov_ai = self.Lov_ai
        Lov_aa = self.Lov_aa
        Lov_ii = self.Lov_ii
        Loo_aa = self.Loo_aa
        Loo_ai = self.Loo_ai
        Loo_ia = self.Loo_ia

        Moo_ii, Moo_ia, Moo_ai, Moo_aa = Moo
        Mov_ii, Mov_ia, Mov_ai, Mov_aa = Mov

# first construct intermediates:
#      

#      JOO (active-active, active-inactive)
        Joo_aa = Loo_aa.copy()
        Joo_aa += lib.einsum("Lic,jc->Lij", Lov_ai, t1_ai)
        
        Joo_ai = Loo_ai.copy()
        Joo_ai += lib.einsum("Lic,jc->Lij", Lov_ai, t1_ii)
        Joo_ai += lib.einsum("Lic,jc->Lij", Lov_aa, t1_ia)

        Joo_ia = Loo_ia.copy()
        Joo_ia += lib.einsum("Lic,jc->Lij", Lov_ii, t1_ai)

#      JVV  (active-active, active-inactive)
        Jvv_aa = self.Lvv_aa.copy()           
        Jvv_aa -= lib.einsum("Lkb,ka->Lab", self.Lov_ia, t1_ia)

        Jvv_ai = self.Lvv_ai.copy()
        Jvv_ai -= lib.einsum("Lkb,ka->Lab", Lov_ii, t1_ia)

#      JOV (active-active)
        Jov_aa  = Lov_aa.copy()
        Jov_aa += Mov_aa
        Jov_aa += Mov_t2[numpy.ix_(numpy.arange(self.naux), self.act_hole, self.act_particle)]
        Jov_aa -= lib.einsum("Lki,ka->Lia", Joo_ia, t1_ia)  
    
#Now construct Fock matrix: (active-active, active-inactive)

        Foo_aa  = self._eris.foo[numpy.ix_(self.act_hole, self.act_hole)].copy()
        Foo_aa += lib.einsum("Lij,L->ij", Loo_aa, M)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",Loo_ia,Moo_ia)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",Loo_aa,Moo_aa)

        Foo_ai  = self._eris.foo[numpy.ix_(self.act_hole, self.inact_hole)].copy()
        Foo_ai += lib.einsum("Lij,L->ij", Loo_ai, M)
        Foo_ai -= lib.einsum("Lmi,Lmj->ij",Loo_ia,Moo_ii)
        Foo_ai -= lib.einsum("Lmi,Lmj->ij",Loo_aa,Moo_ai)
        
       #are we missng any term here?
         
        Fvv_aa = self._eris.fvv[numpy.ix_(self.act_particle, self.act_particle)].copy()        
        Fvv_aa += lib.einsum("Lab,L->ab",self.Lvv_aa,M) 
        Fvv_aa -= lib.einsum("Lma,Lmb->ab",self.Lov_ia,Mov_ia)
        Fvv_aa -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_aa)

        Fvv_ai = self._eris.fvv[numpy.ix_(self.act_particle, self.inact_particle)].copy()      
        Fvv_ai += lib.einsum("Lab,L->ab",self.Lvv_ai,M)
        Fvv_ai -= lib.einsum("Lma,Lmb->ab",self.Lov_ia,Mov_ii)
        Fvv_ai -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_ai)
       
        #Fov (active-active, inactive-inactive)
        Fov_aa = self._eris.fov[numpy.ix_(self.act_hole, self.act_particle)].copy()
        Fov_aa += lib.einsum("Lia,L->ia",self.Lov_aa,M)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",self.Lov_ia,Moo_ia)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",self.Lov_aa,Moo_aa)


        Fov_ii = self._eris.fov[numpy.ix_(self.inact_hole, self.inact_particle)].copy()
        Fov_ii += lib.einsum("Lia,L->ia",Lov_ii,M)
        Fov_ii -= lib.einsum("Lma,Lmi->ia",Lov_ii,Moo_ii)
        Fov_ii -= lib.einsum("Lma,Lmi->ia",Lov_ai,Moo_ai)

        Fov_ia = self._eris.fov[numpy.ix_(self.inact_hole, self.act_particle)].copy()
        Fov_ia += lib.einsum("Lia,L->ia",self.Lov_ia,M)
        Fov_ia -= lib.einsum("Lma,Lmi->ia",self.Lov_ia,Moo_ii)
        Fov_ia -= lib.einsum("Lma,Lmi->ia",self.Lov_aa,Moo_ai)


        Fov_ai = self._eris.fov[numpy.ix_(self.act_hole, self.inact_particle)].copy()
        Fov_ai += lib.einsum("Lia,L->ia",self.Lov_ai,M)
        Fov_ai -= lib.einsum("Lma,Lmi->ia",self.Lov_ii,Moo_ia)
        Fov_ai -= lib.einsum("Lma,Lmi->ia",self.Lov_ai,Moo_aa)

        Foo_aa_t1 =  lib.einsum("me,ie->im", Fov_ai, t1_ai)*0.5
        Foo_aa_t2 =  lib.einsum("me,ie->im", Fov_ai, t1_ai)

        Foo_aa_t1 += Foo_aa
        Foo_aa_t2 += Foo_aa

        Fvv_aa_t1 = -1.0*lib.einsum("me, ma -> ae", Fov_ia, t1_ia)*0.5  
        Fvv_aa_t2 = -1.0*lib.einsum("me, ma -> ae", Fov_ia, t1_ia)

        Fvv_aa_t1 += Fvv_aa
        Fvv_aa_t2 += Fvv_aa

        Joo = [Joo_ai, Joo_aa]
        Jvv = [Jvv_ai, Jvv_aa]
        Jov = [Jov_aa]

        Foo = [Foo_ai, Foo_aa_t1, Foo_aa_t2]
        Fvv = [Fvv_ai, Fvv_aa_t1, Fvv_aa_t2]
        Fov = [Fov_ii, Fov_aa, Fov_ai, Fov_ia]

        return Joo, Jvv, Jov, Foo, Fvv, Fov

    def create_M_intermediates(self, t1, t2):
        #construct the intermediates for the t2 update:

        t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)

        #M0
        M0 = lib.einsum("Lkc,kc->L", self.Lov_ii, t1_ii)*2
        M0 += lib.einsum("Lkc,kc->L", self.Lov_ia, t1_ia)*2
        M0 += lib.einsum("Lkc,kc->L", self.Lov_ai, t1_ai)*2

        #Moo
        Moo_ii = lib.einsum("Lic,jc->Lij", self.Lov_ii, t1_ii) + lib.einsum("Lic,jc->Lij", self.Lov_ia, t1_ia)
        Moo_ia = lib.einsum("Lic,jc->Lij", self.Lov_ii, t1_ai) #we can construct it from taa block as well 
        Moo_ai = lib.einsum("Lic,jc->Lij", self.Lov_ai, t1_ii) + lib.einsum("Lic,jc->Lij", self.Lov_aa, t1_ia)
        Moo_aa = lib.einsum("Lic,jc->Lij", self.Lov_ai, t1_ai)  

        #Mov
        Mov_ii = lib.einsum("Lac,ic->Lia", self.Lvv_ii, t1_ii) + lib.einsum("Lac,ic->Lia", self.Lvv_ia, t1_ia)
        Mov_ia = lib.einsum("Lac,ic->Lia", self.Lvv_ai, t1_ii) + lib.einsum("Lac,ic->Lia", self.Lvv_aa, t1_ia)
        Mov_ai = lib.einsum("Lac,ic->Lia", self.Lvv_ii, t1_ai)
        Mov_aa = lib.einsum("Lac,ic->Lia", self.Lvv_ai, t1_ai)

        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        nocc = self.nocc
        nvir = self.nvir
        Mov_t2  = lib.einsum("Lme, imae -> Lia", self.Lov_ii, 
                             t2_antisym[numpy.ix_(numpy.arange(nocc), self.inact_hole, numpy.arange(nvir), self.inact_particle)])
        Mov_t2 += lib.einsum(
            "Lme, imae -> Lia",
            self.Lov_ia,
            t2_antisym[
            numpy.ix_(
                numpy.arange(self.nocc),
                self.inact_hole,
                numpy.arange(self.nvir),
                self.act_particle,
            )
            ]
        )
        Mov_t2 += lib.einsum("Lme, imae -> Lia", self.Lov_ai, 
                             t2_antisym[numpy.ix_(numpy.arange(nocc), self.act_hole, numpy.arange(nvir), self.inact_particle)])

        Moo = [Moo_ii, Moo_ia, Moo_ai, Moo_aa]
        Mov = [Mov_ii, Mov_ia, Mov_ai, Mov_aa]

        return M0, Moo, Mov, Mov_t2


    def add_t2_to_fock(self, Fvv, Foo, Mov_t2):    


        #construct antisymmetrized t2:

        Foo_ai, Foo_aa_t1, Foo_aa_t2 = Foo
        Fvv_ai, Fvv_aa_t1, Fvv_aa_t2 = Fvv


        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle


        #generate the intermediate with full arrays 
        Foo_tmp = lib.einsum("Lie,Lje->ij", self.Lov_ai, Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, inact_particle)])
        Foo_tmp += lib.einsum("Lie,Lje->ij", self.Lov_aa, Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, act_particle)])

        Foo_aa_t1 += Foo_tmp
        Foo_aa_t2 += Foo_tmp


        Foo_ai += lib.einsum("Lie,Lje->ij", self.Lov_ai, Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, inact_particle)])
        Foo_ai += lib.einsum("Lie,Lje->ij", self.Lov_aa, Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, act_particle)])


        Fvv_tmp  = -lib.einsum("Lma,Lmb->ab",self.Lov_ia,Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, act_particle)])
        Fvv_tmp -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, act_particle)])


        Fvv_aa_t1 += Fvv_tmp
        Fvv_aa_t2 += Fvv_tmp


        Fvv_ai -= lib.einsum("Lma,Lmb->ab",self.Lov_ia,Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, inact_particle)])
        Fvv_ai -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, inact_particle)])

        Foo = [Foo_ai, Foo_aa_t1, Foo_aa_t2]
        Fvv = [Fvv_ai, Fvv_aa_t1, Fvv_aa_t2]

        return Foo, Fvv


    def R1_residue_active(self, t1, t2, Fov, Fvv, Foo, M0=None, Mov=None, Mov_t2=None):
         # Get index arrays for inactive holes and active particles
         inact_hole = self.inact_hole
         act_hole = self.act_hole
         inact_particle = self.inact_particle
         act_particle = self.act_particle
     

         t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)

         #construct antisymmetrized t2:
         t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

         # You may need to define or pass Fov_aa, Fov_ii, Foo_ai, Fvv_ai, etc.
         # For now, let's assume they are slices of Fov, Fvv, Foo as in your t1_transform
         Fov_aa = Fov[1]
         Fov_ii = Fov[0]
         Fov_ai = Fov[2]
         Fov_ia = Fov[3]         
         Foo_ai = Foo[0]
         Fvv_ai = Fvv[0]
         Mov_ii, Mov_ia, Mov_ai, Mov_aa = Mov

         R1 = Fov_aa.copy()
     
         Foo_ai_tmp = Foo_ai.copy()
         Fvv_ai_tmp = Fvv_ai.copy() 

         Foo_ai_tmp += lib.einsum("me,ie->im", Fov_ii, t1_ai)*0.5
#new
         Foo_ai_tmp += lib.einsum("me,ie->im", Fov_ia, t1_aa)*0.5

     
         R1 -= lib.einsum("im, ma -> ia", Foo_ai_tmp, t1_ia)  
     
         Fvv_ai_tmp -= lib.einsum("me, ma -> ae", Fov_ii, t1_ia)*0.5
#new
         Fvv_ai_tmp -= lib.einsum("me, ma -> ae", Fov_ai, t1_aa)*0.5
     
         R1 += lib.einsum("ae, ie -> ia", Fvv_ai_tmp, t1_ai)
         #R1 -= lib.einsum("me, imae -> ia", Fov_ii, t2_antisym) #many terms
     
         R1 -= lib.einsum("me, imae -> ia", Fov_ii,
                          t2_antisym[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.inact_particle)])
     
         R1 -= lib.einsum("me, imae -> ia", Fov_ai, 
                          t2_antisym[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.inact_particle)])
     
         R1 -= lib.einsum("me, imae -> ia", Fov_ia, 
                          t2_antisym[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])
     
         if M0 is not None:
             R1 += lib.einsum("Lia, L -> ia", self.Lov_aa, M0)
         if Mov is not None:
             R1 -= lib.einsum("Lim, Lma -> ia", self.Loo_ai, Mov_ia)
             R1 -= lib.einsum("Lim, Lma -> ia", self.Loo_aa, Mov_aa) #probably not allowed
         if Mov_t2 is not None:
             R1 -= lib.einsum("Lim, Lma -> ia", self.Loo_ai, Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, act_particle)])
             R1 -= lib.einsum("Lim, Lma -> ia", self.Loo_aa, Mov_t2[numpy.ix_(numpy.arange(self.naux), self.act_hole, self.act_particle)])

         R1 += lib.einsum("Lae, Lie -> ia", self.Lvv_ai, Mov_t2[numpy.ix_(numpy.arange(self.naux), self.act_hole, self.inact_particle)])
         R1 += lib.einsum("Lae, Lie -> ia", self.Lvv_aa, Mov_t2[numpy.ix_(numpy.arange(self.naux), self.act_hole, self.act_particle)])

        # Foo_ai -= lib.einsum("me,ie->im", Fov_ii, t1_ai)*0.5
        # Fvv_ai += lib.einsum("me, ma -> ae", Fov_ii, t1_ia)*0.5

         return R1

    def R2_residue_active(self, t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo):


        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle

        #get active hole dimensions:
        n_act_hole = len(act_hole)
        #get active particle dimensions:
        n_act_particle = len(act_particle)

        Joo_ai = Joo[0]
        Jvv_ai = Jvv[0]
        Jov_aa = Jov[0]
        Jvv_aa = Jvv[1]
        Joo_aa = Joo[1]

        Fov_aa = Fov[1]
        Fov_ii = Fov[0]
        Fov_ai = Fov[2]
        Fov_ia = Fov[3]         
        Foo_ai = Foo[0]
        Fvv_ai = Fvv[0]
        
        t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)

        #Non DCA terms:
        if (self.add_DCA):
           Ijemb, Ijebm, Imnij = self.t2_transform_quadratic_inactive(t2)  


        #factorized part of the residue:
       # R2 = lib.einsum("Lia, Ljb -> ijab", Jov_aa, Jov_aa)
        #PPL 
        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_ai)
#       R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)#three possibilities (ii, ia, ai)
        R2  = lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, inact_particle, inact_particle)])

        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_aa)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)])

        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_aa, Jvv_ai)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])

        Wabef = None
        #HHL
        Wijmn = lib.einsum("Lim, Ljn -> ijmn", Joo_ai, Joo_ai) 
        if (self.add_DCA):
           Wijmn += Imnij[numpy.ix_(numpy.arange(n_act_hole), numpy.arange(n_act_hole), inact_hole, inact_hole)]

#       R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2) #three possibilities (aa, ai, ia)

        R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2[numpy.ix_(inact_hole, inact_hole, act_particle, act_particle)])

        Wijmn = lib.einsum("Lim, Ljn -> ijmn", Joo_aa, Joo_ai) 
        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(numpy.arange(n_act_hole), numpy.arange(n_act_hole), act_hole, inact_hole)]

        R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2[numpy.ix_(act_hole, self.inact_hole, act_particle, act_particle)])

        Wijmn = lib.einsum("Lim, Ljn -> ijmn", Joo_ai, Joo_aa)

        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(numpy.arange(n_act_hole), numpy.arange(n_act_hole), inact_hole, act_hole)]

        R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2[numpy.ix_(inact_hole, act_hole, act_particle, act_particle)])

        Wijmn = None

        #Fock matrix contribution:

        Foo_ai_tmp = Foo_ai.copy()
        Fvv_ai_tmp = Fvv_ai.copy()

        #Foo_aa += lib.einsum("me,ie->im", Fov_ai, t1_ai)

        Foo_ai_tmp += lib.einsum("me,ie->im", Fov_ii, t1_ai)
        Foo_ai_tmp += lib.einsum("me,ie->im", Fov_ia, t1_aa)

        R2_tmp = -lib.einsum("im, mjab -> ijab", Foo_ai_tmp, t2[numpy.ix_(inact_hole, act_hole, act_particle, act_particle)]) #only one possibility m has to be inactive.

        #Fvv -= lib.einsum("me, ma -> ae", Fov, t1)

        #Fvv_aa -= lib.einsum("me, ma -> ae", Fov_ia, t1_ia)

        Fvv_ai_tmp -= lib.einsum("me, ma -> ae", Fov_ii, t1_ia)
        Fvv_ai_tmp -= lib.einsum("me, ma -> ae", Fov_ai, t1_aa)

        R2_tmp += lib.einsum("ae, ijeb -> ijab", Fvv_ai_tmp, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)]) # only one possibility e has to be inactive.

        #N3V3 terms:

        W_jebm = lib.einsum("Ljm, Lbe -> jebm", Joo_ai, Jvv_ai) 

        if (self.add_DCA):
            W_jebm  -= Ijebm[numpy.ix_(numpy.arange(n_act_hole), inact_particle, numpy.arange(n_act_particle), inact_hole)]

#       R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2) # em should be ii, ia, ai types

        R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2[numpy.ix_(act_hole, inact_hole, act_particle, inact_particle)])

        W_jebm = lib.einsum("Ljm, Lbe -> jebm", Joo_aa, Jvv_ai)

        if (self.add_DCA):
            W_jebm -= Ijebm[numpy.ix_(numpy.arange(n_act_hole), inact_particle, numpy.arange(n_act_particle), act_hole)]

        R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])

        W_jebm = lib.einsum("Ljm, Lbe -> jebm", Joo_ai, Jvv_aa)

        if (self.add_DCA):

           W_jebm -= Ijebm[numpy.ix_(numpy.arange(n_act_hole), act_particle, numpy.arange(n_act_particle), inact_hole)]

        R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

        if (self.add_DCA):
            R2_tmp += lib.einsum("jemb, imae -> ijab", Ijemb[numpy.ix_(numpy.arange(n_act_hole), inact_particle, inact_hole, numpy.arange(n_act_particle))], t2[numpy.ix_(act_hole, inact_hole, act_particle, inact_particle)])

            R2_tmp += lib.einsum("jemb, imae -> ijab", Ijemb[numpy.ix_(numpy.arange(n_act_hole), inact_particle, act_hole, numpy.arange(n_act_particle))], t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])

            R2_tmp += lib.einsum("jemb, imae -> ijab", Ijemb[numpy.ix_(numpy.arange(n_act_hole), act_particle, inact_hole, numpy.arange(n_act_particle))], t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

        W_jema = lib.einsum("Ljm, Lae -> jema", Joo_ai, Jvv_ai)

        if (self.add_DCA):
            W_jema -= 0.5*Ijemb[numpy.ix_(numpy.arange(n_act_hole), inact_particle, inact_hole, numpy.arange(n_act_particle))]
#       R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2) # em should be ii, ia, ai types

        R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, inact_hole, inact_particle, act_particle)])

        W_jema = lib.einsum("Ljm, Lae -> jema", Joo_ai, Jvv_aa)      
        if (self.add_DCA):
            W_jema -= 0.5*Ijemb[numpy.ix_(numpy.arange(n_act_hole), act_particle, inact_hole, numpy.arange(n_act_particle))]       

        R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

        W_jema = lib.einsum("Ljm, Lae -> jema", Joo_aa, Jvv_ai)
        if (self.add_DCA):
            W_jema -= 0.5*Ijemb[numpy.ix_(numpy.arange(n_act_hole), inact_particle, act_hole, numpy.arange(n_act_particle))]

        R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)])

        #Non DCA terms:
       # Ijemb, Ijebm, Imnij = self.t2_transform_quadratic(t2)        

       # R2_tmp -= lib.einsum("jebm, imae -> ijab", Ijebm, t2)
       # R2_tmp += lib.einsum("jebm, imae -> ijab", Ijebm, t2)

       # R2_tmp -= lib.einsum("jema, imeb -> ijab", Ijemb, t2)

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        return R2


    def t2_transform_quadratic(self,t2):


       #Extract the active part from the follwing terms:

        # We have already built the DCA terms by assembling the factorized terms.  
        # these terms shall be ignored with the DCA approximation.

        #I^mn_ij

        Vmenf = lib.einsum("Lme, Lnf -> menf", self.Lov_ai, self.Lov_aa)
        Imnij_active = lib.einsum("menf, ijef -> ijmn", Vmenf, t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle)])

        Vmenf = lib.einsum("Lme, Lnf -> menf", self.Lov_aa, self.Lov_ai)
        Imnij_active +=lib.einsum("menf, ijef -> ijmn", Vmenf, t2[numpy.ix_(self.act_hole, self.act_hole,self.act_particle,self.inact_particle)])

        Vmenf = lib.einsum("Lme, Lnf -> menf", self.Lov_ai, self.Lov_ai)
        Imnij_active += lib.einsum("menf, ijef -> ijmn",Vmenf, t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.inact_particle)])

        #I^je_bm
        #Vnemf = lib.einsum("Lne, Lmf -> nemf", self.Lov, self.Lov)
        #Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2) #nf should be ii, ia, ai types


        Vnemf = lib.einsum("Lne, Lmf -> nemf", self.Lov_ia, self.Lov_ai)

        Ijebm_active = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.inact_particle)])
        Ijemb_active = lib.einsum("nemf, jnfb -> jemb", Vnemf, t2[numpy.ix_(self.act_hole, self.inact_hole, self.inact_particle, self.act_particle)])


        Vnemf = lib.einsum("Lne, Lmf -> nemf", self.Lov_ia, self.Lov_aa)
        Ijebm_active += lib.einsum("nemf, jnbf -> jebm", Vnemf, t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])
        Ijemb_active += lib.einsum("nemf, jnfb -> jemb", Vnemf, t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])

        
        Vnemf = lib.einsum("Lne, Lmf -> nemf", self.Lov_aa, self.Lov_ai)
        Ijebm_active += lib.einsum("nemf, jnbf -> jebm", Vnemf, t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.inact_particle)])
        Ijemb_active += lib.einsum("nemf, jnfb -> jemb", Vnemf, t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle)])
        

        return Ijemb_active, Ijebm_active, Imnij_active


    def t2_transform_quadratic_inactive(self,t2):


       #Extract the active part from the follwing terms:

        # We have already built the DCA terms by assembling the factorized terms.  
        # these terms shall be ignored with the DCA approximation.

        #I_mn^ij 

        Vmnef = lib.einsum("Lme, Lnf -> mnef", self._eris.Lov, self._eris.Lov)

        #Imnij = lib.einsum("menf, ijef -> ijmn", Vmenf, t2) #ef should be ii, ia, ai types
        
        Imnij = lib.einsum("mnef, ijef -> ijmn", Vmnef[numpy.ix_(numpy.arange(self.nocc), numpy.arange(self.nocc),self.inact_particle,self.act_particle)],
                            t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle)])

        Imnij += lib.einsum("mnef, ijef -> ijmn", Vmnef[numpy.ix_(numpy.arange(self.nocc), numpy.arange(self.nocc),self.act_particle,self.inact_particle)],
                             t2[numpy.ix_(self.act_hole, self.act_hole,self.act_particle,self.inact_particle)])
        
        Imnij += lib.einsum("mnef, ijef -> ijmn", Vmnef[numpy.ix_(numpy.arange(self.nocc), numpy.arange(self.nocc),self.inact_particle,self.inact_particle)],
                            t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.inact_particle)])


        #I^je_bm
        Vnemf = lib.einsum("Lne, Lmf -> nemf", self._eris.Lov, self._eris.Lov)
        #Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2) #nf should be ii, ia, ai types
                                                             #em should be nvir and nocc
                                                             #jb should be active hole, particle       


        Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf[numpy.ix_(self.inact_hole,numpy.arange(self.nvir),numpy.arange(self.nocc), self.inact_particle)],
                            t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.inact_particle)])


        Ijebm += lib.einsum("nemf, jnbf -> jebm", Vnemf[numpy.ix_(self.inact_hole,numpy.arange(self.nvir), numpy.arange(self.nocc),self.act_particle)],
                            t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])


        Ijebm += lib.einsum("nemf, jnbf -> jebm", Vnemf[numpy.ix_(self.act_hole, numpy.arange(self.nvir), numpy.arange(self.nocc), self.inact_particle)],
                             t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.inact_particle)])


        #I^je_mb

        #Ijemb = lib.einsum("nemf, jnfb -> jemb", Vnemf, t2) # #nf should be ii, ia, ai types

        Ijemb = lib.einsum("nemf, jnfb -> jemb", Vnemf[numpy.ix_(self.inact_hole,numpy.arange(self.nvir),numpy.arange(self.nocc), self.inact_particle)],
                                                                  t2[numpy.ix_(self.act_hole, self.inact_hole, self.inact_particle, self.act_particle)])

        Ijemb += lib.einsum("nemf, jnfb -> jemb", Vnemf[numpy.ix_(self.inact_hole,numpy.arange(self.nvir), numpy.arange(self.nocc),self.act_particle)],
                            t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])

        Ijemb += lib.einsum("nemf, jnfb -> jemb", Vnemf[numpy.ix_(self.act_hole, numpy.arange(self.nvir), numpy.arange(self.nocc), self.inact_particle)],
                             t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle)])


        return Ijemb, Ijebm, Imnij


    @dataclass
    class _IMDS:
       R1: numpy.ndarray
       R2: numpy.ndarray
       Joo: numpy.ndarray
       Jvv: numpy.ndarray
       Jov: numpy.ndarray
       Foo_t1: numpy.ndarray
       Foo_t2: numpy.ndarray
       Fvv_t1: numpy.ndarray
       Fvv_t2: numpy.ndarray
       Fov: numpy.ndarray
       Ijemb_active: numpy.ndarray
       Ijebm_active: numpy.ndarray
       Imnij_active: numpy.ndarray

    def kernel(self, t1, t2):
      
       M0, Moo, Mov, Mov_t2 = self.create_M_intermediates(t1, t2)
       Joo, Jvv, Jov, Foo, Fvv, Fov = self.t1_transform(t1, M0, Moo, Mov, Mov_t2)
       Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mov_t2)
       R1 = self.R1_residue_active(t1, t2, Fov, Fvv, Foo, M0, Mov, Mov_t2)
       Ijemb_active, Ijebm_active, Imnij_active = self.t2_transform_quadratic(t2)

       R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo)
       imds = self._IMDS(R1 = R1, R2 = R2,
                        Joo = Joo[1], Jvv = Jvv[1], Jov = Jov[0],
                        Foo_t1 = Foo[1], Foo_t2 = Foo[2], Fvv_t1 = Fvv[1], Fvv_t2 = Fvv[2],Fov = Fov[1],
                        Ijemb_active = Ijemb_active, Ijebm_active = Ijebm_active, Imnij_active = Imnij_active)       

       return imds
