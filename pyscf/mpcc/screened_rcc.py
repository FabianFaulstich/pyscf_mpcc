from pyscf import df
from pyscf import lib
import numpy

class screened:
    def __init__(self):
        pass
    def t1_transform(self, t1):
        #fetch the 3-center integrals in MO basis
        Loo, Lov, Lvv = get_df_ints()


# first construct intermediates:
#      


#      JOO (active-active, active-inactive)
        Joo_aa = Loo_aa
        Joo_aa += lib.einsum("Lic,jc->Lij", Lov_ai, t1_ai)
        
        Joo_ai = Loo_ai
        Joo_ai += lib.einsum("Lic,jc->Lij", Lov_ai, t1_ii)
        Joo_ai += lib.einsum("Lic,jc->Lij", Lov_aa, t1_ia)

        Joo_ia = Loo_ia
        Joo_ia += lib.einsum("Lic,jc->Lij", Lov_ii, t1_ai)


#      JVV  (active-active, active-inactive)
        Jvv_aa = Lvv_aa           
        Jvv_aa -= lib.einsum("Lkb,ka->Lab", Lov_ia, t1_ia)

        Jvv_ai = Lvv_ai
        Jvv_ai -= lib.einsum("Lkb,ka->Lab", Lov_ii, t1_ia)


#      JOV (active-active)
        Jov_aa  = Lov_aa
        Jov_aa += lib.einsum("Lac,ic->Lia", Lov_ai, t1_ai)
        Jov_aa -= lib.einsum("Lki,ka->Lia", Joo_ia, t1_ia)  

#       Intermediates

#       M_0^L (active-active, active-inactive, inactive-active)
        M = lib.einsum("Lkc,kc->L", Lov_ii, t1_ii)*2
        M += lib.einsum("Lkc,kc->L", Lov_ia, t1_ia)*2
        M += lib.einsum("Lkc,kc->L", Lov_ai, t1_ai)*2


#       M_ij^L
        Moo = lib.einsum("Lic,jc->Lij", Lov, t1)
#       M_ia^L 
        Mov = lib.einsum("Lac,ic->Lia", Lov, t1)

#Now construct Fock matrix: (active-active, active-inactive)
        Foo_aa  = foo[]
        Foo_aa += lib.einsum("Lij,L->ij", Loo_aa, M)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",Loo_ia,Moo_ia)
        Foo_aa -= lib.einsum("Lmi,Lmj->ij",Loo_aa,Moo_aa)

        Foo_ai  = foo[]
        Foo_ai += lib.einsum("Lij,L->ij", Loo_ai, M)
        Foo_ai -= lib.einsum("Lmi,Lmj->ij",Loo_ia,Moo_ii)
        Foo_ai -= lib.einsum("Lmi,Lmj->ij",Loo_aa,Moo_ai)
        
       #are we missng any term here?
         
        Fvv_aa = fvv[]
        Fvv_aa += lib.einsum("Lab,L->ab",Lvv_aa,M) 
        Fvv_aa += lib.einsum("Lma,Lmb->ab",Lov_ia,Mov_ia)
        Fvv_aa += lib.einsum("Lma,Lmb->ab",Lov_aa,Mov_aa)

        Fvv_ai = fvv[]
        Fvv_ai += lib.einsum("Lab,L->ab",Lvv_ai,M)
        Fvv_ai += lib.einsum("Lma,Lmb->ab",Lov_ia,Mov_ii)
        Fvv_ai += lib.einsum("Lma,Lmb->ab",Lov_aa,Mov_ai)
       
        #Fov (active-active, inactive-inactive)
        Fov_aa = fov[]
        Fov_aa += lib.einsum("Lia,L->ia",Lov_aa,M)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",Lov_ia,Moo_ia)
        Fov_aa -= lib.einsum("Lma,Lmi->ia",Lov_aa,Moo_aa)


        Fov_ii = fov[]
        Fov_ii += lib.einsum("Lia,L->ia",Lov_ii,M)
        Fov_ii -= lib.einsum("Lma,Lmi->ia",Lov_ii,Moo_ii)
        Fov_ii -= lib.einsum("Lma,Lmi->ia",Lov_ai,Moo_ai)

        Joo = [Joo_ai, Joo_aa]
        Jvv = [Jvv_ai, Jvv_aa]
        Jov = [Jov_aa]

        Foo = [Foo_ai, Foo_aa]
        Fvv = [Fvv_ai, Fvv_aa]
        Fov = [Fov_ii, Fov_aa]

        return Joo, Jvv, Jov, Foo, Fvv, Fov

    def create_M_intermediates(self, t1, t2):
        #construct the intermediates for the t2 update:
        #M0
        M0 = lib.einsum("Lkc,kc->L", Lov_ii, t1_ii)*2
        M0 += lib.einsum("Lkc,kc->L", Lov_ia, t1_ia)*2
        M0 += lib.einsum("Lkc,kc->L", Lov_ai, t1_ai)*2

        #Moo
        Moo = lib.einsum("Lic,jc->Lij", Lov, t1)

        #Mov
        Mov = lib.einsum("Lac,ic->Lia", Lov, t1)

        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        Mov_t2 = lib.einsum("Lme, imae -> Lia", Lov, t2_antisym)


        return M0, Moo, Mov, Mov_t2



    def add_t2_to_fock(self, Fvv, Foo, Mov_t2):    


        #construct antisymmetrized t2:

        Foo_ai, Foo_aa = Foo
        Fvv_ai, Fvv_aa = Fvv

        Mov_t2_ii, Mov_t2_ia, Mov_t2_ai, Mov_t2_aa = Mov_t2

        #generate the intermediate with full arrays 
        Foo_aa += lib.einsum("Lie,Lje->ij", Lov_ai, Mov_t2_ai)
        Foo_aa += lib.einsum("Lie,Lje->ij", Lov_aa, Mov_t2_aa)

        Foo_ai += lib.einsum("Lie,Lje->ij", Lov_ai, Mov_t2_ii)
        Foo_ai += lib.einsum("Lie,Lje->ij", Lov_aa, Mov_t2_ia)


        Fvv_aa -= lib.einsum("Lma,Lmb->ab",Lov_ia,Mov_t2_ia)
        Fvv_aa -= lib.einsum("Lma,Lmb->ab",Lov_aa,Mov_t2_aa)

        Fvv_ai -= lib.einsum("Lma,Lmb->ab",Lov_ia,Mov_t2_ii)
        Fvv_ai -= lib.einsum("Lma,Lmb->ab",Lov_aa,Mov_t2_ai)

        Foo = [Foo_ai, Foo_aa]
        Fvv = [Fvv_ai, Fvv_aa]

        return Foo, Fvv


    def R1_residue_active(self, t1, t2, Fov, Fvv, Foo):
       
       R1 = Fov_aa

       Foo += lib.einsum("me,ie->im", Fov_ii, t1)

       R1 -= lib.einsum("im, ma -> ia", Foo_ai, t1)       

       Fvv_ai -= lib.einsum("me, ma -> ae", Fov_ii, t1)
 
       R1 += lib.einsum("ae, ie -> ia", Fvv_ai, t1)
       R1 -= lib.einsum("me, imae -> ia", Fov_ii, t2_antisym)
       R1 += lib.einsum("Lia, L -> ia", Lov_aa, M0)
       R1 -= lib.einsum("Lim, Lma -> ia", Loo_ai, Mov_t1)
       R1 -= lib.einsum("Lim, Lma -> ia", Loo_ai, Mov_t2)
       R1 += lib.einsum("Lae, Lie -> ia", Lvv_ai, Mov_t2)


       return R1


    def R2_residue_active(self, t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo):

        Joo_ai = Joo[0]
        Jvv_ai = Jvv[1]
        Jov_aa = Jov[0]

        #factorized part of the residue:
        R2 = lib.einsum("Lia, Ljb -> ijab", Jov_aa, Jov_aa)
        #PPL 
        Waebf = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_ai)
        R2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)
        #HHL
        Wijmn = lib.einsum("Lim, Ljn -> ijmn", Joo_ai, Joo_ai)
        R2 += lib.einsum("ijmn, mnab -> ijab", Wijmn, t2)

        #Fock matrix contribution:

        Foo += lib.einsum("me,ie->im", Fov, t1)
        R2_tmp -= lib.einsum("im, mjab -> ijab", Foo, t2)

        Fvv -= lib.einsum("me, ma -> ae", Fov, t1)
        R2_tmp += lib.einsum("ae, ijeb -> ijab", Fvv, t2)

        #N3V3 terms:

        W_jebm = lib.einsum("Ljm, Lbe -> jebm", Joo_ai, Jvv_ai)
        R2_tmp -= lib.einsum("jebm, imae -> ijab", W_jebm, t2)

        W_jema = lib.einsum("Ljm, Lae -> jema", Joo_ai, Jvv_ai)
        R2_tmp -= lib.einsum("jema, imeb -> ijab", W_jema, t2)

        #Non DCA terms:
        Ijemb, Ijebm, Imnij = self.t2_transform_quadratic(t2)        

        R2_tmp -= lib.einsum("jebm, imae -> ijab", Ijebm, t2)
        R2_tmp += lib.einsum("jebm, imae -> ijab", Ijebm, t2)

        R2_tmp -= lib.einsum("jema, imeb -> ijab", Ijemb, t2)

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        return R2




  #  def t2_transform_linear():


  #     # reconstruct antisymmetrized t2:

  #     # Aia
  #     M = lib.einsum("Lkc, kicd -> Lid",Jov,t2)  

       # generate two types of M: inactive-active, active-inactive


  #     A = lib.einsum("Lid, Lad -> ia",M,Jvv) #NauxNoNv^2

       # Bia

   #    B = lib.einsum("Lka,Lki -> ia", M, Joo)


       # Cia

    #   C = lib.einsum("kc, ikac -> ia", Fov, t2)

    #   return


    def t2_transform_quadratic():


        # We have already built the DCA terms by assembling the factorized terms.  

        #I_mn^ij 

        Vmenf = lib.einsum("Lme, Lnf -> menf", Lov, Lov)

        Imnij = lib.einsum("menf, ijef -> ijmn", Vmenf, t2)


        #I^je_bm
        Vnemf = lib.einsum("Lne, Lmf -> nemf", Lov, Lov)
        Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2)

        #I^je_mb

        Ijemb = lib.einsum("nemf, jnfb -> jemb", Vnemf, t2)

        return Ijemb, Ijebm, Imnij
    


    def kernel(self, t1, t2):
      class _IMDS: pass
      imds = _IMDS()

      M0, Moo, Mov, Mov_t2 = self.create_M_intermediates(t1, t2)
      Joo, Jvv, Jov, Foo, Fvv, Fov = self.t1_transform(t1)
      Foo, Fvv = self.add_t2_to_fock(t2, M0, Fvv, Foo)
      R1 = self.R1_residue_active(t1, t2, Fov, Fvv, Foo)
      R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo)

      imds.R1 = R1
      imds.R2 = R2
      imds.Joo = Joo[1]  # Joo_aa
      imds.Jvv = Jvv[1]
      imds.Jov = Jov[0]
      imds.Foo = Foo[1]
      imds.Fvv = Fvv[1]
      imds.Fov = Fov[1]
      imds.Moo = Moo
      imds.Mov = Mov
      imds.Mov_t2 = Mov_t2  

      return imds


