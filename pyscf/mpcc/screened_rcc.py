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


#      JOO
        Joo = Loo
        Joo += lib.einsum("Lic,jc->Lij", Lov, t1)

#      JVV             
        Jvv  = Lvv
        Jvv -= lib.einsum("Lkb,ka->Lab", Lov, t1)

#      JOV 
        Jov  = Lov
        Jov += lib.einsum("Lac,ic->Lia", Lov, t1)
        Jov -= lib.einsum("Lki,ka->Lia", Joo, t1)  

#       Intermediates

#       M_0^L
        M = lib.einsum("Lkc,tkc->L", Lov, t1)*2

#       M_ij^L
        Moo = lib.einsum("Lic,jc->Lij", Lov, t1)
#       M_ia^L 
        Mov = lib.einsum("Lac,ic->Lia", Lov, t1)

#Now construct Fock matrix:
        Foo  = foo
        Foo += lib.einsum("Lij,L->ij", Loo, M)
        Foo -= lib.einsum("Lmi,Lmj->ij",Loo,Moo)
       #are we missng any term here?
         
        Fvv = fvv
        Fvv += lib.einsum("Lab,L->ab",Lvv,M) 
        Fvv += lib.einsum("Lma,Lmb->ab",Lov,Mov)
       

        Fov  = fov
        Fov += lib.einsum("Lia,L->ia",Lov,M)
        Fov += lib.einsum("Lma,Lmi->ia",Lov,Moo)


    def t2_transform_linear():


    def t2_transform_quadratic():









