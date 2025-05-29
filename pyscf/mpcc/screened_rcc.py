from pyscf import df
from pyscf import lib
import numpy

class screened:
    def __init__(self):
        pass
    def t1_transform(self, t1):
        #fetch the 3-center integrals in MO basis
        Loo, Lov, Lvv = get_df_ints()       

#      JOO
        Joo = Loo
        Joo += lib.einsum("Lic,jc->Lij", Lov, t1) 
        
#      JVV             
        Jvv  = Lvv
        Jvv -= lib.einsum("Lkb,ka->Lab", Lov, t1) 

#      JOV 
        Jov  = Lov
        Jov -= lib.einsum("Lkb,ka->Lab", Lov, t1) 

    
