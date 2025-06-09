from pyscf import lib

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel, eri, **kwargs):

        self.mol = mf.mol
        self._scf = mf

        self.eris = eri.ERIs(mf)

        self.lowlevel = lowlevel.MPCC_LL(mf, self.eris)
        
        if 'll_max_its' in kwargs:
            self.lowlevel.ll_max_its = kwargs['ll_max_its']
        else:    
            self.lowlevel.ll_max_its = 50

        if 'll_con_tol' in kwargs:
            self.lowlevel.ll_con_tol = kwargs['ll_con_tol']
        else:
            self.lowlevel.ll_con_tol = 1e-6
        
        # Setting MPCC attributes 
        # use "_" for variable protection 

        # "Screened_interaction"
        # "High-level"

    def kernel(self):

        self.eris.make_eri()
        
        self.lowlevel.kernel()
