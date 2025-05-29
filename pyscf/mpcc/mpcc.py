from pyscf import lib

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel, eri):

        self.mol = mf.mol
        self._scf = mf

        self.eris = eri.ERIs(mf)

        self.lowlevel = lowlevel.MPCC_LL(mf, self.eris)
        self.lowlevel.con_tol = 1e-6
        self.lowlevel.max_its = 50


        # Setting MPCC attributes 
        # Do NOT modify these attributes, they are not input options

        # "Screened_interaction"
        # "High-level"

    def kernel(self):

        self.eris.make_eri()
        
        self.lowlevel.kernel()
