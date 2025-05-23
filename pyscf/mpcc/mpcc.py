from pyscf import lib

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel):

        self.mol = mf.mol
        self._scf = mf

        # Hold the integrals here? -> ChemistryERI


        self.lowlevel = lowlevel.MPCC_LL(mf)
        self.lowlevel.con_tol = 1e-6
        self.lowlevel.max_its = 50

        self.lowlevel.kernel()

        # Setting MPCC attributes 
        # Do NOT modify these attributes, they are not input options

        # "Screened_interaction"
        # "High-level"

