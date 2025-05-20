from pyscf import lib

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel):

        self.mol = mf.mol
        self._scf = mf

        self.lowlevel = lowlevel.MPCC_LL(mf)

        # Setting MPCC attributes 
        # Do NOT modify these attributes, they are not input options
