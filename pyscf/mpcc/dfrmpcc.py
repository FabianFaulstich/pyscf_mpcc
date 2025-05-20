from pyscf import df
from pyscf.mpcc import mpcc, dfrmpcc_lowlevel

class RMPCC(mpcc.MPCC):
    def __init__(self, mf):
        mpcc.MPCC.__init__(self, mf, dfrmpcc_lowlevel)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

