from pyscf import df
from pyscf.mpcc import mpcc, dfrmpcc_lowlevel, df_eri


class RMPCC(mpcc.MPCC):
    def __init__(self, mf, **kwargs):

        mpcc.MPCC.__init__(self, mf, dfrmpcc_lowlevel, df_eri, **kwargs)

    # NOTE need to define a kernel
