from pyscf import df
from pyscf.mpcc import mpcc, dfrmpcc_lowlevel, screened_rcc, dfrmpcc_highlevel, df_eri


class RMPCC(mpcc.MPCC):
    def __init__(self, mf, mo_coeff=None, **kwargs):

        mpcc.MPCC.__init__(self, mf, dfrmpcc_lowlevel, screened_rcc, dfrmpcc_highlevel, df_eri, mo_coeff, **kwargs)

    # NOTE need to define a kernel
