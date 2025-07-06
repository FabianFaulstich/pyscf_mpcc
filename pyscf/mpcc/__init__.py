from pyscf import scf


def MPCC(mf, mo_coeff, **kwargs):
    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError("UMPCC is not implemented")
    elif isinstance(mf, scf.ghf.GHF):
        raise RuntimeError("GMPCC is not implemented")
    else:
        # NOTE only DF implemented! Change this later
        return RMPCC(mf, with_df= True, **kwargs)


def RMPCC(mf, with_df, mo_coeff, **kwargs):
    from pyscf.mpcc import dfrmpcc
    
    if with_df:
        return dfrmpcc.RMPCC(mf, mo_coeff, **kwargs)
    else:
        raise RuntimeError("RMPCC variant is not implemented")
