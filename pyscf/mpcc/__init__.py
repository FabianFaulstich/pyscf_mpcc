from pyscf import scf


def MPCC(mf, **kwargs):
    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError("UMPCC is not implemented")
    elif isinstance(mf, scf.ghf.GHF):
        raise RuntimeError("GMPCC is not implemented")
    else:
        # NOTE only DF implemented! Change this later
        return RMPCC(mf, with_df= True, **kwargs)


def RMPCC(mf, with_df, **kwargs):
    from pyscf.mpcc import dfrmpcc
   
    if mf.istype('UHF'):
        raise RuntimeError('RMPCC cannot be used with UHF method.')
 
    # NOTE we are enforcing the meanfield object to be on df type
    # Is this necessary?
    if mf.istype('DFRHF'):
        return dfrmpcc.RMPCC(mf, **kwargs)
    else:
        raise RuntimeError("RMPCC variant is not implemented")
