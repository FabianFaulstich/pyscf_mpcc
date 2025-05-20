from pyscf import scf

def MPCC(mf):
    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError('UMPCC is not implemented')
    elif isinstance(mf, scf.ghf.GHF):
        raise RuntimeError('GMPCC is not implemented')
    else:
        return RMPCC(mf)

def RMPCC(mf):
    from pyscf.mpcc import dfrmpcc

    if getattr(mf, 'with_df', None):
        return dfrmpcc.RMPCC(mf)
    else:
        raise RuntimeError('Chosen RMPCC variant is not implemented')


