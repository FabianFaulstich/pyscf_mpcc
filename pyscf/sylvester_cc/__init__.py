from pyscf import scf

def DFRCCSD(mf, **kwargs):
    from pyscf.sylvester_cc import df_eri, dfrccsd

    if mf.istype('UHF'):
        raise RuntimeError('DFRCCSD cannot be used with UHF method.')

    eris = df_eri.ERIs(mf)
    return dfrccsd.DFRCCSD(mf, eris, **kwargs)


def DFRPA(mf, **kwargs):
    from pyscf.sylvester_cc import df_eri, dfrpa

    if mf.istype('UHF'):
        raise RuntimeError('DFRPA cannot be used with UHF method.')

    eris = df_eri.ERIs(mf)
    return dfrpa.DFRPA(mf, eris, **kwargs)


def DFRPASylvester(mf, **kwargs):
    from pyscf.sylvester_cc import df_eri, dfrpa_sylvester

    if mf.istype('UHF'):
        raise RuntimeError('DFRPASylvester cannot be used with UHF method.')

    eris = df_eri.ERIs(mf)
    return dfrpa_sylvester.DFRPASylvester(mf, eris, **kwargs)
