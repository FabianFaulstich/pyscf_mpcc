from pyscf import lib

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel, eri, **kwargs):

        self.mol = mf.mol
        self._scf = mf

        self.eris = eri.ERIs(mf)

        self.lowlevel = lowlevel.MPCC_LL(mf, self.eris, **kwargs)
       
        # Setting MPCC attributes 
        # use "_" for variable protection 

        # "Screened_interaction"
        # "High-level"

    def kernel(self, localization = False, **kwargs):

        if localization:
            try:
                c_lo = kwargs['c_lo']
                frag = kwargs['frag']
            except:
                print('Localization orbital transformation and fragments not provided. \nDefaulting to AVAS!')
                breakpoint() 

        self.eris.make_eri()
        
        self.lowlevel.kernel()
