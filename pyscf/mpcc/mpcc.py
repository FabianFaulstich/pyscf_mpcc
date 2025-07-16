from pyscf import lib
import numpy

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel, screened, highlevel, eri, mo_coeff=None, **kwargs):

        self.mol = mf.mol
        self._scf = mf

        self.mo_coeff = mo_coeff if mo_coeff is not None else mf.mo_coeff

        self.eris = eri.ERIs(mf, self.mo_coeff)
        self.frags = kwargs.get('frag')
        if self.frags is None:
            raise ValueError("Missing required keyword argument 'frag' in kwargs.")

        print('MPCC fragments:', self.frags)

        self.lowlevel = lowlevel.MPCC_LL(mf, self.eris, self.frags, **kwargs)
        self.screened = screened.screened(mf, self.eris, self.frags[0], **kwargs)
        self.highlevel = highlevel.MPCC_HL(mf, self.eris, self.frags[0], **kwargs)
       
        # Setting MPCC attributes 
        # use "_" for variable protection 

    def kernel(self, **kwargs):

        #if localization:
        #    try:
        #        c_lo = kwargs['c_lo']
        #    except:
        #        print('Localization orbital transformation and fragments not provided. \nDefaulting to AVAS!')
        #        breakpoint() 

#       self.eris.make_eri()

        count = 0
        e_mpcc_prev = -numpy.inf
        e_diff = numpy.inf
        tol = kwargs.get('tol', 1e-6)
        count_tol = kwargs.get('count_tol', 100)

        t1, t2 = self.lowlevel.init_amps()

        #start an iteration loop here:
        while e_diff > tol and count < count_tol:
            count += 1

            #get the active fragments
            #frags = self.lowlevel.get_active_fragments()
            #frags = [frag]  #for now we will use only one fragment, but later we can use multiple fragments
            #if no fragments are found, break the loop
            #if not frags:
            #    print('No active fragments found. Exiting loop.')
            #    break


            t1, t2 = self.lowlevel.kernel(t1, t2) #should take infos for multiple fragments, and keep the subsequent active amplitudes unaltered..
            #the following loop is parallelizable over fragments
            for frag in self.frags: 
               #modified the self.screened object
               self.screened.frag = frag
               self.highlevel.frag = frag

               imds = self.screened.kernel(t1, t2)
               #print the attributes of the imds object
               print('MPCC: Screened kernel calculated for fragment:', frag)
               
               print("JOO:", imds.Joo.shape)

               t1_act, t2_act = self.highlevel.kernel(imds, t1, t2)
               print('MPCC: High-level kernel calculated for fragment:')
        #NOTE: when we will use T3 amplitudes, we can directly return it here. we don't need to reuse them for any other purposes. Therefore
        #we can update them using diis only in the high level solver
        #store active amplitudes in a container or may be in a hdf5 file for later use:

        #modify a section of the array with new elements:
               act_hole = frag[0]
               act_particle = frag[1]

               t1[numpy.ix_(act_hole, act_particle)] = t1_act[:,:]
               t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act
            #calculate the energy:
               e_mpcc = self.lowlevel.energy(t1, t2)
               e_diff = abs(e_mpcc - e_mpcc_prev)
               e_mpcc_prev = e_mpcc
               print(f"It {count}; Energy {e_mpcc:.6e}; Energy difference {e_diff:.6e}")
        #Now use DIIS extrapolation to update t1 and t2 amplitudes all together. But I believe this step may be skipped.
        #also at this step we can evaluate total energy to test the convergence of the whole procedure..  
         


        


