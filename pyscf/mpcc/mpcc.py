from pyscf import lib
import numpy as np

class MPCC(lib.StreamObject):

    def __init__(self, mf, lowlevel, screened, highlevel, eri, **kwargs):

        self.mol = mf.mol
        self._scf = mf

        if 'lo_coeff' in kwargs:
            self.lo_coeff = kwargs['lo_coeff'] 
        else:
            # FIXME Have a default localization here 
            raise ValueError(f'No local orbitals provided!')


        self.eris = eri.ERIs(mf, self.lo_coeff)
        self.frags = kwargs.get('frag')
        if self.frags is None:
            raise ValueError("Missing required keyword argument 'frag' in kwargs.")

        print('MPCC fragments:', self.frags)

        self.lowlevel = lowlevel.MPCC_LL(mf, self.eris, self.frags, **kwargs)
        self.screened = screened.screened(mf, self.eris, self.frags[0], **kwargs)
        self.highlevel = highlevel.MPCC_HL(mf, self.eris, self.frags[0], **kwargs)
       

    def kernel(self, verbose = None, **kwargs):

        log = lib.logger.new_logger(self, verbose)

        count = 0
        e_mpcc_prev = -np.inf
        e_diff = np.inf
        tol = kwargs.get('tol', 1e-6)
        count_tol = kwargs.get('count_tol', 100)

        t1, t2 = self.lowlevel.init_amps()

        #start an iteration loop here:
        while e_diff > tol and count < count_tol:
            count += 1
            
            print(f'MPCC macro iteration: {count}')

            if (count > 1):
                print(f'Starting low-level MPCC iteration. Low-level kernel type {self.lowlevel.kernel_type}')
                t1, t2 = self.lowlevel.kernel(t1, t2_act, **kwargs) 

            t1_act = []
            t2_act = []

            #the following loop is parallelizable over fragments
            for frag in self.frags: 
               #modified the self.screened object
               self.screened.frag = frag
               self.highlevel.frag = frag

               # NOTE can we remove the t2 dependence? 
               imds = self.screened.kernel(t1, t2)
               #print the attributes of the imds object
               print('MPCC: Screened kernel calculated for fragment:', frag)

               # NOTE can we remove the t2 dependence? 
               # YES, remove t2!
               t1_act_tmp, t2_act_tmp = self.highlevel.kernel(imds, t1, t2)

               t1_act.append(t1_act_tmp)    
               t2_act.append(t2_act_tmp) 

               print('MPCC: High-level kernel calculated for fragment:')
        #NOTE: when we will use T3 amplitudes, we can directly return it here. we don't need to reuse them for any other purposes. Therefore
        #we can update them using diis only in the high level solver
        #store active amplitudes in a container or may be in a hdf5 file for later use:
            frag_i = 0 
            for frag in self.frags: 
               #modify a section of the array with new elements:
               act_hole = frag[0]
               act_particle = frag[1]
                
               t1[np.ix_(act_hole, act_particle)] = t1_act[frag_i]
               t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[frag_i]

               frag_i += 1

            #calculate the energy:
            e_mpcc = self.lowlevel.get_energy(t1, t2)
            e_diff = abs(e_mpcc - e_mpcc_prev)
            e_mpcc_prev = e_mpcc
            print(f"It {count}; Energy {e_mpcc:.6e}; Energy difference {e_diff:.6e}")

        return e_mpcc+self._scf.e_tot
