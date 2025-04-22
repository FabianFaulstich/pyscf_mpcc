import numpy as np

class MPCCSD():
    
    def __init__(self,t1,t2):

        self.act_hole 
        self.act_particle 
        self.nocc 
        self.nvir

        self.inact_particle = np.delete(np.arange(nvir), act_particle)
        self.inact_hole = np.delete(np.arange(nocc), act_hole)

        singles = []
        singles.append(np.ix_(act_hole, inact_particle))
        singles.append(np.ix_(inact_hole, act_particle))
        singles.append(np.ix_(inact_hole, inact_particle))
        singles.append(np.ix_(act_hole, act_particle))
        
        doubles = []
        doubles.append(np.ix_(act_hole, act_hole, inact_particle, act_particle))
        doubles.append(np.ix_(inact_hole, act_hole, inact_particle, act_particle))
        doubles.append(np.ix_(act_hole, inact_hole, inact_particle, act_particle))
        doubles.append(np.ix_(inact_hole, inact_hole, inact_particle, act_particle))
        
        doubles.append(np.ix_(act_hole, act_hole, act_particle, inact_particle))
        doubles.append(np.ix_(inact_hole, act_hole, act_particle, inact_particle))
        doubles.append(np.ix_(act_hole, inact_hole, act_particle, inact_particle))
        doubles.append(np.ix_(inact_hole, inact_hole, act_particle, inact_particle))
        
        doubles.append(np.ix_(act_hole, act_hole, inact_particle, inact_particle))
        doubles.append(np.ix_(inact_hole, act_hole, inact_particle, inact_particle))
        doubles.append(np.ix_(act_hole, inact_hole, inact_particle, inact_particle))
        doubles.append(np.ix_(inact_hole, inact_hole, inact_particle, inact_particle))
     
        doubles.append(np.ix_(inact_hole, act_hole, act_particle, act_particle))
        doubles.append(np.ix_(act_hole, inact_hole, act_particle, act_particle))
        doubles.append(np.ix_(inact_hole, inact_hole, act_particle, act_particle))
        doubles.append(np.ix_(act_hole, act_hole, act_particle, act_particle))


        def get_t_mix():
            singles_iA = np.ix_(self.inact_hole, self.act_particle) # i, A
            singles_Ia = np.ix_(self.act_hole, self.inact_particle) # I, a
            
            t1_mix_a = [t1[0][single_iA], t1[0][single_Ia]]
            t1_mix_b = [t1[1][single_iA], t1[1][single_Ia]]
            t1_mix = [t1_mix_a, t1_mix_b]
            

            doubles_IJaB = np.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle) # IJ, aB
            # doubles.append(np.ix_(act_hole, act_hole, act_particle, inact_particle)) # IJ, Ab = - IJ, aB
            
            doubles_iJaB = np.ix_(self.inact_hole, self.act_hole, self.inact_particle, self.act_particle) # iJ, aB
            # doubles.append(np.ix_(inact_hole, act_hole, act_particle, inact_particle)) # iJ, Ab = - iJ, aB  
            # doubles.append(np.ix_(act_hole, inact_hole, inact_particle, act_particle)) # Ij, aB = - iJ, aB
            # doubles.append(np.ix_(act_hole, inact_hole, act_particle, inact_particle)) # Ij, Ab = iJ, aB 

            doubles_ijaB = np.ix_(self.inact_hole, self.inact_hole, self.inact_particle, self.act_particle) # ij, aB
            # doubles.append(np.ix_(inact_hole, inact_hole, act_particle, inact_particle)) # ij, Ab = - ij, aB
             
            doubles_iJab = np.ix_(self.inact_hole, self.act_hole, self.inact_particle, self.inact_particle) # iJ, ab
            # doubles.append(np.ix_(act_hole, inact_hole, inact_particle, inact_particle)) # Ij, ab = - iJ, ab 
         
            doubles_iJAB = np.ix_(self.inact_hole, self.act_hole, self.act_particle, self.act_particle) # iJ, AB 
            # doubles.append(np.ix_(act_hole, inact_hole, act_particle, act_particle)) # Ij, AB = - iJ, AB
             
            doubles_IJab = np.ix_(self.act_hole, self.act_hole, self.inact_particle, self.inact_particle) # IJ, ab
            doubles_ijAB = np.ix_(self.inact_hole, self.inact_hole, self.act_particle, self.act_particle) # ij, AB

            t2_mix_aa = [t2[0][doubles_IJab], t2[0][doubles_ijAB], t2[0][doubles_iJaB], t2[0][doubles_iJAB], t2[0][doubles_iJab], t2[0][doubles_ijaB], t2[0][doubles_IJaB]]
            t2_mix_ab = [t2[1][doubles_IJab], t2[1][doubles_ijAB], t2[1][doubles_iJaB], t2[1][doubles_iJAB], t2[1][doubles_iJab], t2[1][doubles_ijaB], t2[1][doubles_IJaB]]
            t2_mix_bb = [t2[2][doubles_IJab], t2[2][doubles_ijAB], t2[2][doubles_iJaB], t2[2][doubles_iJAB], t2[2][doubles_iJab], t2[2][doubles_ijaB], t2[2][doubles_IJaB]]
        
            t2_mix = [t2_mix_aa, t2_mix_ab, t2_mix_bb]

            return t1_mix, t2_mix
    
        def get_t_frag():

            singles = np.ix_(self.act_hole, self.act_particle) 
            doubles = np.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)
          
            t1_frag_a[singles] = t1[0][singles]
            t1_frag_b[singles] = t1[1][singles]
            t1_frag = [t1_frag_a, t1_frag_b]

            t2_frag_aa[doubles] = t2[0][doubles]
            t2_frag_ab[doubles] = t2[1][doubles]
            t2_frag_bb[doubles] = t2[2][doubles]
            t2_frag = [t2_frag_aa, t2_frag_ab, t2_frag_bb]

            return t1_frag, t2_frag

        def get_t_env():

            singles = np.ix_(self.inact_hole, self.inact_particle) 
            doubles = np.ix_(self.inact_hole, self.inact_hole, self.inact_particle, self.inact_particle)
          
            t1_env_a[singles] = t1[0][singles]
            t1_env_b[singles] = t1[1][singles]
            t1_env = [t1_env_a, t1_env_b]

            t2_env_aa[doubles] = t2[0][doubles]
            t2_env_ab[doubles] = t2[1][doubles]
            t2_env_bb[doubles] = t2[2][doubles]
            t2_env = [t2_env_aa, t2_env_ab, t2_env_bb]

            return t1_env, t2_env


                

