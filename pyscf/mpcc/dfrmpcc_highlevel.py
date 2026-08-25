from pyscf import df
from pyscf import lib
import numpy

class MPCC_HL:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
    
        self.diis = True

        self.ll_con_tol = kwargs.get("ll_con_tol")
        self.ll_max_its = kwargs.get("ll_max_its")

        self.set_fragment(frags)
    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc
        
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    @property
    def naux(self):
        return self.with_df.get_naoaux()

    @property
    def act_hole(self):
        return self.frag[0]

    @property
    def nact_particle(self):
        return len(self.act_particle)
    
    @property
    def nact_hole(self):
        return len(self.act_hole)

    @property
    def act_particle(self):
        return self.frag[1]
        
    @property
    def inact_hole(self):
        return numpy.setdiff1d(numpy.arange(self.nocc), self.act_hole)
        
    @property
    def inact_particle(self):
        return numpy.setdiff1d(numpy.arange(self.nvir), self.act_particle)

    def set_fragment(self, frag):
        """Update the fragment and refresh fragment-dependent integral blocks."""
        self.frag = frag
        self._set_integral_blocks()

        # at this point we will classify integralsi: 

    def _set_integral_blocks(self):

        act_hole = self.act_hole
        act_particle = self.act_particle
        naux_idx = numpy.arange(self.naux)
        
        self.Loo_aa = self._eris.Loo[numpy.ix_(naux_idx, act_hole, act_hole)]
        self.Lvv_aa = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, act_particle)]
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)]


    def clear_integral_blocks(self):
        """Free fragment-specific integral slices after the kernel completes."""
        self.Loo_aa = None
        self.Lvv_aa = None
        self.Lov_aa = None

    def t1_transform(self, imds, t1, M, Moo, Mvo, Mvo_t2):
        #fetch the 3-center integrals in MO basis


        #in this function we will build only those terms where t1_aa contributes. also these terms will be iteratively updated.
        #initialization could be done outside the iterative loop as well. 

# first construct intermediates:
#      
        Moo_aa = Moo[0]
        Mvo_aa = Mvo[0]
#       Mvo_ai = Mvo[1]
#      JOO (active-active)
        Joo_aa = numpy.array(imds.Joo).copy()
        Joo_aa += Moo_aa

#      JVV  (active-active, active-inactive)
        Jvv_aa = numpy.array(imds.Jvv).copy()      
        Jvv_aa -= lib.einsum("Lkb,ka->Lab", self.Lov_aa, t1)

#      JOV (active-active)
        Jvo_aa  = numpy.array(imds.Jvo).copy()

#       Jvo_aa += lib.einsum("Lac,ic->Lia", self.Lvv_aa, t1)
        Jvo_aa += Mvo_aa
        Jvo_aa += Mvo_t2
        Jvo_aa -= lib.einsum("Lji,ja->Lai", Joo_aa, t1)  

#Now construct Fock matrix: (active-active, active-inactive)
        Foo_aa  = lib.einsum("Lij,L->ij", self.Loo_aa, M)
        Foo_aa -= lib.einsum("Lmj,Lim->ij",self.Loo_aa,Moo_aa)
     #  Foo_aa -= lib.einsum("Lmj,Lim->ij",self.Loo_ia,Moo_ia)###perhaps not required?
        Foo_aa_t1  = numpy.array(imds.Foo_t1).copy()
       
        Foo_aa_t1 += Foo_aa

        Foo_aa_t2  = numpy.array(imds.Foo_t2).copy()
        
        Foo_aa_t2 += Foo_aa
       #are we missng any term here?
         
#       Fvv_aa  = numpy.array(imds.Fvv).copy()
        Fvv_aa = lib.einsum("Lab,L->ab",self.Lvv_aa,M) 
        Fvv_aa -= lib.einsum("Lmb,Lam->ab",self.Lov_aa,Mvo_aa)


        Fvv_aa_t1  = numpy.array(imds.Fvv_t1).copy()
        
        Fvv_aa_t1 += Fvv_aa

        Fvv_aa_t2  = numpy.array(imds.Fvv_t2).copy()
        
        Fvv_aa_t2 += Fvv_aa

        #Fov (active-active)
        Fov_aa  = numpy.array(imds.Fov).copy()
        
        Fov_aa += lib.einsum("Lia,L->ia",self.Lov_aa,M)
        Fov_aa -= lib.einsum("Lib,Lji->jb",self.Lov_aa,Moo_aa)
#       Fov_aa -= lib.einsum("Lib,Lji->jb",self.Lov_ia,Moo_ia)

        Joo = [Joo_aa]
        Jvv = [Jvv_aa]
        Jvo = [Jvo_aa]

        Foo = [Foo_aa_t1, Foo_aa_t2]
        Fvv = [Fvv_aa_t1, Fvv_aa_t2]
        Fov = Fov_aa

        return Joo, Jvv, Jvo, Foo, Fvv, Fov

    def create_M_intermediates(self, t1, t2):
        #construct the intermediates for the t2 update:
        #M0
        M0 = lib.einsum("Lkc,kc->L", self.Lov_aa, t1)*2.0

        #Moo
        Moo_aa = lib.einsum("Lic,jc->Lij", self.Lov_aa, t1)
      
        #Mvo
        Mvo_aa = lib.einsum("Lac,ic->Lai", self.Lvv_aa, t1)

        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
        Mvo_t2 = lib.einsum("Lme, imae -> Lai", self.Lov_aa, t2_antisym)

        Moo = [Moo_aa]
        Mvo = [Mvo_aa]
        return M0, Moo, Mvo, Mvo_t2


    def add_t2_to_fock(self, Fvv, Foo, Mvo_t2):    

        #construct antisymmetrized t2:

        #generate the intermediate with full arrays 
        Foo_tmp = lib.einsum("Lie,Lej->ij", self.Lov_aa, Mvo_t2)
        Fvv_tmp = -lib.einsum("Lmb,Lam->ab",self.Lov_aa,Mvo_t2)

        Foo_aa_t1, Foo_aa_t2 = Foo
        Fvv_aa_t1, Fvv_aa_t2 = Fvv

        Foo = [Foo_aa_t1+Foo_tmp, Foo_aa_t2+Foo_tmp]
        Fvv = [Fvv_aa_t1+Fvv_tmp, Fvv_aa_t2+Fvv_tmp]

        return Foo, Fvv


    def R1_residue_active(self, imds, t1, t2, Fov, Fvv, Foo, M0, Mvo, Mvo_t2 ):


        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)
       
        R1 = imds.R1.copy()
       
        Foo_t1, _ = Foo 
        Fvv_t1, _ = Fvv

        Foo_t1 += lib.einsum("ic,jc->ij", Fov, t1)*0.5

        R1 -= lib.einsum("ki,ka->ia", Foo_t1, t1)       

        Fvv_t1 -= lib.einsum("lb,la->ab", Fov, t1)*0.5

        R1 += lib.einsum("ab,ib->ia", Fvv_t1, t1)
        R1 += lib.einsum("me, imae -> ia", Fov, t2_antisym) #many terms
        R1 += lib.einsum("Lia, L -> ia", self.Lov_aa, M0)
        R1 -= lib.einsum("Lji,Laj -> ia", self.Loo_aa, Mvo[0])
        R1 -= lib.einsum("Lji,Laj -> ia", self.Loo_aa, Mvo_t2)
        R1 += lib.einsum("Lae, Lei -> ia", self.Lvv_aa, Mvo_t2)

        return R1

    def _ppl_contraction_symmetric(self, factors, t2):
        """Evaluate the active-space PPL term in dense packed-pair form.

        The symmetric and antisymmetric combinations follow Eqs. (38)-(43) of
        J. Chem. Theory Comput. 2021, 17, 4799-4822.  All packed pair matrices
        are constructed and contracted at once because the high-level active
        space is expected to be small.
        """
        nhole = t2.shape[0]
        nparticle = factors.shape[2]
        nout = factors.shape[1]
        dtype = numpy.result_type(factors, t2)
        result_shape = (nhole, nhole, nout, nout)
        if nhole == 0 or nparticle == 0 or nout == 0:
            return numpy.zeros(result_shape, dtype=dtype)

        hole_i, hole_j = numpy.tril_indices(nhole)
        out_a, out_b = numpy.tril_indices(nout)
        part_e, part_f = numpy.tril_indices(nparticle)

        direct = t2[
            hole_i[:, None],
            hole_j[:, None],
            part_e[None, :],
            part_f[None, :],
        ]
        exchange = t2[
            hole_i[:, None],
            hole_j[:, None],
            part_f[None, :],
            part_e[None, :],
        ]
        t_minus = direct - exchange
        off_diagonal_ef = part_e != part_f
        exchange[:, off_diagonal_ef] += direct[:, off_diagonal_ef]
        exchange[:, ~off_diagonal_ef] = direct[:, ~off_diagonal_ef]
        t_plus = exchange
        del direct

        # Assemble all packed interactions in fixed-width panels.  The panels
        # are used to expose large matrix products to BLAS; they are not a
        # low-memory fallback, and the complete V+ and V- matrices remain live.
        npair_out = len(out_a)
        npair_particle = len(part_e)
        v_plus = numpy.empty((npair_out, npair_particle), dtype=dtype)
        v_minus = numpy.empty_like(v_plus)
        panel_size = 16
        for a_start in range(0, nout, panel_size):
            a_stop = min(nout, a_start + panel_size)
            left_panel = factors[:, a_start:a_stop, :].reshape(
                factors.shape[0], -1
            )
            right_panel = factors[:, :a_stop, :].reshape(
                factors.shape[0], -1
            )
            interaction = lib.dot(left_panel.T, right_panel).reshape(
                a_stop - a_start,
                nparticle,
                a_stop,
                nparticle,
            )

            for a in range(a_start, a_stop):
                pair_start = a * (a + 1) // 2
                pair_stop = (a + 1) * (a + 2) // 2
                row = interaction[a - a_start, :, : a + 1, :].transpose(
                    1, 0, 2
                )
                direct = row[:, part_e, part_f]
                exchange = row[:, part_f, part_e]
                v_plus[pair_start:pair_stop] = direct + exchange
                v_minus[pair_start:pair_stop] = direct - exchange

            del left_panel, right_panel, interaction

        sigma_plus = lib.dot(t_plus, v_plus.T)
        sigma_plus *= 0.5
        sigma_minus = lib.dot(t_minus, v_minus.T)
        sigma_minus *= 0.5
        del t_plus, t_minus, v_plus, v_minus

        residual_ab = sigma_plus + sigma_minus
        residual_ba = sigma_plus - sigma_minus
        result = numpy.zeros(result_shape, dtype=dtype)
        result[
            hole_i[:, None],
            hole_j[:, None],
            out_a[None, :],
            out_b[None, :],
        ] = residual_ab

        off_diagonal_ab = out_a != out_b
        if numpy.any(off_diagonal_ab):
            result[
                hole_i[:, None],
                hole_j[:, None],
                out_b[None, off_diagonal_ab],
                out_a[None, off_diagonal_ab],
            ] = residual_ba[:, off_diagonal_ab]

        off_diagonal_ij = hole_i != hole_j
        if numpy.any(off_diagonal_ij):
            result[
                hole_j[off_diagonal_ij, None],
                hole_i[off_diagonal_ij, None],
                out_b[None, :],
                out_a[None, :],
            ] = residual_ab[off_diagonal_ij]
            if numpy.any(off_diagonal_ab):
                result[
                    hole_j[off_diagonal_ij, None],
                    hole_i[off_diagonal_ij, None],
                    out_a[None, off_diagonal_ab],
                    out_b[None, off_diagonal_ab],
                ] = residual_ba[off_diagonal_ij][:, off_diagonal_ab]

        return result

    def R2_residue_active(self, imds, t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo):
 
        Jvo = Jvo[0]
        Joo = Joo[0]
        Jvv = Jvv[0]

        Imbje, Imbej, Imnij = self.t2_transform_quadratic(t2)  

        Imbje += imds.Imbje_active
        Imbej += imds.Imbej_active
        Imnij += imds.Imnij_active

        R2 = imds.R2.copy()

        #factorized part of the residue:
        R2 += lib.einsum("Lai, Lbj -> ijab", Jvo, Jvo)
        #PPL 
        R2 += self._ppl_contraction_symmetric(Jvv, t2)
        
        #HHL
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo, Joo) + Imnij
        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2) #three possibilities (aa, ai, ia)
        del Wijmn

        #Fock matrix contribution:

        _, Foo_t2 = Foo 
        _, Fvv_t2 = Fvv

        Foo_t2 += lib.einsum("ic,jc->ij", Fov, t1)
        R2_tmp = -lib.einsum("mi, mjab -> ijab", Foo_t2, t2) #only one possibility m has to be inactive.

        Fvv_t2 -= lib.einsum("lb,la->ab", Fov, t1)
        R2_tmp += lib.einsum("bc, ijac -> ijab", Fvv_t2, t2) # only one possibility e has to be inactive

        #N3V3 terms:

        W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv) - Imbje 
        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2)

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo, Jvv) - Imbje*0.5
        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2)

        R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej, t2)

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        del Foo_t2, Fvv_t2

        return R2

    def t2_transform_quadratic(self,t2):

        #I_mn^ij 
        Vnemf = lib.einsum("Lne, Lmf -> nmef", self.Lov_aa, self.Lov_aa)
        Imnij = lib.einsum("mnef, ijef -> mnij", Vnemf, t2) #hh ladder
        #I^je_bm
        Imbej = lib.einsum("nmef, jnbf -> mbej", Vnemf, t2) #exchange
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2) #hp ladder

        return Imbje, Imbej, Imnij
    

    def run_diis(self, t1, t2, adiis):

        vec = self.amplitudes_to_vector(t1, t2)
        t1, t2 = self.vector_to_amplitudes(adiis.update(vec))

        return t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        nov = self.nact_hole * self.nact_particle
        size = nov + nov * (nov + 1) // 2
        vector = numpy.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(t2.transpose(0, 2, 1, 3).reshape(nov, nov), out=vector[nov:])
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nact_hole * self.nact_particle
        t1 = vector[:nov].copy().reshape((self.nact_hole, self.nact_particle))
        # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
        t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
        t2 = t2.reshape(self.nact_hole, self.nact_particle, self.nact_hole, self.nact_particle).transpose(
            0, 2, 1, 3
        )
        return t1, numpy.asarray(t2, order="C")


    def updated_amps(self, imds, t1, t2):
        """
        Following Table XXX in Future Paper
        """
        M0, Moo, Mvo, Mvo_t2 = self.create_M_intermediates(t1, t2)
        Joo, Jvv, Jvo, Foo, Fvv, Fov = self.t1_transform(imds, t1, M0, Moo, Mvo, Mvo_t2)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mvo_t2)

        R1 = self.R1_residue_active(imds, t1, t2, Fov, Fvv, Foo, M0, Mvo, Mvo_t2 )
        R2 = self.R2_residue_active(imds, t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo)

        res1 = R1/ self._eris.eia[numpy.ix_(self.act_hole, self.act_particle)]
        res2 = R2/ self._eris.D[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]
 
        t1 -= res1
        t2 -= res2
        res = numpy.linalg.norm(res1) + numpy.linalg.norm(res2)
#       res = numpy.linalg.norm(res2)
        return res, t1, t2


    def kernel(self, imds, t1full, t2full):

      #we have to extract the active only amplitudes from the full amplitudes:

      t1 = t1full[numpy.ix_(self.act_hole, self.act_particle)].copy()

      t2 = t2full[numpy.ix_(self.act_hole,self.act_hole, self.act_particle,self.act_particle)].copy()

      e_corr = None
      adiis = lib.diis.DIIS()
      err = numpy.inf
      print("Starting High-level MPCC iteration...")
      count = 0
      while err > self.ll_con_tol and count < self.ll_max_its:

          res, t1_new, t2_new = self.updated_amps(imds, t1, t2)
          if self.diis:
              t1_new, t2_new = self.run_diis(t1_new, t2_new, adiis)
          else:
              t1_new, t2_new = t1_new, t2_new

          t1 = t1_new
          t2 = t2_new       
          t1_new = None
          t2_new = None
          count += 1
          err = res
          # NOTE change this to logger!
          #print(f"It {count}; correlation energy {e_corr:.6e}; residual {res:.6e}")
          print(f"It {count}; residual {res:.6e}")

          # NOTE Checking CCSD correlation energy convergence, remove later
          #e_cc = self.get_cc_energy(t1full, t2full, t1, t2)
          #print(f'    CCSD correlation energy: {e_cc}')

      del adiis

      return t1, t2

    def get_cc_energy(self, t1, t2, t1_act, t2_act):

        act_hole = self.frag[0]
        act_particle = self.frag[1]
       
        t1[numpy.ix_(act_hole, act_particle)] = t1_act
        t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act


        fock = self._eris.fov.copy()
        e = 2*lib.einsum('ia,ia', fock, t1)    
        tau = lib.einsum('ia,jb->ijab',t1,t1)                                                         
        tau += t2 
        eris_ovov = lib.einsum("Lia,Ljb->iajb", self._eris.Lov, self._eris.Lov)
        e += 2*lib.einsum('ijab,iajb', tau, eris_ovov)                                                
        e -=  lib.einsum('ijab,ibja', tau, eris_ovov)                                                
        return e.real                       
