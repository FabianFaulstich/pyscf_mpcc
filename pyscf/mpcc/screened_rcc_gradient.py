from pyscf import df
from pyscf import lib
import numpy
from dataclasses import dataclass

class screened:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
        self.frag = frags
        self._set_integral_blocks()
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
    def act_particle(self):
        return self.frag[1]
        
    @property
    def inact_hole(self):
        return numpy.setdiff1d(numpy.arange(self.nocc), self.act_hole)
        
    @property
    def inact_particle(self):
        return numpy.setdiff1d(numpy.arange(self.nvir), self.act_particle)

        # at this point we will classify integralsi: 

    def _set_integral_blocks(self):

        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle
        naux_idx = numpy.arange(self.naux)
        
        self.Loo_ii = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, inact_hole)].copy()
        self.Loo_ia = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, act_hole)].copy()
        self.Loo_ai = self._eris.Loo[numpy.ix_(naux_idx, act_hole, inact_hole)].copy()
        self.Loo_aa = self._eris.Loo[numpy.ix_(naux_idx, act_hole, act_hole)].copy()

        self.Lvv_ii = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, inact_particle)].copy()
        self.Lvv_ia = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, act_particle)].copy()
        self.Lvv_ai = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, inact_particle)].copy()
        self.Lvv_aa = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, act_particle)].copy()

        self.Lov_ii = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, inact_particle)].copy()
        self.Lov_ia = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, act_particle)].copy()
        self.Lov_ai = self._eris.Lov[numpy.ix_(naux_idx, act_hole, inact_particle)].copy()
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)].copy()


    def _set_t1_blocks(self, t1):

        t1_ii = t1[numpy.ix_(self.inact_hole, self.inact_particle)]
        t1_ia = t1[numpy.ix_(self.inact_hole, self.act_particle)]
        t1_ai = t1[numpy.ix_(self.act_hole, self.inact_particle)]
        t1_aa = t1[numpy.ix_(self.act_hole, self.act_particle)]

        return t1_ii, t1_ia, t1_ai, t1_aa

    def t1_transform(self, t1, M, Moo, Mov, Mov_t2):
        #fetch the 3-center integrals in MO basis

        X, Xoo, Xvo = self.get_X(t1)

        Joo, Jvo, Jvv = self.get_J_all(Xoo, Xvo, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Jvo)

        return Joo, Jvv, Jvo, Foo, Fvv, Fov


    def get_X(self, t1):

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return X, Xoo, Xvo

    def get_J_all(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + self._eris.Lvo - lib.einsum("Lij,ja->Lai", Joo, t1)
        )

        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) 

        return Joo, Jvo, Jvv

    def get_F(self, t1, X, Xoo, Xvo, Jvo):

        Foo = self._eris.foo.copy()
        Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
        Foo -= lib.einsum("Lmi,Lmj->ij", self._eris.Loo,Xoo)

        Fvv = self._eris.fvv.copy()
        Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X) 
        Fvv -= lib.einsum("Lma,Lbm->ab",self._eris.Lov,Xvo)
              
        Fov  = self._eris.fov.copy()
        Fov += lib.einsum("Lbj,L->jb", self._eris.Lvo, X)
        #Fov += lib.einsum("Lbj,L->jb", Jvo, X)
        Fov -= lib.einsum("Lij,Lib->jb", Xoo, self._eris.Lov)

        return Foo, Fvv, Fov
 

    def create_M_intermediates(self, t2):
        #construct the intermediates for the t2 update:
        #construct antisymmetrized t2:
        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        nocc = self.nocc
        nvir = self.nvir
        Mov_t2  = lib.einsum("Lme, imae -> Lia", self.Lov_ii, 
                             t2_antisym[numpy.ix_(numpy.arange(nocc), self.inact_hole, numpy.arange(nvir), self.inact_particle)])
        Mov_t2 += lib.einsum(
            "Lme, imae -> Lia",
            self.Lov_ia,
            t2_antisym[
            numpy.ix_(
                numpy.arange(self.nocc),
                self.inact_hole,
                numpy.arange(self.nvir),
                self.act_particle,
            )
            ]
        )
        Mov_t2 += lib.einsum("Lme, imae -> Lia", self.Lov_ai, 
                             t2_antisym[numpy.ix_(numpy.arange(nocc), self.act_hole, numpy.arange(nvir), self.inact_particle)])

        return Mov_t2


    def add_t2_to_fock(self, Fvv, Foo, Mov_t2):    


        #construct antisymmetrized t2:

        Foo_ai, Foo_aa_t1, Foo_aa_t2 = Foo
        Fvv_ai, Fvv_aa_t1, Fvv_aa_t2 = Fvv


        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle


        #generate the intermediate with full arrays 
        Foo_tmp = lib.einsum("Lie,Lje->ij", self.Lov_ai, Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, inact_particle)])
        Foo_tmp += lib.einsum("Lie,Lje->ij", self.Lov_aa, Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, act_particle)])

        Foo_aa_t1 += Foo_tmp
        Foo_aa_t2 += Foo_tmp


        Foo_ai += lib.einsum("Lie,Lje->ij", self.Lov_ai, Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, inact_particle)])
        Foo_ai += lib.einsum("Lie,Lje->ij", self.Lov_aa, Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, act_particle)])


        Fvv_tmp  = -lib.einsum("Lma,Lmb->ab",self.Lov_ia,Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, act_particle)])
        Fvv_tmp -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, act_particle)])


        Fvv_aa_t1 += Fvv_tmp
        Fvv_aa_t2 += Fvv_tmp


        Fvv_ai -= lib.einsum("Lma,Lmb->ab",self.Lov_ia,Mov_t2[numpy.ix_(numpy.arange(self.naux), inact_hole, inact_particle)])
        Fvv_ai -= lib.einsum("Lma,Lmb->ab",self.Lov_aa,Mov_t2[numpy.ix_(numpy.arange(self.naux), act_hole, inact_particle)])

        Foo = [Foo_ai, Foo_aa_t1, Foo_aa_t2]
        Fvv = [Fvv_ai, Fvv_aa_t1, Fvv_aa_t2]

        return Foo, Fvv


    def wefam(self, t1, t2, Fov, Fvv, Foo, M0=None, Mov=None, Mov_t2=None):

# In this contraction, index a spans both active and inactive particles, and all other (e, f, m) indices are active. 
# Follow a strict in-out rule to construct it..

       wefam = lib.einsum("Lea,Lfm->efam", Jvv, Jvo)

       wefam -= lib.einsum("na,nmef->efam", Fov, t2)

       wefam += lib.einsum("Lna,Lom,noef->efam", self._eris.Lov, self._eris.Loo, t2) # factor = 0.5?

       wefam -= lib.einsum("Lea,Lng,gfmn->efam", Jvv, self._eris.Lov, t2_antisym)

       wefam -= lib.einsum("Leg,Lna,gfnm->efam", Jvv, self._eris.Lov, t2)

       wefam -= lib.einsum("Lfg,Lna,gemn->efam", Jvv, self._eris.Lov, t2)

       return wefam


    def wiemn(self, t1, t2, Fov, Fvv, Foo, M0=None, Mov=None, Mov_t2=None): 

        wiemn = lib.einsum("Lim,Len->iemn", Joo, Jvo)

        wiemn -= lib.einsum("if,mnef->iemn", Fov, t2)

        wiemn += lib.einsum("Leg,Lif,mnfg->iemn", Jvv, self._eris.Lov, t2) # factor = 0.5?

        wiemn += lib.einsum("Lim,Lof,noef->iemn", self._eris.Lov, Joo, t2_antisym) # factor = 0.5?
        
        wiemn -= lib.einsum("Lom,Lig,onge->iemn", Joo, self._eris.Lov, t2)

        wiemn -= lib.einsum("Lon,Lig,omeg->iemn", Joo, self._eris.Lov, t2)

        return


    def wijmn(self, t2, Joo):

        wijmn = lib.einsum("Lij,Lmn->ijmn", Joo, Joo)

        v_tmp = lib.einsum("Lie,Ljf->ijef", self._eris.Lov, self._eris.Lov)

        wijmn += lib.einsum("ijef,mnef->ijmn", v_tmp, t2)
 
        return wijmn

    def wamef(self, Jvv, Jov):
        
        wamef = lib.einsum("Lae,Lmf->amef", Jvv, self._eris.Lov)

        return wamef

    def wefab(self, Jvv, t2):

        #Here (e,f) could be inactive and (a,b) could be active.

        wefab = lib.einsum("Lea,Lfb->efab", Jvv, Jvv)        
        wefab += lib.einsum("Lma, Lnb, mnef->efab", self._eris.Lov, self._eris.Lov, t2)

        return wefab

    def wamei(self, Jvv, t2):

        wamei = lib.einsum("Lae,Lmi->amei", Jvv, self._eris.Lov)


        return wamei


    def R2_residue_active(self, t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo):

        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle

        Joo_ai = Joo[0]
        Jvv_ai = Jvv[0]
        Jov_aa = Jov[0]
        Jvv_aa = Jvv[1]
        Joo_aa = Joo[1]

        Fov_aa = Fov[1]
        Fov_ii = Fov[0]
        Fov_ai = Fov[2]
        Fov_ia = Fov[3]         
        Foo_ai = Foo[0]
        Fvv_ai = Fvv[0]
        
        

        return Ijemb, Ijebm, Imnij


    @dataclass
    class _IMDS:
       R1: numpy.ndarray
       R2: numpy.ndarray
       Joo: numpy.ndarray
       Jvv: numpy.ndarray
       Jov: numpy.ndarray
       Foo_t1: numpy.ndarray
       Foo_t2: numpy.ndarray
       Fvv_t1: numpy.ndarray
       Fvv_t2: numpy.ndarray
       Fov: numpy.ndarray
       Ijemb_active: numpy.ndarray
       Ijebm_active: numpy.ndarray
       Imnij_active: numpy.ndarray

    def kernel(self, t1, t2):
      
       M0, Moo, Mov, Mov_t2 = self.create_M_intermediates(t1, t2)
       Joo, Jvv, Jov, Foo, Fvv, Fov = self.t1_transform(t1, M0, Moo, Mov, Mov_t2)
       Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mov_t2)
       R1 = self.R1_residue_active(t1, t2, Fov, Fvv, Foo, M0, Mov, Mov_t2)
       Ijemb_active, Ijebm_active, Imnij_active = self.t2_transform_quadratic(t2)

       R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jov, Fov, Fvv, Foo)
       imds = self._IMDS(R1 = R1, R2 = R2,
                        Joo = Joo[1], Jvv = Jvv[1], Jov = Jov[0],
                        Foo_t1 = Foo[1], Foo_t2 = Foo[2], Fvv_t1 = Fvv[1], Fvv_t2 = Fvv[2],Fov = Fov[1],
                        Ijemb_active = Ijemb_active, Ijebm_active = Ijebm_active, Imnij_active = Imnij_active)       

       return imds
