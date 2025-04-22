import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import mp2
from pyscf.mp.mp2 import make_rdm1, make_rdm2
from pyscf import __config__
from pyscf.mp.dfmp2_native import DFMP2


WITH_T2 = getattr(__config__, "mp_dfmp2_with_t2", True)


def amplitudes_to_vector(t1, t2, out = None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov + nov*(nov+1)//2
    vector = np.ndarray(size, t1.dtype, buffer=out)
    vector[:nov] = t1.ravel()
    lib.pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov), out=vector[nov:])
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].copy().reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, np.asarray(t2, order='C')


def kernel(myll, mo_energy=None, mo_coeff=None, eris=None, with_t2=None):

    np.set_printoptions(linewidth=300, suppress=True)
    # NOTE have this in myll
    tol = 1e-6
    maxcount = 50

    # NOTE initialize Ω
    # myll.t1 = np.zeros((myll.nocc, myll.nvir))

    err = np.inf
    count = 0

    # Constructing 3-legged integrals
    myll.compute_three_center_ints()

    initialize_t1(myll)
    e_init = compute_CC2_energy(myll, myll.t1, np.transpose(myll.t2, [2, 0, 3, 1]))
    print(f"Initial energy: {e_init}")

    adiis = lib.diis.DIIS()
    while err > tol and count < maxcount:

        res, DE, t1_new, t2_new = updated_amp(myll)
        if myll.diis:
            t1, t2 = run_diis(myll, t1_new, t2_new, adiis)
        else:
            t1 = t1_new
            t2 = t2_new

        myll.t1 = t1
        myll.t2 = t2
       
        count += 1
        err = res
        print(f"It {count} Energy progress {DE:.6e} residual progress {res:.6e}")
    breakpoint()   
    return DE


def run_diis(myll, t1,t2,adiis):
    
    vec = amplitudes_to_vector(t1, t2)
    t1, t2 = vector_to_amplitudes(adiis.update(vec),myll.nmo, myll.nocc)
    
    return t1, t2


def compute_CC2_energy(myll, t1, t2):

    Loo = myll.Loo
    Lov = myll.Lov
    Lvv = myll.Lvv

    # Contractions
    Xvo = np.einsum("Lab,ib->Lai", Lvv, t1)
    Xoo = np.einsum("Lia,ja->Lij", Lov, t1)

    Joo = Xoo + Loo
    Ωvo = -1 * np.einsum("Laj,Lji->ai", Xvo, Joo)
    Jvo = Xvo + np.transpose(Lov, (0, 2, 1)) - np.einsum("Lij,ja->Lai", Joo, t1)

    X = np.einsum("Lia,ia->L", Lov, t1)

    Fov = np.einsum("Lbj,L->jb", Jvo, X) - np.einsum("Lij,Lib->jb", Xoo, Lov)

    eris = np.einsum("Lai,Ljb->aijb", Jvo, Jvo)

    Yvo = np.einsum("aibj,Ljb->Lai", t2, Lov)

    Jvv = np.einsum("Ljb,ja->Lba", Lov, t1) + Lvv

    e1 = np.einsum("Lij,ja->Lai", Xoo, t1) + np.einsum("L,ia->Lai", X, t1) + Jvo
    ΔE = np.einsum("Lai,Lai", e1, Yvo)

    return ΔE


def initialize_t1(myll):

    mymp = mp2.MP2(myll.mymf).run()
    mydfmp = DFMP2(myll.mymf).run()
    # t2 = myll.mycc.t2
    t2 = mymp.t2
    # NOTE anti-symmetrized
    t2 = 2 * t2 - np.transpose(t2, [0, 1, 3, 2])
    myll.t2 = t2

    mf = scf.RHF(mol).density_fit().run()
    myccdf = cc.CCSD(mf)
    myccdf.with_df = df.DF(mf.mol, auxbasis='ccpvdz-ri')
    myccdf.run()

    print(f"Computed MP2 initial guess")
    print(f"Comparing with MP2 energy   : {mymp.e_tot} with {mymp.e_corr} corr. energy")
    print(f"Comparing with DFMP2 energy : {mydfmp.e_tot} with {mymp.e_corr} corr. energy")
    print(
        f"Comparing with CCSD energy  : {myll.mycc.e_tot} with {myll.mycc.e_corr} corr. energy"
    )
    print(
        f"Comparing with DFCCSD energy: {myccdf.e_tot} with {myccdf.e_corr} corr. energy"
    )
    Loo = myll.Loo
    Lov = myll.Lov
    Lvv = myll.Lvv

    Yvo = np.einsum("ijab,Ljb->Lai", t2, Lov)
    Ω = np.einsum("Lba,Lbi->ia", Lvv, Yvo)
    Ω -= np.einsum("Lji,Laj->ia", Loo, Yvo)

    e_init = np.einsum("Lia,Lai", Lov, Yvo)
    print(f"Initial CC2 energy update : {e_init}")
    myll.t1 = 0* Ω


def updated_amp(myll, mo_energy=None, mo_coeff=None, eris=None, with_t2=None):
    """
    Following Table I in  J. Chem. Phys. 146, 194102 (2017) [Mester, Nagy and Kallay]
    """

    t1 = myll.t1
    Loo = myll.Loo
    Lov = myll.Lov
    Lvv = myll.Lvv

    fock = myll.fock
    foo = myll.foo
    fvv = myll.fvv

    # Contractions
    Xvo = np.einsum("Lab,ib->Lai", Lvv, t1)
    Xoo = np.einsum("Lia,ja->Lij", Lov, t1)

    Joo = Xoo + Loo
    Ωvo = -1 * np.einsum("Laj,Lji->ai", Xvo, Joo)
    Jvo = Xvo + np.transpose(Lov, (0, 2, 1)) - np.einsum("Lij,ja->Lai", Joo, t1)

    X = np.einsum("Lia,ia->L", Lov, t1)

    Ωvo += np.einsum("Ljk,ka,Lji->ai", Xoo, t1, Joo)
    Ωvo += np.einsum("Lai,L->ai", Jvo, X)

    # NOTE singles contraction with F for non-diagonal basis
    #Ωvo += np.einsum('ia,ia -> ai',t1, fock)
    # NOTE include fov contractrion here 
    Ωvo += np.einsum('ib,ba -> ai',t1,fvv)
    Ωvo -= np.einsum('ka,ik -> ai',t1,foo)

    Fov = np.einsum("Lbj,L->jb", Jvo, X) - np.einsum("Lij,Lib->jb", Xoo, Lov)

    eris = np.einsum("Lai,Ljb->aijb", Jvo, Jvo)

    t2 = (2 * eris - np.transpose(eris, (0, 3, 2, 1))) / myll.D
    Yvo = np.einsum("aibj,Ljb->Lai", t2, Lov)
    Ωvo += np.einsum("aijb,bj->ai", t2, Fov)

    Jvv = np.einsum("Ljb,ja->Lba", Lov, t1) + Lvv
    Ωvo += np.einsum("Lba,Lbi->ai", Jvv, Yvo)
    Ωvo -= np.einsum("Lji,Laj->ai", Joo, Yvo)

    e1 = np.einsum("Lij,ja->Lai", Xoo, t1) + np.einsum("L,ia->Lai", X, t1) + Jvo
    ΔE = np.einsum("Lai,Lai", e1, Yvo)

    res = np.linalg.norm(myll.t1 + Ωvo.T / fock)

    myll.t1 = -Ωvo.T / fock
    # NOTE rename Ωvo to just Ω
    return res, ΔE, myll.t1, t2


# NOTE how to remove the mp2 dependence?
class mpccLL(mp2.MP2):
    def __init__(self, mf, mycc, frozen=None, mo_coeff=None, mo_occ=None):

        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._keys.update(["with_df"])

        # NOTE can be potentially initialized
        self.t1 = None
        self.t2 = None

        # NOTE use DIIS as default
        self.diis = True

        # NOTE make this better! Only keep the necessary parts of mycc
        self.mycc = mycc
        self.mymf = mf

        self.nmo = mycc.nmo
        self.nocc = mycc.nocc
        self.nvir = self.nmo - self.nocc
        self.eris = mycc.ao2mo(mycc.mo_coeff)
        self.naux = self.with_df.get_naoaux()

        if isinstance(mf, scf.rhf.RHF):
            self.moa = self.eris.mo_coeff
            self.nocca = self.eris.nocc
            self.nao = self.moa.shape[0]
            self.nmoa = self.moa.shape[1]

            self.nvira = self.nmoa - self.nocca
            self.nvira_pair = self.nvira * (self.nvira + 1) // 2

        elif isinstance(mf, scf.uhf.UHF):
            self.moa, self.mob = self.eris.mo_coeff
            self.nocca, self.noccb = self.eris.nocc
            self.nao = self.moa.shape[0]
            self.nmoa = self.moa.shape[1]
            self.nmob = self.mob.shape[1]

            self.nvira = self.nmoa - self.nocca
            self.nvirb = self.nmob - self.noccb
            self.nvira_pair = self.nvira * (self.nvira + 1) // 2
            self.nvirb_pair = self.nvirb * (self.nvirb + 1) // 2

        else:
            print("NotImplementedError")
            exit()

        fock = [
            [mf.mo_energy[a] - mf.mo_energy[i] for a in range(self.nocc, self.nmo)]
            for i in range(self.nocc)
        ]

        fock_mo = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
        foo = fock_mo[: self.nocc, : self.nocc]
        fvv = fock_mo[self.nocc : self.nmo, self.nocc : self.nmo]

        self.foo = foo
        self.fvv = fvv
        self.fock = np.array(fock)

        D = [
            [
                [
                    [
                        mf.mo_energy[i]
                        + mf.mo_energy[j]
                        - mf.mo_energy[a]
                        - mf.mo_energy[b]
                        for i in range(self.nocc)
                    ]
                    for b in range(self.nocc, self.nmo)
                ]
                for j in range(self.nocc)
            ]
            for a in range(self.nocc, self.nmo)
        ]

        self.D = np.array(D)

    def compute_three_center_ints(self):

        if isinstance(mf, scf.rhf.RHF):
            Loo = np.empty((self.naux, self.nocca, self.nocca))
            Lov = np.empty((self.naux, self.nocca, self.nvira))
            Lvo = np.empty((self.naux, self.nvira, self.nocca))
            Lvv = np.empty((self.naux, self.nvira, self.nvira))

            p1 = 0
            oa, va = np.s_[: self.nocca], np.s_[self.nocca :]
            # Transform three-center integrals to MO basis
            einsum = lib.einsum
            for eri1 in self.with_df.loop():
                eri1 = lib.unpack_tril(eri1).reshape(-1, self.nao, self.nao)
                # (L|aa)
                Lpq = einsum("Lab,ap,bq->Lpq", eri1, self.moa, self.moa)
                p0, p1 = p1, p1 + Lpq.shape[0]
                blk = np.s_[p0:p1]
                Loo[blk] = Lpq[:, oa, oa]
                Lov[blk] = Lpq[:, oa, va]
                Lvo[blk] = Lpq[:, va, oa]
                Lvv[blk] = Lpq[:, va, va]
                # Lvv[blk] = lib.pack_tril(Lpq[:,va,va].reshape(-1,self.nvira,self.nvira))

            # Loo = Loo.reshape(self.naux,self.nocca*self.nocca)
            # Lov = Lov.reshape(self.naux,self.nocca*self.nvira)
            # Lvo = Lvo.reshape(self.naux,self.nocca*self.nvira)
            # Lvv = Lvv.reshape(self.naux,self.nvira*self.nvira)

            self.Loo = Loo
            self.Lov = Lov
            self.Lvo = Lvo
            self.Lvv = Lvv

        elif isinstance(mf, scf.uhf.UHF):
            Loo = np.empty((self.naux, self.nocca, self.nocca))
            Lov = np.empty((self.naux, self.nocca, self.nvira))
            Lvo = np.empty((self.naux, self.nvira, self.nocca))
            Lvv = np.empty((self.naux, self.nvira_pair))

            # (L|bb)
            LOO = np.empty((self.naux, self.noccb, self.noccb))
            LOV = np.empty((self.naux, self.noccb, self.nvirb))
            LVO = np.empty((self.naux, self.nvirb, self.noccb))
            LVV = np.empty((self.naux, self.nvirb_pair))
            p1 = 0
            oa, va = np.s_[: self.nocca], np.s_[self.nocca :]
            ob, vb = np.s_[: self.noccb], np.s_[self.noccb :]
            # Transform three-center integrals to MO basis
            einsum = lib.einsum
            for eri1 in self.with_df.loop():
                eri1 = lib.unpack_tril(eri1).reshape(-1, self.nao, self.nao)
                # (L|aa)
                Lpq = einsum("Lab,ap,bq->Lpq", eri1, self.moa, self.moa)
                p0, p1 = p1, p1 + Lpq.shape[0]
                blk = np.s_[p0:p1]
                Loo[blk] = Lpq[:, oa, oa]
                Lov[blk] = Lpq[:, oa, va]
                Lvo[blk] = Lpq[:, va, oa]
                Lvv[blk] = lib.pack_tril(
                    Lpq[:, va, va].reshape(-1, self.nvira, self.nvira)
                )
                # (L|bb)
                Lpq = einsum("Lab,ap,bq->Lpq", eri1, self.mob, self.mob)
                LOO[blk] = Lpq[:, ob, ob]
                LOV[blk] = Lpq[:, ob, vb]
                LVO[blk] = Lpq[:, vb, ob]
                LVV[blk] = lib.pack_tril(
                    Lpq[:, vb, vb].reshape(-1, self.nvirb, self.nvirb)
                )

            Loo = Loo.reshape(self.naux, self.nocca * self.nocca)
            Lov = Lov.reshape(self.naux, self.nocca * self.nvira)
            Lvo = Lvo.reshape(self.naux, self.nocca * self.nvira)
            LOO = LOO.reshape(self.naux, self.noccb * self.noccb)
            LOV = LOV.reshape(self.naux, self.noccb * self.nvirb)
            LVO = LVO.reshape(self.naux, self.noccb * self.nvirb)

            self.Loo = Loo
            self.Lov = Lov
            self.Lvo = Lvo
            self.LOO = LOO
            self.LOV = LOV
            self.LVO = LVO

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return mp2.MP2.reset(self, mol)

    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order="F")
        nmo = mo.shape[1]
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory * 0.9 - mem_now)
        blksize = int(
            min(
                naux,
                max(
                    with_df.blockdim,
                    (max_memory * 1e6 / 8 - nocc * nvir**2 * 2) / (nocc * nvir),
                ),
            )
        )
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym="s2", out=Lov)
            yield Lov

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        # Initialize only the mo_coeff
        eris._common_init_(self, mo_coeff)
        return eris

    def make_rdm1(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm1(self, t2, ao_repr=ao_repr)

    def make_rdm2(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm2(self, t2, ao_repr=ao_repr)

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)


MP2 = mpccLL

from pyscf import scf, cc

# scf.hf.RHF.DFMP2 = lib.class_as_method(DFMP2)
# scf.rohf.ROHF.DFMP2 = None
# scf.uhf.UHF.DFMP2 = None

# del (WITH_T2)


if __name__ == "__main__":
    from pyscf import scf
    from pyscf import gto

    # Testing CO
    
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ["C", [0.0, 0.0, 0.0]],
        ["O", [0.0, 0.0, 1.4]]
    ]

    mol.basis = "cc-pvdz"
    mol.build()
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()

    mpccll = mpccLL(mf, mycc)
    mpccll.diis = True
    print(f'Reference Energy : -0.371111485169')
    mpccll.kernel()


    # Testing H2O
    
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.757, 0.587)],
        [1, (0.0, 0.757, 0.587)],
    ]

    mol.basis = "cc-pvdz"
    mol.build()
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()

    mpccll = mpccLL(mf, mycc)
    mpccll.diis = False
    print(f'Reference Energy -0.204867860525')
    ecc2 = mpccll.kernel()

