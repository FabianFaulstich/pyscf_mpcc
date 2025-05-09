import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import ump2 as mp2
from pyscf.mp.ump2 import make_rdm1, make_rdm2
from pyscf import __config__
from pyscf.mp.dfump2_native import DFMP2
from pyscf.cc import ccsd


WITH_T2 = getattr(__config__, "mp_dfmp2_with_t2", True)

def amplitudes_to_vector(t1, t2, out=None):
    nocca, nvira = t1[0].shape
    noccb, nvirb = t1[1].shape
    sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
    sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
    sizeab = nocca * noccb * nvira * nvirb
    vector = np.ndarray(sizea+sizeb+sizeab, t2[0].dtype, buffer=out)
    ccsd.amplitudes_to_vector_s4(t1[0], t2[0], out=vector[:sizea])
    ccsd.amplitudes_to_vector_s4(t1[1], t2[2], out=vector[sizea:])
    vector[sizea+sizeb:] = t2[1].ravel()
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = (nmo, nmo)
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nocc = nocca + noccb
    nvir = nvira + nvirb
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    if vector.size == size:
        #return ccsd.vector_to_amplitudes_s4(vector, nmo, nocc)
        raise RuntimeError('Input vector is GCCSD vecotr')
    else:
        sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
        sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
        sections = np.cumsum([sizea, sizeb])
        veca, vecb, t2ab = np.split(vector, sections)
        t1a, t2aa = ccsd.vector_to_amplitudes_s4(veca, nmoa, nocca)
        t1b, t2bb = ccsd.vector_to_amplitudes_s4(vecb, nmob, noccb)
        t2ab = t2ab.copy().reshape(nocca,noccb,nvira,nvirb)
        return [t1a,t1b], [t2aa,t2ab,t2bb]

def kernel(myll, mo_energy=None, mo_coeff=None, eris=None, with_t2=None):

    np.set_printoptions(linewidth=300, suppress=True)
    # NOTE have this in myll
    tol = 1e-6
    maxcount = 100

    # NOTE initialize Ω
    # myll.t1 = np.zeros((myll.nocc, myll.nvir))

    err = np.inf
    count = 0

    # Constructing 3-legged integrals
    myll.compute_three_center_ints()

    initialize_t1(myll)
    # NOTE need to adjust CC2 energy calculation 
    #e_init = compute_CC2_energy(myll, myll.t1, np.transpose(myll.t2, [2, 0, 3, 1]))
    #print(f"Initial energy: {e_init}")

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
     
        # print(f'Amplitude difference T1: {np.linalg.norm(t1[0] - t1[1])}')
        # print(f'Amplitude difference T2: {np.linalg.norm(mycc.t2[0] - mycc.t2[2])}')
        # breakpoint()
        count += 1
        err = res
        print(f"It {count} Energy progress {DE:.8f} residual progress {res:.6e}")
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
    mydfmp = DFMP2(myll.mymf)
    mydfmp.kernel()

    # t2 = myll.mycc.t2
    t2 = list(mymp.t2)
    # NOTE anti-symmetrized
    t2[0] = t2[0] - np.transpose(t2[0], [0, 1, 3, 2])
    t2[2] = t2[2] - np.transpose(t2[2], [0, 1, 3, 2])
    myll.t2 = mymp.t2

    mf = scf.UHF(mol).density_fit().run()
    myccdf = cc.CCSD(mf)
    myccdf.with_df = df.DF(mf.mol, auxbasis='ccpvdz-ri')
    # myccdf.run()

    print(f"Computed MP2 initial guess")
    print(f"Comparing with UHF energy   : {myll.mymf.e_tot}")
    print(f"Comparing with DF MP2 energy: {mydfmp.e_tot} with {mydfmp.e_corr} corr. energy")
    # print(f"Comparing with DFMP2 energy : {mydfmp.e_tot} with {mymp.e_corr} corr. energy")
    print(
        f"Comparing with CCSD energy  : {myll.mycc.e_tot} with {myll.mycc.e_corr} corr. energy"
    )
    #print(
    #    f"Comparing with DFCCSD energy: {myccdf.e_tot} with {myccdf.e_corr} corr. energy"
    #)
    Loo = myll.Loo
    Lov = myll.Lov
    Lvv = myll.mycc.t1

    '''
    Yvo = np.einsum("ijab,Ljb->Lai", t2, Lov)
    Ω = np.einsum("Lba,Lbi->ia", Lvv, Yvo)
    Ω -= np.einsum("Lji,Laj->ia", Loo, Yvo)

    e_init = np.einsum("Lia,Lai", Lov, Yvo)
    print(f"Initial CC2 energy update : {e_init}")
    '''

    myll.t1 = [0* myll.mycc.t1[0], 0* myll.mycc.t1[1]]


def updated_amp(myll, mo_energy=None, mo_coeff=None, eris=None, with_t2=None):
    """
    Following Table I in  J. Chem. Phys. 146, 194102 (2017) [Mester, Nagy and Kallay]
    """

    t1a = myll.t1[0]
    t1b = myll.t1[1]

    Loo = myll.Loo
    Lov = myll.Lov
    Lvv = myll.Lvv

    LOO = myll.LOO
    LOV = myll.LOV
    LVV = myll.LVV

    fock = myll.fock
    foo = myll.foo
    fOO = myll.fOO
    fvv = myll.fvv
    fVV = myll.fVV

    # Contractions
    # Step 1
    Xvo = np.einsum("Lab,ib->Lai", Lvv, t1a)
    Xoo = np.einsum("Lia,ja->Lij", Lov, t1a)

    XVO = np.einsum("Lab,ib->Lai", LVV, t1b)
    XOO = np.einsum("Lia,ja->Lij", LOV, t1b)

    # Step 2
    Joo = Xoo + Loo
    Ωvo = -1 * np.einsum("Laj,Lji->ai", Xvo, Joo)
    Jvo = Xvo + np.transpose(Lov, (0, 2, 1)) - np.einsum("Lij,ja->Lai", Joo, t1a)

    JOO = XOO + LOO
    ΩVO = -1 * np.einsum("Laj,Lji->ai", XVO, JOO)
    JVO = XVO + np.transpose(LOV, (0, 2, 1)) - np.einsum("Lij,ja->Lai", JOO, t1b)

    # Step 3
    Xa = np.einsum("Lia,ia->L", Lov, t1a)
    Xb = np.einsum("Lia,ia->L", LOV, t1b)
    X = (Xa + Xb)

    # Step 4
    Ωvo += np.einsum("Ljk,ka,Lji->ai", Xoo, t1a, Joo)
    Ωvo += np.einsum("Lai,L->ai", Jvo, X)

    ΩVO += np.einsum("Ljk,ka,Lji->ai", XOO, t1b, JOO)
    ΩVO += np.einsum("Lai,L->ai", JVO, X)

    # NOTE singles contraction with F for non-diagonal basis
    #Ωvo += np.einsum('ia,ia -> ai',t1, fock)
    # NOTE include fov contractrion here 
    Ωvo += np.einsum('ib,ba -> ai',t1a,fvv)
    Ωvo -= np.einsum('ka,ik -> ai',t1a,foo)
 
    ΩVO += np.einsum('ib,ba -> ai',t1b,fVV)
    ΩVO -= np.einsum('ka,ik -> ai',t1b,fOO)

    # Step 5
    Fov = np.einsum("Lbj,L->jb", Jvo, X) - np.einsum("Lij,Lib->jb", Xoo, Lov)
    FOV = np.einsum("Lbj,L->jb", JVO, X) - np.einsum("Lij,Lib->jb", XOO, LOV)

    # Step 6
    erisaa = np.einsum("Lai,Lbj->aibj", Jvo, Jvo)
    erisab = np.einsum("Lai,Lbj->aibj", Jvo, JVO)
    erisbb = np.einsum("Lai,Lbj->aibj", JVO, JVO)

    # Step 7
    t2aa = (erisaa - np.transpose(erisaa, (0, 3, 2, 1))) / myll.D[0]
    t2bb = (erisbb - np.transpose(erisbb, (0, 3, 2, 1))) / myll.D[2]
    t2ab = erisab / myll.D[1]

    # ------------
    # Update : 05/09 
     
    Erisaa = np.einsum("Lia,Ljb->aibj", Lov, Lov)
    Erisbb = np.einsum("Lia,Ljb->aibj", LOV, LOV)

    Erisaa = Erisaa - np.transpose(Erisaa, (0,3,2,1))
    Erisbb = Erisbb - np.transpose(Erisbb, (0,3,2,1))

    Erisab = np.einsum("Lia,LJB->aiBJ", Lov, LOV)

    tau1aa = np.einsum('ia,jb->ijab', t1a, t1a)
    tau1aa-= np.einsum('ia,jb->jiab', t1a, t1a)
    tau1aa = tau1aa - tau1aa.transpose(0,1,3,2)
    tau1aa *= .5
    tau1aa = np.transpose(tau1aa, (2,0,3,1))
    
    tau1bb = np.einsum('ia,jb->ijab', t1b, t1b)
    tau1bb-= np.einsum('ia,jb->jiab', t1b, t1b)
    tau1bb = tau1bb - tau1bb.transpose(0,1,3,2)
    tau1bb *= .5
    tau1bb = np.transpose(tau1bb, (2,0,3,1))
     
    tau1ab = np.einsum('ia,JB->iJaB', t1a, t1b)
    tau1ab += np.einsum('ia,JB->iJaB', t1a, t1b)
    tau1ab *= .5
    tau1ab = np.transpose(tau1ab, (2,0,3,1))

    E2 = 0.25 * np.einsum('aibj, aibj', Erisaa, t2aa + tau1aa)
    E2 += 0.25 * np.einsum('aibj, aibj', Erisbb, t2bb + tau1bb)
    E2 += np.einsum('aibj, aibj', Erisab, t2ab + tau1ab)

    # ------------

    Yvo = np.einsum("aibj,Ljb->Lai", t2aa, Lov)
    Yvo += np.einsum("aiBJ,LJB->Lai", t2ab, LOV)

    YVO = np.einsum("aibj,Ljb->Lai", t2bb, LOV)
    YVO += np.einsum("aiBJ,Lia->LBJ", t2ab, Lov)

    Ωvo += np.einsum("aibj,jb->ai", t2aa, Fov)
    Ωvo += np.einsum("aiBJ,JB->ai", t2ab, FOV)
   
    ΩVO += np.einsum("aibj,jb->ai", t2bb, FOV)
    ΩVO += np.einsum("aiBJ,ia->BJ", t2ab, Fov)

    # Step 8
    Jvv = np.einsum("Ljb,ja->Lba", Lov, t1a) + Lvv
    Ωvo += np.einsum("Lba,Lbi->ai", Jvv, Yvo)
    Ωvo -= np.einsum("Lji,Laj->ai", Joo, Yvo)

    JVV = np.einsum("Ljb,ja->Lba", LOV, t1b) + LVV
    ΩVO += np.einsum("Lba,Lbi->ai", JVV, YVO)
    ΩVO -= np.einsum("Lji,Laj->ai", JOO, YVO)

    # Step 9
    # NOTE check the sign here!
    #e1a = -1* np.einsum("Lij,ja->Lai", Xoo, t1a) + np.einsum("L,ia->Lai", X, t1a) + Jvo
    #e1b = -1* np.einsum("Lij,ja->Lai", XOO, t1b) + np.einsum("L,ia->Lai", X, t1b) + JVO
 
    #ΔEa = np.einsum("Lai,Lai", e1a, Yvo) 
    #ΔEb = np.einsum("Lai,Lai", e1b, YVO)
    
    # -------------
    # NOTE update 05/02
    e1a = np.einsum("L,ia->Lia", X, t1a) + Lov
    e1b = np.einsum("L,ia->Lia", X, t1b) + LOV

    ΔEa = np.einsum("Lia,Lai", e1a, Yvo) 
    ΔEa -= np.einsum("Lai, Lia", np.einsum("Lij,ja->Lai", Xoo, t1a), Lov)
    
    ΔEb = np.einsum("Lia,Lai", e1b, YVO)
    ΔEb -= np.einsum("Lai, Lia", np.einsum("Lij,ja->Lai", XOO, t1b), LOV)
    # -------------

    ΔE = (ΔEa + ΔEb)/2 

    res = np.linalg.norm(myll.t1[0] + Ωvo.T / fock[0])
    res += np.linalg.norm(myll.t1[1] + ΩVO.T / fock[1])

    myll.t1[0] = -Ωvo.T / fock[0]
    myll.t1[1] = -ΩVO.T / fock[1]

    t2 = [t2aa, t2ab, t2bb]
    # NOTE rename Ωvo to just Ω
    
    print(f'E2 :{E2} ')
    print(f'ΔE :{ΔE} ')

    breakpoint()
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

        self.mo_coeff = mf.mo_coeff
        self.moa = self.mo_coeff[0]
        self.mob = self.mo_coeff[1]

        self.nmo = mol.nao
        self.nao = mol.nao
        self.nocca, self.noccb = mol.nelec
        self.nvira = self.nmo - self.nocca
        self.nvirb = self.nmo - self.noccb
        self.eris = mycc.ao2mo(self.mo_coeff)
        self.naux = self.with_df.get_naoaux()

        mo_ea_o = self.eris.mo_energy[0][:self.nocca]
        mo_ea_v = self.eris.mo_energy[0][self.nocca:]
        mo_eb_o = self.eris.mo_energy[1][:self.noccb]
        mo_eb_v = self.eris.mo_energy[1][self.noccb:]

        self.nvira_pair = self.nvira * (self.nvira + 1) // 2
        self.nvirb_pair = self.nvirb * (self.nvirb + 1) // 2

        fock_a = [
            [mf.mo_energy[0][a] - mf.mo_energy[0][i] for a in range(self.nocca, self.nmo)]
            for i in range(self.nocca)
        ]


        fock_b = [
            [mf.mo_energy[1][a] - mf.mo_energy[1][i] for a in range(self.noccb, self.nmo)]
            for i in range(self.noccb)
        ]

        self.fock = [np.array(fock_a), np.array(fock_b)]

        fock_mo_a = mf.mo_coeff[0].T @ mf.get_fock()[0] @ mf.mo_coeff[0]
        fock_mo_b = mf.mo_coeff[1].T @ mf.get_fock()[1] @ mf.mo_coeff[1]
        
        foo = fock_mo_a[: self.nocca, : self.nocca]
        fOO = fock_mo_b[: self.noccb, : self.noccb]

        fvv = fock_mo_a[self.nocca : self.nmo, self.nocca : self.nmo]
        fVV = fock_mo_b[self.noccb : self.nmo, self.noccb : self.nmo]

        self.foo = foo
        self.fOO = fOO
        self.fvv = fvv
        self.fVV = fVV

        '''
        D_a = [
            [
                [
                    [
                        mf.mo_energy[0][i]
                        + mf.mo_energy[0][j]
                        - mf.mo_energy[0][a]
                        - mf.mo_energy[0][b]
                        for i in range(self.nocca)
                    ]
                    for b in range(self.nocca, self.nmo)
                ]
                for j in range(self.nocca)
            ]
            for a in range(self.nocca, self.nmo)
        ]


        D_b = [
            [
                [
                    [
                        mf.mo_energy[1][i]
                        + mf.mo_energy[1][j]
                        - mf.mo_energy[1][a]
                        - mf.mo_energy[1][b]
                        for i in range(self.noccb)
                    ]
                    for b in range(self.noccb, self.nmo)
                ]
                for j in range(self.noccb)
            ]
            for a in range(self.noccb, self.nmo)
        ]
        ''' 
            
        eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
        eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

        D_aa = lib.direct_sum('ia+jb->aibj', eia_a, eia_a)
        D_ab = lib.direct_sum('ia+jb->aibj', eia_a, eia_b)
        D_bb = lib.direct_sum('ia+jb->aibj', eia_b, eia_b)

        self.D = np.array([D_aa, D_ab, D_bb])

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
            Lvv = np.empty((self.naux, self.nvira, self.nvira))

            # (L|bb)
            LOO = np.empty((self.naux, self.noccb, self.noccb))
            LOV = np.empty((self.naux, self.noccb, self.nvirb))
            LVO = np.empty((self.naux, self.nvirb, self.noccb))
            LVV = np.empty((self.naux, self.nvirb, self.nvirb))

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
                Lvv[blk] = Lpq[:, va, va]
                # Lvv[blk] = lib.pack_tril(
                #    Lpq[:, va, va].reshape(-1, self.nvira, self.nvira)
                #)
                # (L|bb)
                Lpq = einsum("Lab,ap,bq->Lpq", eri1, self.mob, self.mob)
                LOO[blk] = Lpq[:, ob, ob]
                LOV[blk] = Lpq[:, ob, vb]
                LVO[blk] = Lpq[:, vb, ob]
                LVV[blk] = Lpq[:, vb, vb]
                # LVV[blk] = lib.pack_tril(
                #    Lpq[:, vb, vb].reshape(-1, self.nvirb, self.nvirb)
                #)

            '''
            Loo = Loo.reshape(self.naux, self.nocca * self.nocca)
            Lov = Lov.reshape(self.naux, self.nocca * self.nvira)
            Lvo = Lvo.reshape(self.naux, self.nocca * self.nvira)
            LOO = LOO.reshape(self.naux, self.noccb * self.noccb)
            LOV = LOV.reshape(self.naux, self.noccb * self.nvirb)
            LVO = LVO.reshape(self.naux, self.noccb * self.nvirb)
            '''

            self.Loo = Loo
            self.Lov = Lov
            self.Lvo = Lvo
            self.Lvv = Lvv

            self.LOO = LOO
            self.LOV = LOV
            self.LVO = LVO
            self.LVV = LVV

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
        # eris._common_init_(self, mo_coeff)
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

    test_co = True


    # Testing CN

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ["C", [0.0, 0.0, 0.0]],
        ["N", [1.177, 0.0, 0.0]]
    ]

    mol.spin = 1
    mol.basis = "cc-pvdz"
    mol.unit = 'Angstrom'
    mol.build()

    mf = mol.UHF().newton()
    mf = mf.run()

    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mf.stability()

    mymp = mp2.MP2(mf).run()
    mycc = cc.CCSD(mf).run()

    print(f'Nuclear repulsion: {mol.get_enuc()}')
    print(f'CFOUR reference  : 18.8831250637')

    print(f'PySCF UHF energy : {mf.e_tot}')
    print(f'CFOUR UHF energy : -92.212689093544')

    print(f'PySCF MP2 corr. energy : {mymp.e_corr}')
    print(f'CFOUR MP2 corr. energy : -0.225540711981')

    mpccll = mpccLL(mf, mycc)
    mpccll.diis = True
    print(f'Reference CC2 Energy : -0.25842706596457')
    mpccll.kernel()

    breakpoint()

    if test_co:
        # Testing CO
        
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ["C", [0.0, 0.0, 0.0]],
            ["O", [0.0, 0.0, 1.4]]
        ]

        mol.basis = "cc-pvdz"
        mol.unit = 'angstrom'
        mol.build()

        mf = mol.UHF().newton()
        dm0 = None
        if dm0 is not None:
            mf = mf.run(dm0)
        else:
            mf = mf.run()

        mo1 = mf.stability()[0]
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1 = mf.stability()[0]
        mf = mf.newton().run(mo1, mf.mo_occ)
        mf.stability()

        mymp = mp2.MP2(mf).run()
        mycc = cc.CCSD(mf).run()

        print(f'Amplitude difference T1: {np.linalg.norm(mycc.t1[0] - mycc.t1[1])}')
        print(f'Amplitude difference T2: {np.linalg.norm(mycc.t2[0] - mycc.t2[2])}')

        mpccll = mpccLL(mf, mycc)
        mpccll.diis = True
        print(f'Reference Energy : -0.360218968957')
        mpccll.kernel()

    else:
        # Testing H2O
        
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8, (0.0, 0.0, -0.12414375)],
            [1, (0.00000000,  1.43052298, 0.98512572)],
            [1, (0.00000000, -1.43052298, 0.98512572)],
        ]

        mol.basis = "cc-pvdz"
        mol.unit = 'bohr'
        mol.build()

        mf = mol.UHF().newton()
        dm0 = None
        if dm0 is not None:
            mf = mf.run(dm0)
        else:
            mf = mf.run()

        mo1 = mf.stability()[0]
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1 = mf.stability()[0]
        mf = mf.newton().run(mo1, mf.mo_occ)
        mf.stability()

        mycc = cc.CCSD(mf).run()

        print(f'Amplitude difference T1: {np.linalg.norm(mycc.t1[0] - mycc.t1[1])}')
        print(f'Amplitude difference T2: {np.linalg.norm(mycc.t2[0] - mycc.t2[2])}')

        mpccll = mpccLL(mf, mycc)
        mpccll.diis = True 
        print(f'Reference Energy -0.204867860539')
        ecc2 = mpccll.kernel()

