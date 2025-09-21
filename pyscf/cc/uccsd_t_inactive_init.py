#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import scipy
from pyscf import lib
from pyscf.cc import uccsd
from pyscf import df
import time

'''
UCCSD(T)
'''

def kernel(mycc, eris, t1, t2, act_hole, act_particle): 

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

    mo_coeff = eris.mo_coeff
#generate all transformed integrals:

    ints = _make_4c_integrals_bare(mycc, eris, t1, t2, mo_coeff)
#   ints = _make_4c_integrals(mycc, eris, t1, t2, mo_coeff)

    u3 = update_amps(ints, t2)

    return u3

def update_amps(ints, t2):
    '''Update non-canonical MP2 amplitudes'''
    #assert (isinstance(eris, _ChemistsERIs))

    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))

    def cyclic_hole(u):
        return (u + u.transpose(1,2,0,3,4,5)+u.transpose(2,0,1,3,4,5))     

    def cyclic_particle(u):
        return (u + u.transpose(0,1,2,4,5,3)+u.transpose(0,1,2,5,3,4))     


    t2aa,t2ab,t2bb = t2

    mo_ea_o = ints.ea_occ 
    mo_ea_v = ints.ea_vir
    mo_eb_o = ints.eb_occ
    mo_eb_v = ints.eb_vir

    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)


    d3aaa  = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    d3bbb = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    d3baa = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    d3bba = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eia)


    # aaa
    start = time.time()

    x  = lib.einsum('ijae,beck->ijkabc', t2aa, ints.Wvvvo )
    x -= lib.einsum('imab,mjck->ijkabc', t2aa, ints.Woovo )

    end = time.time()
    print("time to make aaa contribution to t3", end-start)

    start = time.time()
    u3aaa = cyclic_hole(cyclic_particle(x)) 
    end = time.time()
    print("time to make aaa permutation", end-start)

    # bbb
    x = lib.einsum('ijae,beck->ijkabc', t2bb, ints.WVVVO )
    x -= lib.einsum('imab,mjck->ijkabc',t2bb, ints.WOOVO )

    u3bbb = cyclic_particle(cyclic_hole(x)) 

    # baa
    u3baa  = lib.einsum('jIeA,beck->IjkAbc', t2ab, ints.Wvvvo)    # 2

   #P(jk)
    r = u3baa - u3baa.transpose(0,2,1,3,4,5)

    u3baa = lib.einsum('jIbE,AEck->IjkAbc', t2ab, ints.WVVvo)     # 2

   #P(bc)p(jk)
    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = lib.einsum('jkbe,ceAI->IjkAbc', t2aa, ints.WvvVO)
    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mIbA,mjck->IjkAbc', t2ab, ints.Woovo) 

    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('jMbA,MIck->IjkAbc', t2ab, ints.WOOvo ) 

   #P(bc)P(jk)

    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mjcb,mkAI->IjkAbc', t2aa, ints.WooVO )

    #P(jk) 
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)

    # bba
    u3bba  = lib.einsum('IJAE,BEck->IJkABc', t2bb, ints.WVVvo ) 
#  P(AB)

    v = u3bba - u3bba.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('kJcE,BEAI->IJkABc', t2ab, ints.WVVVO )  
#  P(IJ) 
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = lib.einsum('kIeA,ceBJ->IJkABc', t2ab, ints.WvvVO )
 # P(IJ)P(AB)  

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = -lib.einsum('IMAB,MJck->IJkABc', t2bb, ints.WOOvo ) 
#P(IJ)

    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = -lib.einsum('kMcB,MJAI->IJkABc', t2ab, ints.WOOVO ) 

#P(AB)
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)
    u3bba = -lib.einsum('mJcB,mkAI->IJkABc', t2ab, ints.WooVO )

#P(IJ)P(AB)

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)


    u3aaa /=d3aaa 
    u3bbb /=d3bbb 
    u3bba = v/d3bba 
    u3baa = r/d3baa 


    u3aaa_tr = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3aaa, ints.umat_occ_a,
                          ints.umat_occ_a, ints.umat_occ_a, ints.umat_vir_a, ints.umat_vir_a, ints.umat_vir_a,optimize=True)


    u3bbb_tr = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3bbb, ints.umat_occ_b,
                          ints.umat_occ_b, ints.umat_occ_b, ints.umat_vir_b, ints.umat_vir_b, ints.umat_vir_b,optimize=True)


    u3baa_tr = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3baa, ints.umat_occ_b,
                          ints.umat_occ_a, ints.umat_occ_a, ints.umat_vir_b, ints.umat_vir_a, ints.umat_vir_a,optimize=True)


    u3bba_tr = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3bba, ints.umat_occ_b,
                          ints.umat_occ_b, ints.umat_occ_a, ints.umat_vir_b, ints.umat_vir_b, ints.umat_vir_a,optimize=True)

    w3 = u3aaa_tr, u3bbb_tr, u3baa_tr, u3bba_tr

    return w3 

def _make_4c_integrals_bare(mycc, eris, t1, t2, mo_coeff):
    assert mycc._scf.istype('UHF')
#    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(mycc.stdout, mycc.verbose)
#   eris = _ChemistsERIs()
#   eris._common_init_(mycc, mo_coeff)

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

    moa, mob = mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nvira_pair = nvira * (nvira + 1) // 2
    nvirb_pair = nvirb * (nvirb + 1) // 2

    class _ints: pass
    ints = _ints()

    Foo = eris.focka[:nocca,:nocca].copy()
    Fvv = eris.focka[nocca:,nocca:].copy()

    FOO = eris.fockb[:noccb,:noccb].copy()
    FVV = eris.fockb[noccb:,noccb:].copy()

    Fov  = eris.focka[:nocca,nocca:].copy()
    FOV  = eris.fockb[:noccb,noccb:].copy()

    Foo = (Foo + Foo.conj().T) / 2
    Fvv = (Fvv + Fvv.conj().T) / 2
    FOO = (FOO + FOO.conj().T) / 2
    FVV = (FVV + FVV.conj().T) / 2

    e_occ_a, umat_occ_a = scipy.linalg.eigh(Foo)     
    e_occ_b, umat_occ_b = scipy.linalg.eigh(FOO)     

    e_vir_a, umat_vir_a = scipy.linalg.eigh(Fvv)     
    e_vir_b, umat_vir_b = scipy.linalg.eigh(FVV)     

    umat_occ_a = numpy.real(umat_occ_a)
    umat_occ_b = numpy.real(umat_occ_b)
                
    umat_vir_a = numpy.real(umat_vir_a)
    umat_vir_b = numpy.real(umat_vir_b)

    print("eigenvalues of dressed fock:", e_occ_a, e_occ_b)

# Wvvvo, WVVVO, WVVvo, WvvVO: untransformed.. 
    
    wvvvo = numpy.asarray(eris.get_ovvv()).transpose(3,2,1,0)
    wVVVO = numpy.asarray(eris.get_OVVV()).transpose(3,2,1,0)
    wVVvo = numpy.asarray(eris.get_ovVV()).transpose(3,2,1,0)
    wvvVO = numpy.asarray(eris.get_OVvv()).transpose(3,2,1,0)

    wvvvo_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wvvvo, umat_vir_a, umat_vir_a,  umat_vir_a, umat_occ_a, optimize=True)
    wVVVO_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wVVVO, umat_vir_b, umat_vir_b,  umat_vir_b, umat_occ_b, optimize=True)
    wVVvo_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wVVvo, umat_vir_b, umat_vir_b,  umat_vir_a, umat_occ_a, optimize=True)
    wvvVO_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wvvVO, umat_vir_a, umat_vir_a,  umat_vir_b, umat_occ_b, optimize=True)

    wvvvo_tr = wvvvo_tr - wvvvo_tr.transpose(2,1,0,3)
    wVVVO_tr = wVVVO_tr - wVVVO_tr.transpose(2,1,0,3)

#   wvvov, wvvOV, wVVov, wVVOV

    wvvov = numpy.asarray(eris.get_ovvv()).transpose(2,3,0,1)
    wVVOV = numpy.asarray(eris.get_OVVV()).transpose(2,3,0,1)
    wvvOV = numpy.asarray(eris.get_OVvv()).transpose(2,3,0,1)
    wVVov = numpy.asarray(eris.get_ovVV()).transpose(2,3,0,1)

    wvvov_tr = numpy.einsum("abic,aA, bB, iI, cC -> ABIC", wvvov, umat_vir_a, umat_vir_a, umat_occ_a, umat_vir_a, optimize=True)
    wVVOV_tr = numpy.einsum("abic,aA, bB, iI, cC -> ABIC", wVVOV, umat_vir_b, umat_vir_b, umat_occ_b, umat_vir_b, optimize=True)
    wvvOV_tr = numpy.einsum("abic,aA, bB, iI, cC -> ABIC", wvvOV, umat_vir_a, umat_vir_a, umat_occ_b, umat_vir_b, optimize=True)
    wVVov_tr = numpy.einsum("abic,aA, bB, iI, cC -> ABIC", wVVov, umat_vir_b, umat_vir_b, umat_occ_a, umat_vir_a, optimize=True)

    wvvov_tr = wvvov_tr - wvvov_tr.transpose(0,3,2,1)
    wVVOV_tr = wVVOV_tr - wVVOV_tr.transpose(0,3,2,1)

#   Wovoo, WOVoo, WovOO, WOVOO

    wovoo = numpy.asarray(eris.ovoo)
    wOVOO = numpy.asarray(eris.OVOO)
    wOVoo = numpy.asarray(eris.OVoo)
    wovOO = numpy.asarray(eris.ovOO)

    wovoo_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wovoo, umat_occ_a, umat_vir_a, umat_occ_a, umat_occ_a, optimize=True)
    wOVOO_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wOVOO, umat_occ_b, umat_vir_b, umat_occ_b, umat_occ_b, optimize=True)
    wOVoo_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wOVoo, umat_occ_b, umat_vir_b, umat_occ_a, umat_occ_a, optimize=True)
    wovOO_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wovOO, umat_occ_a, umat_vir_a, umat_occ_b, umat_occ_b, optimize=True)

    wovoo_tr = wovoo_tr - wovoo_tr.transpose(2,1,0,3)
    wOVOO_tr = wOVOO_tr - wOVOO_tr.transpose(2,1,0,3)

#   Woovo, WooVO, WOOvo, WOOVO 

    woovo = numpy.asarray(eris.ovoo).transpose(3,2,1,0)
    wOOVO = numpy.asarray(eris.OVOO).transpose(3,2,1,0)
    wooVO = numpy.asarray(eris.OVoo).transpose(3,2,1,0)
    wOOvo = numpy.asarray(eris.ovOO).transpose(3,2,1,0)

    woovo_tr = numpy.einsum("ijak, iI, jJ, aA, kK -> IJAK", woovo, umat_occ_a, umat_occ_a, umat_vir_a, umat_occ_a, optimize=True)
    wOOVO_tr = numpy.einsum("ijak, iI, jJ, aA, kK -> IJAK", wOOVO, umat_occ_b, umat_occ_b, umat_vir_b, umat_occ_b, optimize=True)
    wooVO_tr = numpy.einsum("ijak, iI, jJ, aA, kK -> IJAK", wooVO, umat_occ_a, umat_occ_a, umat_vir_b, umat_occ_b, optimize=True)
    wOOvo_tr = numpy.einsum("ijak, iI, jJ, aA, kK -> IJAK", wOOvo, umat_occ_b, umat_occ_b, umat_vir_a, umat_occ_a, optimize=True)
   
    woovo_tr = woovo_tr - woovo_tr.transpose(0,3,2,1)
    wOOVO_tr = wOOVO_tr - wOOVO_tr.transpose(0,3,2,1)

# wovov, wOVOV, wovOV

    wovov = numpy.asarray(eris.ovov)
    wOVOV = numpy.asarray(eris.OVOV)
    wovOV = numpy.asarray(eris.ovOV)


    wovov_tr = numpy.einsum("iajb, iI, aA, jJ, bB -> IAJB", wovov, umat_occ_a, umat_vir_a, umat_occ_a, umat_vir_a, optimize=True)
    wOVOV_tr = numpy.einsum("iajb, iI, aA, jJ, bB -> IAJB", wOVOV, umat_occ_b, umat_vir_b, umat_occ_b, umat_vir_b, optimize=True)
    wovOV_tr = numpy.einsum("iajb, iI, aA, jJ, bB -> IAJB", wovOV, umat_occ_a, umat_vir_a, umat_occ_b, umat_vir_b, optimize=True)

    wovov_tr = wovov_tr - wovov_tr.transpose(0,3,2,1)
    wOVOV_tr = wOVOV_tr - wOVOV_tr.transpose(0,3,2,1)

    ints.ea_occ = numpy.real(e_occ_a) 
    ints.eb_occ = numpy.real(e_occ_b)

    ints.ea_vir = numpy.real(e_vir_a)
    ints.eb_vir = numpy.real(e_vir_b)


    t2_tr_aa = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[0], umat_occ_a,
                          umat_occ_a, umat_vir_a, umat_vir_a, optimize=True)

    t2_tr_ab = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[1], umat_occ_a,
                          umat_occ_b, umat_vir_a, umat_vir_b, optimize=True)

    t2_tr_bb = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[2], umat_occ_b,
                          umat_occ_b, umat_vir_b, umat_vir_b, optimize=True)

    t1_tr_aa = lib.einsum("ia, iI, aA -> IA", t1[0], umat_occ_a, umat_vir_a)
    t1_tr_bb = lib.einsum("ia, iI, aA -> IA", t1[1], umat_occ_b, umat_vir_b)

    Fov_tr = lib.einsum("ia, iI, aA -> IA", Fov, umat_occ_a, umat_vir_a)
    FOV_tr = lib.einsum("ia, iI, aA -> IA", FOV, umat_occ_b, umat_vir_b)

    ints.ea_occ = numpy.real(e_occ_a) 
    ints.eb_occ = numpy.real(e_occ_b)

    ints.ea_vir = numpy.real(e_vir_a)
    ints.eb_vir = numpy.real(e_vir_b)

    ints.Wvvvo, ints.WVVVO, ints.WVVvo, ints.WvvVO = wvvvo_tr, wVVVO_tr, wVVvo_tr, wvvVO_tr
    ints.Wvvov, ints.WvvOV, ints.WVVov, ints.WVVOV = wvvov_tr, wvvOV_tr, wVVov_tr, wVVOV_tr

    ints.Wovoo, ints.WOVoo, ints.WovOO, ints.WOVOO = wovoo_tr, wOVoo_tr, wovOO_tr, wOVOO_tr
    ints.Woovo, ints.WooVO, ints.WOOvo, ints.WOOVO = woovo_tr, wooVO_tr, wOOvo_tr, wOOVO_tr

    ints.Wovov, ints.WOVOV, ints.WovOV = wovov_tr, wOVOV_tr, wovOV_tr
    
    ints.Fov, ints.FOV = Fov_tr, FOV_tr 

    ints.umat_occ_a = umat_occ_a
    ints.umat_vir_a = umat_vir_a
    ints.umat_occ_b = umat_occ_b
    ints.umat_vir_b = umat_vir_b

    ints.t2aa, ints.t2ab, ints.t2bb = t2_tr_aa, t2_tr_ab, t2_tr_bb
    ints.t1a, ints.t1b  = t1_tr_aa, t1_tr_bb

#   ints.t2aa, ints.t2ab, ints.t2bb = t2aa, t2ab, t2bb
#   ints.t1a, ints.t1b  = t1a, t1b

    return ints


def _make_4c_integrals(mycc, eris, t1, t2, mo_coeff):
    assert mycc._scf.istype('UHF')
#    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(mycc.stdout, mycc.verbose)
#   eris = _ChemistsERIs()
#   eris._common_init_(mycc, mo_coeff)

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

    moa, mob = mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nvira_pair = nvira * (nvira + 1) // 2
    nvirb_pair = nvirb * (nvirb + 1) // 2
    with_df = mycc.with_df
    naux = eris.naux = with_df.get_naoaux()

    ints_3c = _make_df_eris(mycc, eris)

    class _ints: pass
    ints = _ints()

    Xvo = lib.einsum("Lab,ib->Lai", ints_3c.Lvv, t1a)
    Xoo = lib.einsum("Lia,ja->Lij", ints_3c.Lov, t1a)

    XVO = lib.einsum("Lab,ib->Lai", ints_3c.LVV, t1b)
    XOO = lib.einsum("Lia,ja->Lij", ints_3c.LOV, t1b)

    X = lib.einsum("Lia,ia->L", ints_3c.Lov, t1a) + lib.einsum("Lia,ia->L", ints_3c.LOV, t1b)

    Joo = Xoo + ints_3c.Loo
    Jvo = (
        Xvo + ints_3c.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1a)
    )

    Jvv = ints_3c.Lvv - lib.einsum("Lkb,ka->Lab", ints_3c.Lov, t1a)

    Jov = ints_3c.Lov


    JOO = XOO + ints_3c.LOO
    JVO = (
        XVO + ints_3c.LVO - lib.einsum("Lji,ja->Lai", JOO, t1b)
    )

    JVV = ints_3c.LVV - lib.einsum("Lkb,ka->Lab", ints_3c.LOV, t1b)
    JOV = ints_3c.LOV

    Foo = eris.focka[:nocca,:nocca].copy()
    Foo += lib.einsum("Lij,L->ij", ints_3c.Loo, X)
    Foo -= lib.einsum("Lmj,Lim->ij", ints_3c.Loo,Xoo)

    Fvv = eris.focka[nocca:,nocca:].copy()
    Fvv += lib.einsum("Lab,L->ab",ints_3c.Lvv,X)
    Fvv -= lib.einsum("Lmb,Lam->ab",ints_3c.Lov,Xvo)

    FOO = eris.fockb[:noccb,:noccb].copy()
    FOO += lib.einsum("Lij,L->ij", ints_3c.LOO, X)
    FOO -= lib.einsum("Lmj,Lim->ij", ints_3c.LOO,XOO)

    FVV = eris.fockb[noccb:,noccb:].copy()
    FVV += lib.einsum("Lab,L->ab",ints_3c.LVV,X)
    FVV -= lib.einsum("Lmb,Lam->ab",ints_3c.LOV,XVO)

    Fov  = eris.focka[:nocca,nocca:].copy()
    Fov += lib.einsum("Ljb,L->jb", ints_3c.Lov, X)
    Fov -= lib.einsum("Lji,Lib->jb", Xoo, ints_3c.Lov)

    FOV  = eris.fockb[:noccb,noccb:].copy()
    FOV += lib.einsum("Ljb,L->jb", ints_3c.LOV, X)
    FOV -= lib.einsum("Lji,Lib->jb", XOO, ints_3c.LOV)

    Foo += lib.einsum("ic,jc->ij",Fov,t1a)
    Fvv -= lib.einsum("lb,la->ab",Fov,t1a)

    FOO += lib.einsum("ic,jc->ij",FOV,t1b)
    FVV -= lib.einsum("lb,la->ab",FOV,t1b)

    e_occ_a, umat_occ_a = scipy.linalg.eig(Foo)     
    e_occ_b, umat_occ_b = scipy.linalg.eig(FOO)     

    e_vir_a, umat_vir_a = scipy.linalg.eig(Fvv)     
    e_vir_b, umat_vir_b = scipy.linalg.eig(FVV)     

    umat_occ_a = numpy.real(umat_occ_a)
    umat_occ_b = numpy.real(umat_occ_b)
                
    umat_vir_a = numpy.real(umat_vir_a)
    umat_vir_b = numpy.real(umat_vir_b)

    print("eigenvalues of dressed fock:", e_occ_a, e_occ_b)

    Joo_tr = lib.einsum("Lij, iI, jJ -> LIJ", Joo, umat_occ_a, umat_occ_a)
    JOO_tr = lib.einsum("Lij, iI, jJ -> LIJ", JOO, umat_occ_b, umat_occ_b)

    Jvv_tr = lib.einsum("Lab, aA, bB -> LAB", Jvv, umat_vir_a, umat_vir_a)
    JVV_tr = lib.einsum("Lab, aA, bB -> LAB", JVV, umat_vir_b, umat_vir_b)

    Jvo_tr = lib.einsum("Lai, aA, iI -> LAI", Jvo, umat_vir_a, umat_occ_a)
    JVO_tr = lib.einsum("Lai, aA, iI -> LAI", JVO, umat_vir_b, umat_occ_b)

    Jov_tr = lib.einsum("Lia, iI, aA -> LIA", ints_3c.Lov, umat_occ_a, umat_vir_a)
    JOV_tr = lib.einsum("Lia, iI, aA -> LIA", ints_3c.LOV, umat_occ_b, umat_vir_b)

# We have to carry out some more transformations as well (t2 and Fov)

    t2_tr_aa = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[0], umat_occ_a,
                          umat_occ_a, umat_vir_a, umat_vir_a, optimize=True)

    t2_tr_ab = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[1], umat_occ_a,
                          umat_occ_b, umat_vir_a, umat_vir_b, optimize=True)

    t2_tr_bb = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[2], umat_occ_b,
                          umat_occ_b, umat_vir_b, umat_vir_b, optimize=True)

    t1_tr_aa = lib.einsum("ia, iI, aA -> IA", t1[0], umat_occ_a, umat_vir_a)
    t1_tr_bb = lib.einsum("ia, iI, aA -> IA", t1[1], umat_occ_b, umat_vir_b)

    Fov_tr = lib.einsum("ia, iI, aA -> IA", Fov, umat_occ_a, umat_vir_a)
    FOV_tr = lib.einsum("ia, iI, aA -> IA", FOV, umat_occ_b, umat_vir_b)

# Now construct all different integral types.. 

# Wvvvo, WVVVO, WVVvo, WvvVO: untransformed.. 

    wvvvo = lib.einsum("Lae, Lbi -> aebi", Jvv_tr, Jvo_tr) 
    wvvvo = wvvvo - wvvvo.transpose(2,1,0,3)

    wVVVO = lib.einsum("Lae, Lbi -> aebi", JVV_tr, JVO_tr) 
    wVVVO = wVVVO - wVVVO.transpose(2,1,0,3)

    wVVvo = lib.einsum("Lae, Lbi -> aebi", JVV_tr, Jvo_tr) 

    wvvVO = lib.einsum("Lae, Lbi -> aebi", Jvv_tr, JVO_tr) 

#   wvvov, wvvOV, wVVov, wVVOV

    wvvov = lib.einsum("Lae, Lmf -> aemf",Jvv_tr, Jov_tr) 
    wvvov = wvvov - wvvov.transpose(0,3,2,1)

    wVVOV = lib.einsum("Lae, Lmf -> aemf",JVV_tr, JOV_tr) 
    wVVOV = wVVOV - wVVOV.transpose(0,3,2,1)

    wVVov = lib.einsum("Lae, Lmf -> aemf",JVV_tr, Jov_tr) 
    wvvOV = lib.einsum("Lae, Lmf -> aemf",Jvv_tr, JOV_tr) 

#   Wovoo, WOVoo, WovOO, WOVOO

    wovoo = lib.einsum("Lkc, Ljm -> kcjm",Jov_tr, Joo_tr) 
    wovoo = wovoo - wovoo.transpose(2,1,0,3)

#   wOVOO = lib.ddot(JOV.T, JOO).reshape(noccb,nvirb,noccb,noccb)
    wOVOO = lib.einsum("Lkc, Ljm -> kcjm",JOV_tr, JOO_tr) 
    wOVOO = wOVOO - wOVOO.transpose(2,1,0,3)

    wOVoo = lib.einsum("Lkc, Ljm -> kcjm",JOV_tr, Joo_tr) 
    wovOO = lib.einsum("Lkc, Ljm -> kcjm",Jov_tr, JOO_tr) 

#   Woovo, WooVO, WOOvo, WOOVO 
   
    woovo = lib.einsum("Lmj, Lck -> mjck",Joo_tr, Jvo_tr) 
    woovo = woovo - woovo.transpose(0,3,2,1)
   
    wOOVO = lib.einsum("Lmj, Lck -> mjck",JOO_tr, JVO_tr) 
    wOOVO = wOOVO - wOOVO.transpose(0,3,2,1)

    wooVO = lib.einsum("Lmj, Lck -> mjck",Joo_tr, JVO_tr) 

    wOOvo = lib.einsum("Lmj, Lck -> mjck",JOO_tr, Jvo_tr) 

# wovov, wOVOV, wovOV

    wovov = lib.einsum("Lia, Ljb -> iajb",Jov_tr, Jov_tr) 
    wOVOV = lib.einsum("Lia, Ljb -> iajb",JOV_tr, JOV_tr) 
    wovOV = lib.einsum("Lia, Ljb -> iajb",Jov_tr, JOV_tr) 

    wovov = wovov - wovov.transpose(0,3,2,1)
    wOVOV = wOVOV - wOVOV.transpose(0,3,2,1)

    ints.ea_occ = numpy.real(e_occ_a) 
    ints.eb_occ = numpy.real(e_occ_b)

    ints.ea_vir = numpy.real(e_vir_a)
    ints.eb_vir = numpy.real(e_vir_b)


    t2_tr_aa = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[0], umat_occ_a,
                          umat_occ_a, umat_vir_a, umat_vir_a, optimize=True)

    t2_tr_ab = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[1], umat_occ_a,
                          umat_occ_b, umat_vir_a, umat_vir_b, optimize=True)

    t2_tr_bb = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[2], umat_occ_b,
                          umat_occ_b, umat_vir_b, umat_vir_b, optimize=True)

    t1_tr_aa = lib.einsum("ia, iI, aA -> IA", t1[0], umat_occ_a, umat_vir_a)
    t1_tr_bb = lib.einsum("ia, iI, aA -> IA", t1[1], umat_occ_b, umat_vir_b)

    Fov_tr = lib.einsum("ia, iI, aA -> IA", Fov, umat_occ_a, umat_vir_a)
    FOV_tr = lib.einsum("ia, iI, aA -> IA", FOV, umat_occ_b, umat_vir_b)


    ints.ea_occ = numpy.real(e_occ_a) 
    ints.eb_occ = numpy.real(e_occ_b)

    ints.ea_vir = numpy.real(e_vir_a)
    ints.eb_vir = numpy.real(e_vir_b)

    ints.Wvvvo, ints.WVVVO, ints.WVVvo, ints.WvvVO = wvvvo, wVVVO, wVVvo, wvvVO

    ints.Wvvov, ints.WvvOV, ints.WVVov, ints.WVVOV = wvvov, wvvOV, wVVov, wVVOV

    ints.Wovoo, ints.WOVoo, ints.WovOO, ints.WOVOO = wovoo, wOVoo, wovOO, wOVOO

    ints.Woovo, ints.WooVO, ints.WOOvo, ints.WOOVO = woovo, wooVO, wOOvo, wOOVO

    ints.Wovov, ints.WOVOV, ints.WovOV = wovov, wOVOV, wovOV
    
    ints.Fov, ints.FOV = Fov_tr, FOV_tr 

    ints.umat_occ_a = umat_occ_a
    ints.umat_vir_a = umat_vir_a
    ints.umat_occ_b = umat_occ_b
    ints.umat_vir_b = umat_vir_b

    ints.t2aa, ints.t2ab, ints.t2bb = t2_tr_aa, t2_tr_ab, t2_tr_bb
    ints.t1a, ints.t1b  = t1_tr_aa, t1_tr_bb

#   ints.t2aa, ints.t2ab, ints.t2bb = t2aa, t2ab, t2bb
#   ints.t1a, ints.t1b  = t1a, t1b

    return ints


def _make_df_eris(mycc, eris):

    assert mycc._scf.istype('UHF')

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc

    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nvira_pair = nvira * (nvira + 1) // 2
    nvirb_pair = nvirb * (nvirb + 1) // 2
    with_df = mycc.with_df
    naux = eris.naux = with_df.get_naoaux()

    class _ints_3c: pass
    ints_3c = _ints_3c()

    # --- Three-center integrals
    # (L|aa)
    Loo = numpy.empty((naux, nocca, nocca))
    Lov = numpy.empty((naux, nocca, nvira))
    Lvo = numpy.empty((naux, nvira, nocca))
    Lvv = numpy.empty((naux, nvira, nvira))
    # (L|bb)
    LOO = numpy.empty((naux, noccb, noccb))
    LOV = numpy.empty((naux, noccb, nvirb))
    LVO = numpy.empty((naux, nvirb, noccb))
    LVV = numpy.empty((naux, nvirb, nvirb))

    # Transform three-center integrals to MO basis
    p1 = 0
    for eri1 in with_df.loop():
        eri1 = lib.unumpyack_tril(eri1).reshape(-1, nao, nao)
        # (L|aa)
        Lpq = lib.einsum('Lab,ap,bq->Lpq', eri1, moa, moa)
        p0, p1 = p1, p1 + Lpq.shape[0]
        blk = numpy.s_[p0:p1]
        Loo[blk] = Lpq[:, :nocca, :nocca]
        Lov[blk] = Lpq[:, :nocca, nocca:]
        Lvo[blk] = Lpq[:, nocca:, :nocca]
        Lvv[blk] = Lpq[:, nocca:, nocca:]
        # (L|bb)
        Lpq = None
        Lpq = lib.einsum('Lab,ap,bq->Lpq', eri1, mob, mob)
        LOO[blk] = Lpq[:, :noccb, :noccb]
        LOV[blk] = Lpq[:, :noccb, noccb:]
        LVO[blk] = Lpq[:, noccb:, :noccb]
        LVV[blk] = Lpq[:, noccb:, noccb:]
        Lpq = None

    ints_3c.Loo = Loo
    ints_3c.Lov = Lov
    ints_3c.Lvo = Lvo
    ints_3c.Lvv = Lvv

    ints_3c.LOO = LOO
    ints_3c.LOV = LOV
    ints_3c.LVO = LVO
    ints_3c.LVV = LVV

    return ints_3c
