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

import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc.rintermediates import _get_vvvv  # noqa
from pyscf.cc.ccsd import BLKMIN

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III


def make_tau(t2, t1, r1, fac=1, out=None):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1aa = make_tau_aa(t2[0], t1a, r1a, fac, out)
    tau1bb = make_tau_aa(t2[2], t1b, r1b, fac, out)
    tau1ab = make_tau_ab(t2[1], t1, r1, fac, out)
    return tau1aa, tau1ab, tau1bb

def make_tau_aa(t2aa, t1a, r1a, fac=1, out=None):
    tau1aa = np.einsum('ia,jb->ijab', t1a, r1a)
    tau1aa-= np.einsum('ia,jb->jiab', t1a, r1a)
    tau1aa = tau1aa - tau1aa.transpose(0,1,3,2)
    tau1aa *= fac * .5
    tau1aa += t2aa
    return tau1aa

def make_tau_ab(t2ab, t1, r1, fac=1, out=None):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1ab = np.einsum('ia,jb->ijab', t1a, r1b)
    tau1ab+= np.einsum('ia,jb->ijab', r1a, t1b)
    tau1ab *= fac * .5
    tau1ab += t2ab
    return tau1ab

def Foo(t1, t2, eris):

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    mo_ea_o = eris.mo_energy[0][:nocca]
    #mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    #mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift
    
#    fova = eris.focka[:nocca,nocca:]
#    fovb = eris.fockb[:noccb,noccb:]

    fova = eris.focka[np.ix_(act_hole[0], inact_particle[0])]
    fovb = eris.focka[np.ix_(act_hole[1], inact_particle[1])]

    Fooa =  .5 * lib.einsum('me,ie->mi', fova, t1_mix[0][1])
    Foob =  .5 * lib.einsum('me,ie->mi', fovb, t1_mix[1][1])
    
    Fooa += eris.focka[:nocca,:nocca] - np.diag(mo_ea_o)
    Foob += eris.fockb[:noccb,:noccb] - np.diag(mo_eb_o)
   
    eris_ovoo = np.asarray(eris.ovoo)
    eris_ovov = np.asarray(eris.ovov)
    
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    
    Fooa += np.einsum('ne,nemi->mi', t1a, ovoo)
    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    Fooa += .5 * lib.einsum('inef,menf->mi', tilaa, ovov)

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
   
    Foob += np.einsum('ne,nemi->mi', t1b, OVOO)
    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    Foob += .5 * lib.einsum('inef,menf->mi', tilbb, OVOV)

    
    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
    Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)

    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    eris_ovOV = np.asarray(eris.ovOV)
    Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)

    return Fooa, Foob

def Fvv(t1, t2, eris, max_memory, BLKMIN):

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    #mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] #+ cc.level_shift
    #mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] #+ cc.level_shift
    
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    Fvva = -.5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb = -.5 * lib.einsum('me,ma->ae', fovb, t1b)
   
    Fvva += eris.focka[nocca:,nocca:] - np.diag(mo_ea_v)
    Fvvb += eris.fockb[noccb:,noccb:] - np.diag(mo_eb_v)

    #mem_now = lib.current_memory()[0]
    #max_memory = max(0, cc.max_memory - mem_now - t2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)

    if nvirb > 0 and noccb > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)

    if nvirb > 0 and nocca > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)

    if nvira > 0 and noccb > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)

    eris_ovov = np.asarray(eris.ovov)

    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * lib.einsum('mnaf,menf->ae', tilaa, ovov)

    eris_OVOV = np.asarray(eris.OVOV)
    
    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * lib.einsum('MNAF,MENF->AE', tilbb, OVOV)

    eris_ovOV = np.asarray(eris.ovOV)
    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
       
    return Fvva, Fvvb

def Fov(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    eris_ovov = np.asarray(eris.ovov)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)

    Fova = np.einsum('nf,menf->me',t1a, ovov)
    
    eris_OVOV = np.asarray(eris.OVOV)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)

    eris_ovOV = np.asarray(eris.ovOV)
    Fova += np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb += np.einsum('nf,nfME->ME',t1a, eris_ovOV)

    Fova += fova
    Fovb += fovb

    return Fova, Fovb

def Woooo(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    
    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    
    eris_OVOO = np.asarray(eris.OVOO)
    eris_OVOV = np.asarray(eris.OVOV)
    
    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)

    Woooo = lib.einsum('je,nemi->mnij', t1a, eris_ovoo)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5

    WOOOO = lib.einsum('je,nemi->mnij', t1b, eris_OVOO)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5


    WoOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
    WoOoO+= lib.einsum('je,neMI->nMjI', t1a, eris_ovOO)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)

    return Woooo, WoOoO, WOOOO


def Wovvo(t1, t2, eris, max_memory, BLKMIN):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

#    mem_now = lib.current_memory()[0]
#    max_memory = max(0, cc.max_memory - mem_now - u2aa.size*8e-6)
    
    if nvira > 0 and nocca > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            wovvo[p0:p1] += lib.einsum('jf,mebf->mbej', t1a, ovvv)

    if nvirb > 0 and noccb > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)

    if nvirb > 0 and nocca > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)

    if nvira > 0 and noccb > 0:
        blksize = max(BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)

    eris_ovoo = np.asarray(eris.ovoo)
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    wovvo += lib.einsum('nb,nemj->mbej', t1a, ovoo)

    eris_ovov = np.asarray(eris.ovov)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    wovvo -= 0.5*lib.einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*lib.einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = lib.einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= lib.einsum('nb,mnej->mbej', t1a, tmpaa)

    eris_OVOO = np.asarray(eris.OVOO)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    wOVVO += lib.einsum('nb,nemj->mbej', t1b, OVOO)

    eris_OVOV = np.asarray(eris.OVOV)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    wOVVO -= 0.5*lib.einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = lib.einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= lib.einsum('nb,mnej->mbej', t1b, tmpbb)

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
    woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
    woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
    wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)

    eris_ovOV = np.asarray(eris.ovOV)
    wovvo += 0.5*lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = lib.einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = lib.einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= lib.einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= lib.einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += lib.einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += lib.einsum('nb,nMeJ->MbeJ', t1a, tmpabab)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)

    return wovvo, wOVVO, woVvO, woVVo, wOvVo, wOvvO







# NOTE these did not appear

def Wooov(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    eris_ovoo = np.asarray(eris.ovoo)
    eris_OVOO = np.asarray(eris.OVOO)
    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    wooov = np.array(     ovoo.transpose(2,3,0,1), dtype=dtype)
    wOOOV = np.array(     OVOO.transpose(2,3,0,1), dtype=dtype)
    wooOV = np.array(eris_OVoo.transpose(2,3,0,1), dtype=dtype)
    wOOov = np.array(eris_ovOO.transpose(2,3,0,1), dtype=dtype)
    eris_ovoo = eris_OVOO = eris_ovOO = eris_OVoo = None

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)

    wooov += lib.einsum('if,mfne->mine', t1a,      ovov)
    wOOOV += lib.einsum('if,mfne->mine', t1b,      OVOV)
    wooOV += lib.einsum('if,mfNE->miNE', t1a, eris_ovOV)
    wOOov += lib.einsum('IF,neMF->MIne', t1b, eris_ovOV)
    return wooov, wooOV, wOOov, wOOOV

def Woovo(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    wovoo = np.zeros((nocca,nvira,nocca,nocca), dtype=dtype)
    wOVOO = np.zeros((noccb,nvirb,noccb,noccb), dtype=dtype)
    woVoO = np.zeros((nocca,nvirb,nocca,noccb), dtype=dtype)
    wOvOo = np.zeros((noccb,nvira,noccb,nocca), dtype=dtype)

    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
    #:self.wovoo  = 0.5 * lib.einsum('mebf,ijef->mbij', eris_ovvv, tauaa)
    #:self.wovoo -= 0.5 * lib.einsum('mfbe,ijef->mbij', eris_ovvv, tauaa)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    blksize = min(nocca, max(BLKMIN, int(max_memory*1e6/8/(nvira**3*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        wovoo[p0:p1] = 0.5 * lib.einsum('mebf,ijef->mbij', ovvv, tauaa)
        ovvv = None

    #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
    #:self.wOVOO  = 0.5 * lib.einsum('mebf,ijef->mbij', OVVV, taubb)
    blksize = min(noccb, max(BLKMIN, int(max_memory*1e6/8/(nvirb**3*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        wOVOO[p0:p1] = 0.5 * lib.einsum('mebf,ijef->mbij', OVVV, taubb)
        OVVV = None

    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:self.Fvvb += np.einsum('mf,mfAE->AE', t1a, eris_ovVV)
    #:self.woVvO = lib.einsum('JF,meBF->mBeJ', t1b, eris_ovVV)
    #:self.woVVo = lib.einsum('jf,mfBE->mBEj',-t1a, eris_ovVV)
    #:self.woVoO  = 0.5 * lib.einsum('meBF,iJeF->mBiJ', eris_ovVV, tauab)
    #:self.woVoO += 0.5 * lib.einsum('mfBE,iJfE->mBiJ', eris_ovVV, tauab)
    blksize = min(nocca, max(BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
        woVoO[p0:p1] = 0.5 * lib.einsum('meBF,iJeF->mBiJ', ovVV, tauab)
        woVoO[p0:p1]+= 0.5 * lib.einsum('mfBE,iJfE->mBiJ', ovVV, tauab)
        ovVV = None

    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:self.Fvva += np.einsum('MF,MFae->ae', t1b, eris_OVvv)
    #:self.wOvVo = lib.einsum('jf,MEbf->MbEj', t1a, eris_OVvv)
    #:self.wOvvO = lib.einsum('JF,MFbe->MbeJ',-t1b, eris_OVvv)
    #:self.wOvOo  = 0.5 * lib.einsum('MEbf,jIfE->MbIj', eris_OVvv, tauab)
    #:self.wOvOo += 0.5 * lib.einsum('MFbe,jIeF->MbIj', eris_OVvv, tauab)
    blksize = min(noccb, max(BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
        wOvOo[p0:p1] = 0.5 * lib.einsum('MEbf,jIfE->MbIj', OVvv, tauab)
        wOvOo[p0:p1]+= 0.5 * lib.einsum('MFbe,jIeF->MbIj', OVvv, tauab)
        OVvv = None

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    tilaa, tilab, tilbb = make_tau(t2, t1, t1, fac=0.5)

    eris_ovoo = np.asarray(eris.ovoo)
    eris_OVOO = np.asarray(eris.OVOO)
    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    tmpaa = lib.einsum('nemi,jnbe->mbij',      ovoo, t2aa)
    tmpaa+= lib.einsum('NEmi,jNbE->mbij', eris_OVoo, t2ab)
    tmpbb = lib.einsum('nemi,jnbe->mbij',      OVOO, t2bb)
    tmpbb+= lib.einsum('neMI,nJeB->MBIJ', eris_ovOO, t2ab)
    woVoO += lib.einsum('nemi,nJeB->mBiJ',      ovoo, t2ab)
    woVoO += lib.einsum('NEmi,JNBE->mBiJ', eris_OVoo, t2bb)
    woVoO -= lib.einsum('meNI,jNeB->mBjI', eris_ovOO, t2ab)
    wOvOo += lib.einsum('NEMI,jNbE->MbIj',      OVOO, t2ab)
    wOvOo += lib.einsum('neMI,jnbe->MbIj', eris_ovOO, t2aa)
    wOvOo -= lib.einsum('MEni,nJbE->MbJi', eris_OVoo, t2ab)
    wovoo += tmpaa - tmpaa.transpose(0,1,3,2)
    wOVOO += tmpbb - tmpbb.transpose(0,1,3,2)

    tmp1aa = lib.einsum('njbf,menf->mbej', t2aa,      ovov)
    tmp1aa-= lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    tmp1bb = lib.einsum('njbf,menf->mbej', t2bb,      OVOV)
    tmp1bb-= lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    tmp1ab = lib.einsum('NJBF,meNF->mBeJ', t2bb, eris_ovOV)
    tmp1ab-= lib.einsum('nJfB,menf->mBeJ', t2ab,      ovov)
    tmp1ba = lib.einsum('njbf,nfME->MbEj', t2aa, eris_ovOV)
    tmp1ba-= lib.einsum('jNbF,MENF->MbEj', t2ab,      OVOV)
    tmp1abba =-lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    tmp1baab =-lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpaa = lib.einsum('ie,mbej->mbij', t1a, tmp1aa)
    tmpbb = lib.einsum('ie,mbej->mbij', t1b, tmp1bb)
    tmpab = lib.einsum('ie,mBeJ->mBiJ', t1a, tmp1ab)
    tmpab-= lib.einsum('IE,mBEj->mBjI', t1b, tmp1abba)
    tmpba = lib.einsum('IE,MbEj->MbIj', t1b, tmp1ba)
    tmpba-= lib.einsum('ie,MbeJ->MbJi', t1a, tmp1baab)
    wovoo -= tmpaa - tmpaa.transpose(0,1,3,2)
    wOVOO -= tmpbb - tmpbb.transpose(0,1,3,2)
    woVoO -= tmpab
    wOvOo -= tmpba
    eris_ovov = eris_OVOV = eris_ovOV = None
    eris_ovoo = np.asarray(eris.ovoo)
    eris_OVOO = np.asarray(eris.OVOO)
    eris_ovOO = np.asarray(eris.ovOO)
    eris_OVoo = np.asarray(eris.OVoo)
    wovoo += eris_ovoo.transpose(3,1,2,0) - eris_ovoo.transpose(2,1,0,3)
    wOVOO += eris_OVOO.transpose(3,1,2,0) - eris_OVOO.transpose(2,1,0,3)
    woVoO += eris_OVoo.transpose(3,1,2,0)
    wOvOo += eris_ovOO.transpose(3,1,2,0)
    eris_ovoo = eris_OVOO = eris_ovOO = eris_OVoo = None

    eris_ovvo = np.asarray(eris.ovvo)
    eris_OVVO = np.asarray(eris.OVVO)
    eris_OVvo = np.asarray(eris.OVvo)
    eris_ovVO = np.asarray(eris.ovVO)
    eris_oovv = np.asarray(eris.oovv)
    eris_OOVV = np.asarray(eris.OOVV)
    eris_OOvv = np.asarray(eris.OOvv)
    eris_ooVV = np.asarray(eris.ooVV)
    tmpaa = lib.einsum('ie,mebj->mbij', t1a, eris_ovvo)
    tmpbb = lib.einsum('ie,mebj->mbij', t1b, eris_OVVO)
    tmpaa-= lib.einsum('ie,mjbe->mbij', t1a, eris_oovv)
    tmpbb-= lib.einsum('ie,mjbe->mbij', t1b, eris_OOVV)
    woVoO += lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    woVoO -= lib.einsum('IE,mjBE->mBjI',-t1b, eris_ooVV)
    wOvOo += lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    wOvOo -= lib.einsum('ie,MJbe->MbJi',-t1a, eris_OOvv)
    wovoo += tmpaa - tmpaa.transpose(0,1,3,2)
    wOVOO += tmpbb - tmpbb.transpose(0,1,3,2)

    woooo, wooOO, wOOoo, wOOOO = Woooo(t1, t2, eris)
    Fova, Fovb = Fov(t1, t2, eris)
    wovoo -= lib.einsum('me,ijbe->mbij', Fova, t2aa)
    wOVOO -= lib.einsum('me,ijbe->mbij', Fovb, t2bb)
    woVoO += lib.einsum('me,iJeB->mBiJ', Fova, t2ab)
    wOvOo += lib.einsum('ME,jIbE->MbIj', Fovb, t2ab)
    wovoo -= lib.einsum('nb,minj->mbij', t1a, woooo)
    wOVOO -= lib.einsum('nb,minj->mbij', t1b, wOOOO)
    woVoO -= lib.einsum('NB,miNJ->mBiJ', t1b, wooOO)
    wOvOo -= lib.einsum('nb,njMI->MbIj', t1a, wooOO)
    eris_ovvo = eris_OVVO = eris_OVvo = eris_ovVO = None
    eris_oovv = eris_OOVV = eris_OOvv = eris_ooVV = None

    woovo = wovoo.transpose(0,2,1,3)
    wOOVO = wOVOO.transpose(0,2,1,3)
    wooVO = woVoO.transpose(0,2,1,3)
    wOOvo = wOvOo.transpose(0,2,1,3)
    return woovo, wooVO, wOOvo, wOOVO

def Wvvov(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    #:Wamef = einsum('na,nmef->amef', -t1, eris.oovv)
    #:Wamef -= np.asarray(eris.ovvv).transpose(1,0,2,3)
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    Waemf = lib.einsum('na,nemf->aemf',-t1a, eris_ovov)
    Waemf+= np.asarray(eris.get_ovvv()).transpose(2,3,0,1)
    Waemf = Waemf - Waemf.transpose(0,3,2,1)

    WaeMF = lib.einsum('na,nemf->aemf',-t1a, eris_ovOV)
    WaeMF+= np.asarray(eris.get_OVvv()).transpose(2,3,0,1)

    WAEmf = lib.einsum('na,mfne->aemf',-t1b, eris_ovOV)
    WAEmf+= np.asarray(eris.get_ovVV()).transpose(2,3,0,1)

    WAEMF = lib.einsum('na,nemf->aemf',-t1b, eris_OVOV)
    WAEMF+= np.asarray(eris.get_OVVV()).transpose(2,3,0,1)
    WAEMF = WAEMF - WAEMF.transpose(0,3,2,1)
    return Waemf, WaeMF, WAEmf, WAEMF

def Wvvvo(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    _wvovv = np.empty((nvira,nocca,nvira,nvira), dtype=dtype)
    _wVOVV = np.empty((nvirb,noccb,nvirb,nvirb), dtype=dtype)
    _wvOvV = np.empty((nvira,noccb,nvira,nvirb), dtype=dtype)
    _wVoVv = np.empty((nvirb,nocca,nvirb,nvira), dtype=dtype)

    Fova, Fovb = Fov(t1, t2, eris)

    # 3 or 4 virtuals
    eris_ovoo = np.asarray(eris.ovoo)
    eris_ovov = np.asarray(eris.ovov)
    eris_ovOV = np.asarray(eris.ovOV)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    eris_oovv = eris_ovvo = None
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    #:wvovv  = .5 * lib.einsum('meni,mnab->eiab', eris_ovoo, tauaa)
    #:wvovv -= .5 * lib.einsum('me,miab->eiab', self.Fova, t2aa)
    #:tmp1aa = lib.einsum('nibf,menf->mbei', t2aa,      ovov)
    #:tmp1aa-= lib.einsum('iNbF,meNF->mbei', t2ab, eris_ovOV)
    #:wvovv+= lib.einsum('ma,mbei->eiab', t1a, tmp1aa)
    #:wvovv+= lib.einsum('ma,mibe->eiab', t1a,      oovv)
    for p0, p1 in lib.prange(0, nvira, nocca):
        wvovv  = .5*lib.einsum('meni,mnab->eiab', eris_ovoo[:,p0:p1], tauaa)
        wvovv -= .5*lib.einsum('me,miab->eiab', Fova[:,p0:p1], t2aa)

        tmp1aa = lib.einsum('nibf,menf->mbei', t2aa, ovov[:,p0:p1])
        tmp1aa-= lib.einsum('iNbF,meNF->mbei', t2ab, eris_ovOV[:,p0:p1])
        wvovv += lib.einsum('ma,mbei->eiab', t1a, tmp1aa)
        wvovv += lib.einsum('ma,mibe->eiab', t1a, oovv[:,:,:,p0:p1])
        _wvovv[p0:p1] = wvovv
        tmp1aa = None
    eris_ovov = eris_ovoo = eris_ovOV = None

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
    #:wvovv += lib.einsum('mebf,miaf->eiab',      ovvv, t2aa)
    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:wvovv += lib.einsum('MFbe,iMaF->eiab', eris_OVvv, t2ab)
    #:wvovv += eris_ovvv.transpose(2,0,3,1).conj()
    #:self.wvovv -= wvovv - wvovv.transpose(0,1,3,2)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    blksize = min(nocca, max(BLKMIN, int(max_memory*1e6/8/(nvira**3*6))))
    for i0,i1 in lib.prange(0, nocca, blksize):
        wvovv = _wvovv[:,i0:i1]
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            wvovv -= lib.einsum('MFbe,iMaF->eiab', OVvv, t2ab[i0:i1,p0:p1])
            OVvv = None
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            if p0 == i0:
                wvovv += ovvv.transpose(2,0,3,1).conj()
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            wvovv -= lib.einsum('mebf,miaf->eiab', ovvv, t2aa[p0:p1,i0:i1])
            ovvv = None
        wvovv = wvovv - wvovv.transpose(0,1,3,2)
        _wvovv[:,i0:i1] = wvovv

    eris_OVOO = np.asarray(eris.OVOO)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    eris_OOVV = eris_OVVO = None
    #:wVOVV  = .5*lib.einsum('meni,mnab->eiab', eris_OVOO, taubb)
    #:wVOVV -= .5*lib.einsum('me,miab->eiab', self.Fovb, t2bb)
    #:tmp1bb = lib.einsum('nibf,menf->mbei', t2bb,      OVOV)
    #:tmp1bb-= lib.einsum('nIfB,nfME->MBEI', t2ab, eris_ovOV)
    #:wVOVV += lib.einsum('ma,mbei->eiab', t1b, tmp1bb)
    #:wVOVV += lib.einsum('ma,mibe->eiab', t1b,      OOVV)
    for p0, p1 in lib.prange(0, nvirb, noccb):
        wVOVV  = .5*lib.einsum('meni,mnab->eiab', eris_OVOO[:,p0:p1], taubb)
        wVOVV -= .5*lib.einsum('me,miab->eiab', Fovb[:,p0:p1], t2bb)

        tmp1bb = lib.einsum('nibf,menf->mbei', t2bb, OVOV[:,p0:p1])
        tmp1bb-= lib.einsum('nIfB,nfME->MBEI', t2ab, eris_ovOV[:,:,:,p0:p1])
        wVOVV += lib.einsum('ma,mbei->eiab', t1b, tmp1bb)
        wVOVV += lib.einsum('ma,mibe->eiab', t1b, OOVV[:,:,:,p0:p1])
        _wVOVV[p0:p1] = wVOVV
        tmp1bb = None
    eris_OVOV = eris_OVOO = eris_ovOV = None

    #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
    #:wVOVV -= lib.einsum('MEBF,MIAF->EIAB',      OVVV, t2bb)
    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:wVOVV -= lib.einsum('mfBE,mIfA->EIAB', eris_ovVV, t2ab)
    #:wVOVV += eris_OVVV.transpose(2,0,3,1).conj()
    #:self.wVOVV += wVOVV - wVOVV.transpose(0,1,3,2)
    blksize = min(noccb, max(BLKMIN, int(max_memory*1e6/8/(nvirb**3*6))))
    for i0,i1 in lib.prange(0, noccb, blksize):
        wVOVV = _wVOVV[:,i0:i1]
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            wVOVV -= lib.einsum('mfBE,mIfA->EIAB', ovVV, t2ab[p0:p1,i0:i1])
            ovVV = None
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            if p0 == i0:
                wVOVV += OVVV.transpose(2,0,3,1).conj()
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            wVOVV -= lib.einsum('mebf,miaf->eiab', OVVV, t2bb[p0:p1,i0:i1])
            OVVV = None
        wVOVV = wVOVV - wVOVV.transpose(0,1,3,2)
        _wVOVV[:,i0:i1] = wVOVV

    eris_ovOV = np.asarray(eris.ovOV)
    eris_ovOO = np.asarray(eris.ovOO)
    eris_OOvv = np.asarray(eris.OOvv)
    eris_ovVO = np.asarray(eris.ovVO)
    #:self.wvOvV = lib.einsum('meNI,mNaB->eIaB', eris_ovOO, tauab)
    #:self.wvOvV -= lib.einsum('me,mIaB->eIaB', self.Fova, t2ab)
    #:tmp1ab = lib.einsum('NIBF,meNF->mBeI', t2bb, eris_ovOV)
    #:tmp1ab-= lib.einsum('nIfB,menf->mBeI', t2ab,      ovov)
    #:tmp1baab = lib.einsum('nIbF,neMF->MbeI', t2ab, eris_ovOV)
    #:tmpab = lib.einsum('ma,mBeI->eIaB', t1a, tmp1ab)
    #:tmpab+= lib.einsum('MA,MbeI->eIbA', t1b, tmp1baab)
    #:tmpab-= lib.einsum('MA,MIbe->eIbA', t1b, eris_OOvv)
    #:tmpab-= lib.einsum('ma,meBI->eIaB', t1a, eris_ovVO)
    #:self.wvOvV += tmpab
    for p0, p1 in lib.prange(0, nvira, nocca):
        wvOvV  = lib.einsum('meNI,mNaB->eIaB', eris_ovOO[:,p0:p1], tauab)
        wvOvV -= lib.einsum('me,mIaB->eIaB', Fova[:,p0:p1], t2ab)
        tmp1ab = lib.einsum('NIBF,meNF->mBeI', t2bb, eris_ovOV[:,p0:p1])
        tmp1ab-= lib.einsum('nIfB,menf->mBeI', t2ab, ovov[:,p0:p1])
        wvOvV+= lib.einsum('ma,mBeI->eIaB', t1a, tmp1ab)
        tmp1ab = None
        tmp1baab = lib.einsum('nIbF,neMF->MbeI', t2ab, eris_ovOV[:,p0:p1])
        wvOvV+= lib.einsum('MA,MbeI->eIbA', t1b, tmp1baab)
        tmp1baab = None
        wvOvV-= lib.einsum('MA,MIbe->eIbA', t1b, eris_OOvv[:,:,:,p0:p1])
        wvOvV-= lib.einsum('ma,meBI->eIaB', t1a, eris_ovVO[:,p0:p1])
        _wvOvV[p0:p1] = wvOvV
    eris_ovOV = eris_ovOO = eris_OOvv = eris_ovVO = None

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
    #:self.wvOvV -= lib.einsum('mebf,mIfA->eIbA',      ovvv, t2ab)
    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:self.wvOvV -= lib.einsum('meBF,mIaF->eIaB', eris_ovVV, t2ab)
    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:self.wvOvV -= lib.einsum('MFbe,MIAF->eIbA', eris_OVvv, t2bb)
    #:self.wvOvV += eris_OVvv.transpose(2,0,3,1).conj()
    blksize = min(noccb, max(BLKMIN, int(max_memory*1e6/8/(nvira**3*6))))
    for i0,i1 in lib.prange(0, noccb, blksize):
        wvOvV = _wvOvV[:,i0:i1]
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            wvOvV -= lib.einsum('meBF,mIaF->eIaB', ovVV, t2ab[p0:p1,i0:i1])
            ovVV = None
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            wvOvV -= lib.einsum('mebf,mIfA->eIbA',ovvv, t2ab[p0:p1,i0:i1])
            ovvv = None
        _wvOvV[:,i0:i1] = wvOvV

    blksize = min(noccb, max(BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
    for i0,i1 in lib.prange(0, noccb, blksize):
        wvOvV = _wvOvV[:,i0:i1]
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            if p0 == i0:
                wvOvV += OVvv.transpose(2,0,3,1).conj()
            wvOvV -= lib.einsum('MFbe,MIAF->eIbA', OVvv, t2bb[p0:p1,i0:i1])
            OVvv = None
        _wvOvV[:,i0:i1] = wvOvV

    eris_ovOV = np.asarray(eris.ovOV)
    eris_OVoo = np.asarray(eris.OVoo)
    eris_ooVV = np.asarray(eris.ooVV)
    eris_OVvo = np.asarray(eris.OVvo)
    #:self.wVoVv = lib.einsum('MEni,nMbA->EiAb', eris_OVoo, tauab)
    #:self.wVoVv -= lib.einsum('ME,iMbA->EiAb', self.Fovb, t2ab)
    #:tmp1ba = lib.einsum('nibf,nfME->MbEi', t2aa, eris_ovOV)
    #:tmp1ba-= lib.einsum('iNbF,MENF->MbEi', t2ab,      OVOV)
    #:tmp1abba = lib.einsum('iNfB,mfNE->mBEi', t2ab, eris_ovOV)
    #:tmpba = lib.einsum('MA,MbEi->EiAb', t1b, tmp1ba)
    #:tmpba+= lib.einsum('ma,mBEi->EiBa', t1a, tmp1abba)
    #:tmpba-= lib.einsum('ma,miBE->EiBa', t1a, eris_ooVV)
    #:tmpba-= lib.einsum('MA,MEbi->EiAb', t1b, eris_OVvo)
    #:self.wVoVv += tmpba
    for p0, p1 in lib.prange(0, nvirb, noccb):
        wVoVv  = lib.einsum('MEni,nMbA->EiAb', eris_OVoo[:,p0:p1], tauab)
        wVoVv -= lib.einsum('ME,iMbA->EiAb', Fovb[:,p0:p1], t2ab)
        tmp1ba = lib.einsum('nibf,nfME->MbEi', t2aa, eris_ovOV[:,:,:,p0:p1])
        tmp1ba-= lib.einsum('iNbF,MENF->MbEi', t2ab, OVOV[:,p0:p1])
        wVoVv += lib.einsum('MA,MbEi->EiAb', t1b, tmp1ba)
        tmp1ba = None
        tmp1abba = lib.einsum('iNfB,mfNE->mBEi', t2ab, eris_ovOV[:,:,:,p0:p1])
        wVoVv += lib.einsum('ma,mBEi->EiBa', t1a, tmp1abba)
        tmp1abba = None
        wVoVv -= lib.einsum('ma,miBE->EiBa', t1a, eris_ooVV[:,:,:,p0:p1])
        wVoVv -= lib.einsum('MA,MEbi->EiAb', t1b, eris_OVvo[:,p0:p1])
        _wVoVv[p0:p1] = wVoVv
    eris_ovOV = eris_OVoo = eris_ooVV = eris_OVvo = None

    #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
    #:self.wVoVv -= lib.einsum('MEBF,iMaF->EiBa',      OVVV, t2ab)
    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:self.wVoVv -= lib.einsum('MEbf,iMfA->EiAb', eris_OVvv, t2ab)
    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:self.wVoVv -= lib.einsum('mfBE,miaf->EiBa', eris_ovVV, t2aa)
    #:self.wVoVv += eris_ovVV.transpose(2,0,3,1).conj()
    blksize = min(noccb, max(BLKMIN, int(max_memory*1e6/8/(nvirb**3*6))))
    for i0,i1 in lib.prange(0, nocca, blksize):
        wVoVv = _wVoVv[:,i0:i1]
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            wVoVv -= lib.einsum('MEbf,iMfA->EiAb', OVvv, t2ab[i0:i1,p0:p1])
            OVvv = None
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            wVoVv -= lib.einsum('MEBF,iMaF->EiBa', OVVV, t2ab[i0:i1,p0:p1])
            OVVV = None
        _wVoVv[:,i0:i1] = wVoVv

    blksize = min(nocca, max(BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
    for i0,i1 in lib.prange(0, nocca, blksize):
        wVoVv = _wVoVv[:,i0:i1]
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            if p0 == i0:
                wVoVv += ovVV.transpose(2,0,3,1).conj()
            wVoVv -= lib.einsum('mfBE,miaf->EiBa', ovVV, t2aa[p0:p1,i0:i1])
            ovVV = None
        _wVoVv[:,i0:i1] = wVoVv
    wvvvo = _wvovv.transpose(2,0,3,1)
    wVVvo = _wVoVv.transpose(2,0,3,1)
    wvvVO = _wvOvV.transpose(2,0,3,1)
    wVVVO = _wVOVV.transpose(2,0,3,1)
    return wvvvo, wvvVO, wVVvo, wVVVO


def _get_vvVV(eris):
    if eris.vvVV is None and getattr(eris, 'VVL', None) is not None:  # DF eris
        vvL = np.asarray(eris.vvL)
        VVL = np.asarray(eris.VVL)
        vvVV = lib.dot(vvL, VVL.T)
    elif len(eris.vvVV.shape) == 2:
        vvVV = np.asarray(eris.vvVV)
    else:
        return eris.vvVV

    nvira = int(np.sqrt(vvVV.shape[0]*2))
    nvirb = int(np.sqrt(vvVV.shape[1]*2))
    vvVV1 = np.zeros((nvira**2,nvirb**2))
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    lib.takebak_2d(vvVV1, vvVV, vtrila[0]*nvira+vtrila[1], vtrilb[0]*nvirb+vtrilb[1])
    lib.takebak_2d(vvVV1, vvVV, vtrila[1]*nvira+vtrila[0], vtrilb[1]*nvirb+vtrilb[0])
    lib.takebak_2d(vvVV1, vvVV, vtrila[0]*nvira+vtrila[1], vtrilb[1]*nvirb+vtrilb[0])
    lib.takebak_2d(vvVV1, vvVV, vtrila[1]*nvira+vtrila[0], vtrilb[0]*nvirb+vtrilb[1])
    return vvVV1.reshape(nvira,nvira,nvirb,nvirb)

def _get_VVVV(eris):
    if eris.VVVV is None and getattr(eris, 'VVL', None) is not None:  # DF eris
        VVL = np.asarray(eris.VVL)
        nvir = int(np.sqrt(eris.VVL.shape[0]*2))
        return ao2mo.restore(1, lib.dot(VVL, VVL.T), nvir)
    elif len(eris.VVVV.shape) == 2:
        nvir = int(np.sqrt(eris.VVVV.shape[0]*2))
        return ao2mo.restore(1, np.asarray(eris.VVVV), nvir)
    else:
        return eris.VVVV

########################
