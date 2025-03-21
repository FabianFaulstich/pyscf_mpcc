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
from pyscf import lib
from pyscf.cc import uccsd
from pyscf.cc import uintermediates

'''
UCCSD(T)
'''

def kernel(mcc, eris, t1=None, t2=None):
    if t1 is None or t2 is None:
        t1, t2 = mcc.t1, mcc.t2

    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:,:nocca]
    fVO = eris.fockb[noccb:,:noccb]

    imds = make_intermediates(mcc, t1, t2, eris)

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,kceb->ijkabc', t2aa, numpy.asarray(eris.get_ovvv()).conj())
    w-= numpy.einsum('mkbc,iajm->ijkabc', t2aa, numpy.asarray(eris.ovoo).conj())
    r = r6(w)
    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.ovov).conj(), t1a)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5
    wvd = p6(w + v) / d3
    et = numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
#    w = numpy.einsum('ijae,kceb->ijkabc', t2bb, numpy.asarray(eris.get_OVVV()).conj())

    w = numpy.einsum('ijae,ebkc->ijkabc', t2bb, numpy.asarray(imds.wvvov_act))

    w-= numpy.einsum('imab,kcjm->ijkabc', t2bb, numpy.asarray(eris.OVOO).conj())
    r = r6(w)
    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.OVOV).conj(), t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
    wvd = p6(w + v) / d3
    et += numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

    # baa
#    w  = numpy.einsum('jIeA,kceb->IjkAbc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2

    w  = numpy.einsum('jIeA,ebkc->IjkAbc', t2ab, numpy.asarray(imds.wvvov_act)) * 2

#    w += numpy.einsum('jIbE,kcEA->IjkAbc', t2ab, numpy.asarray(eris.get_ovVV()).conj()) * 2

    w += numpy.einsum('jIbE,EAkc->IjkAbc', t2ab, numpy.asarray(imds.wVVov_act)) * 2

#    w += numpy.einsum('jkbe,IAec->IjkAbc', t2aa, numpy.asarray(eris.get_OVvv()).conj())
    w += numpy.einsum('jkbe,ecIA->IjkAbc', t2aa, numpy.asarray(imds.wvvOV_act))

#    w -= numpy.einsum('mIbA,kcjm->IjkAbc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
#    w -= numpy.einsum('jMbA,kcIM->IjkAbc', t2ab, numpy.asarray(eris.ovOO).conj()) * 2
#    w -= numpy.einsum('jmbc,IAkm->IjkAbc', t2aa, numpy.asarray(eris.OVoo).conj())


    w -= numpy.einsum('ImAb,mjck->IjkAbc', t2ab, numpy.asarray(imds.oovo)) * 2
    w -= numpy.einsum('MjAb,MIck->IjkAbc', t2ab, numpy.asarray(eris.OOvo)) * 2
    w -= numpy.einsum('mjcb,mkAI->IjkAbc', t2aa, numpy.asarray(eris.ooVO))


    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('jbkc,IA->IjkAbc', numpy.asarray(eris.ovov).conj(), t1b)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += numpy.einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    r /= d3
    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)

    # bba
#   w  = numpy.einsum('ijae,kceb->ijkabc', t2ab, numpy.asarray(eris.get_OVVV()).conj()) * 2
    w  = numpy.einsum('ijae,ebkc->ijkabc', t2ab, numpy.asarray(imds.Wvvov) * 2

#   w += numpy.einsum('ijeb,kcea->ijkabc', t2ab, numpy.asarray(eris.get_OVvv()).conj()) * 2
    w += numpy.einsum('ijeb,eakc->ijkabc', t2ab, numpy.asarray(imds.Wvvov) * 2


#   w += numpy.einsum('jkbe,iaec->ijkabc', t2bb, numpy.asarray(eris.get_ovVV()).conj())
    w += numpy.einsum('jkbe,ecia->ijkabc', t2bb, numpy.asarray(imds.Wvvov)


#    w -= numpy.einsum('imab,kcjm->ijkabc', t2ab, numpy.asarray(eris.OVOO).conj()) * 2
#    w -= numpy.einsum('mjab,kcim->ijkabc', t2ab, numpy.asarray(eris.OVoo).conj()) * 2
#    w -= numpy.einsum('jmbc,iakm->ijkabc', t2bb, numpy.asarray(eris.ovOO).conj())


    w -= numpy.einsum('imab,mjck->ijkabc', t2ab, numpy.asarray(eris.OVOO).conj()) * 2
    w -= numpy.einsum('mjab,mick->ijkabc', t2ab, numpy.asarray(eris.OVoo).conj()) * 2
    w -= numpy.einsum('jmbc,mkai->ijkabc', t2bb, numpy.asarray(eris.ovOO).conj())


    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.OVOV).conj(), t1a)
    v += numpy.einsum('iakc,jb->ijkabc', numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum('iakc,jb->ijkabc', numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum('JKBC,ai->iJKaBC', t2bb, fvo) * .5
    v += numpy.einsum('iKaC,BJ->iJKaBC', t2ab, fVO) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    r /= d3
    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)

    et *= .25
    return et




def update_amps(mcc, t1, t2, eris):
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


    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]


    imds = make_intermediates(mcc, t1, t2, eris)

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,ebkc->ijkabc', t2aa[np.ix_(act_hole[0], act_hole[0], act_particle[0], inact_particle[0])], numpy.asarray(imds.Wvvov_act))
    w-= numpy.einsum('mkbc,iajm->ijkabc', t2aa[np.ix_(inact_hole[0], act_hole[0], act_particle[0], act_particle[0])], numpy.asarray(imds.Woovo_act))

    w += lib.einsum('ijkabe,ce->ijkabc', t3aaa, fvva)
    w -= lib.einsum('mjkabc,mi->ijkabc', t3aaa, fooa)

    u3aaa = r6(w)


    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
#    w = numpy.einsum('ijae,kceb->ijkabc', t2bb, numpy.asarray(eris.get_OVVV()).conj())

    w = numpy.einsum('ijae,ebkc->ijkabc', t2bb[np.ix_(act_hole[1], act_hole[1], act_particle[1], inact_particle[1])], numpy.asarray(imds.Wvvov_act))

#   w-= numpy.einsum('imab,kcjm->ijkabc', t2bb, numpy.asarray(eris.OVOO).conj())
    w-= numpy.einsum('imab,mjck->ijkabc', t2bb[np.ix_(inact_hole[0], act_hole[0], act_particle[0], act_particle[0])], numpy.asarray(imds.WOOVO_act))

    w  = lib.einsum('ijkabe,ce->ijkabc', t3bbb, fvvb)
    w -= lib.einsum('mjkabc,mi->ijkabc', t3bbb, foob)

    u3bbb = r6(w)

    # baa
#   w  = numpy.einsum('jIeA,kceb->IjkAbc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
    w  = numpy.einsum('jIeA,ebkc->IjkAbc', t2ab[np.ix_(act_hole[0], act_hole[1], act_particle[0], inact_particle[1])], numpy.asarray(imds.Wvvov_act)) * 2

#    w += numpy.einsum('jIbE,kcEA->IjkAbc', t2ab, numpy.asarray(eris.get_ovVV()).conj()) * 2

    w += numpy.einsum('jIbE,EAkc->IjkAbc', t2ab[np.ix_(act_hole[0], act_hole[1], act_particle[0], inact_particle[1])], numpy.asarray(imds.WVVov_act)) * 2

#    w += numpy.einsum('jkbe,IAec->IjkAbc', t2aa, numpy.asarray(eris.get_OVvv()).conj())
    w += numpy.einsum('jkbe,ecIA->IjkAbc', t2aa[np.ix_(act_hole[0], act_hole[0], act_particle[0], inact_particle[0])], numpy.asarray(imds.WvvOV_act))

#    w -= numpy.einsum('mIbA,kcjm->IjkAbc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
#    w -= numpy.einsum('jMbA,kcIM->IjkAbc', t2ab, numpy.asarray(eris.ovOO).conj()) * 2
#    w -= numpy.einsum('jmbc,IAkm->IjkAbc', t2aa, numpy.asarray(eris.OVoo).conj())


    w -= numpy.einsum('ImAb,mjck->IjkAbc', t2ab[np.ix_(act_hole[1], inact_hole[0], act_particle[1], act_particle[0])], numpy.asarray(imds.Woovo_act)) * 2
    w -= numpy.einsum('MjAb,MIck->IjkAbc', t2ab[np.ix_(inact_hole[1], act_hole[0], act_particle[1], act_particle[0])], numpy.asarray(imds.WOOvo_act)) * 2
    w -= numpy.einsum('mjcb,mkAI->IjkAbc', t2aa[np.ix_(inact_hole[0], act_hole[0], act_particle[0], act_particle[0])], numpy.asarray(imds.WooVO_act))

    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)

    # bba
#   w  = numpy.einsum('ijae,kceb->ijkabc', t2ab, numpy.asarray(eris.get_OVVV()).conj()) * 2
    w  = numpy.einsum('ijae,ebkc->ijkabc', t2ab[np.ix_(act_hole[0], act_hole[1], act_particle[0], inact_particle[1])], numpy.asarray(imds.WVVOV_act) * 2

#   w += numpy.einsum('ijeb,kcea->ijkabc', t2ab, numpy.asarray(eris.get_OVvv()).conj()) * 2
    w += numpy.einsum('ijeb,eakc->ijkabc', t2ab[np.ix_(act_hole[0], act_hole[1], inact_particle[0], act_particle[1])], numpy.asarray(imds.WvvOV_act) * 2

#   w += numpy.einsum('jkbe,iaec->ijkabc', t2bb, numpy.asarray(eris.get_ovVV()).conj())
    w += numpy.einsum('jkbe,ecia->ijkabc', t2bb[np.ix_(act_hole[1], act_hole[1], act_particle[1], inact_particle[1])], numpy.asarray(imds.WVVov_act)

#    w -= numpy.einsum('imab,kcjm->ijkabc', t2ab, numpy.asarray(eris.OVOO).conj()) * 2
#    w -= numpy.einsum('mjab,kcim->ijkabc', t2ab, numpy.asarray(eris.OVoo).conj()) * 2
#    w -= numpy.einsum('jmbc,iakm->ijkabc', t2bb, numpy.asarray(eris.ovOO).conj())


    w -= numpy.einsum('imab,mjck->ijkabc', t2ab[np.ix_(inact_hole[0], act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.WOOVO_act)) * 2
    w -= numpy.einsum('mjab,mick->ijkabc', t2ab[np.ix_(inact_hole[0], act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.WooVO_act)) * 2
    w -= numpy.einsum('jmbc,mkai->ijkabc', t2bb[np.ix_(inact_hole[1], act_hole[1], act_particle[1], act_particle[1])], numpy.asarray(eris.WOOvo_act))


    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)

#    u2ab   = lib.einsum('iJaE,BE->iJaB', t2ab, fvvb)
#    u2ab  += lib.einsum('iJeA,be->iJbA', t2ab, fvva)

    u3baa  = lib.einsum('IjkEbc,AE->IjkAbc', t3baa, fvvb)  
    u3baa += lib.einsum('IjkAec,be->IjkAbc', t3baa, fvva)  

    u3bba = lib.einsum('IJkEBc,AE->IJkABc', t3bba, fvvb)  
    u3bba += lib.einsum('IJkABe,ce->IJkABc',t3bba, fvva)  



    u3baa -= lib.einsum('MjkAbc,MI->IjkAbc', t3baa, foob)  
    u3baa -= lib.einsum('ImkAbc,mj->IjkAbc', t3baa, fooa)  


    u3bba = lib.einsum('MJkABc,MI->IJkABc', t3bba, foob)  
    u3bba += lib.einsum('IJmABc,mk->IJkABc',t3bba, fooa)  

#Now add symmetrization of the u3 tensors:
 
    x = r6(u3aaa)     
    y = r6(u3bbb)     

    r = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)

    v = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v = v + v.transpose(1,0,2,4,3,5)

# divide by denominator..




    return u3 


def _iterative_kernel(mcc, w, t1 = None, t2 = None, verbose=None):
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mcc, verbose)

    if t1 is not None and t2 is not None:
        t1 = t1
        t2 = t2
        emp2 = 0
    else:
        emp2, t2 = mp.init_amps(eris=eris)
        t1 = mp.get_t1(eris, t2)

    log.info('Init E(MP2) = %.15g', emp2)

    adiis = lib.diis.DIIS(mcc)

    conv = False
    for istep in range(mcc.max_cycle):
       
        # t1new, t2new = mp.update_amps_oomp2(t1, t2, eris) 
        error = mcc.update_amps(t1, t2, t3, eris)

        if isinstance(error, numpy.ndarray):
            t3new = error + t3
            normt = numpy.linalg.norm(error)
            t3 = None
            t3new = adiis.update(t3new)
        else: # UMP2
            normt = numpy.linalg.norm([numpy.linalg.norm(error[i])
                                       for i in range(4)])
            t3 = None
            t3shape = [x.shape for x in t3new]
            t3new = numpy.hstack([x.ravel() for x in t3new])
            t3new = adiis.update(t3new)
            t3new = lib.split_reshape(t3new, t3shape)

        t3, t3new = t3new, None
        ecct, e_last = mcc.energy(t2, eris, t1), ecct
        log.info('cycle = %d  E_corr(MP2) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep+1, ecct, ecct - e_last, normt)
        cput1 = log.timer('MP2 iter', *cput1)
        if abs(ecct-e_last) < mcc.conv_tol and normt < mcc.conv_tol_normt:
            conv = True
            break
    log.timer('MP2', *cput0)
    return conv, ecct, t2, t1


def make_intermediates():
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(self.stdout, self.verbose)

    t1, t2, eris = self.t1, self.t2, self.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    inact_particle_a = np.delete(np.arange(nvira), act_particle[0])
    inact_particle_b = np.delete(np.arange(nvirb), act_particle[1])

    inact_hole_a = np.delete(np.arange(nocca), act_hole[0])
    inact_hole_b = np.delete(np.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)


    class _IMDS: pass
    imds = _IMDS()

# Foo, Fvv and Fov
    Foo, FOO = uintermediates.Foo(t1, t2, eris)
    Fvv, FVV = uintermediates.Fvv(t1, t2, eris)
    Fov, FOV = uintermediates.Fov(t1, t2, eris)

# Woovo 
    Woovo, WooVO, WOOvo, WOOVO = uintermediates.Woovo(t1, t2, eris)
#Wvvov
#    Wvvov, WvvOV, WVVov, WVVOV = uintermediates.Wvvov(t1, t2, eris) // this will be required for the contraction with lambda

    wvvvo, wvvVO, wVVvo, wVVVO = uintermediates.Wvvvo(t1, t2, eris)

    imds.Foo_act = Foo[np.ix_(act_hole[0], act_hole[0])]
    imds.FOO_act = FOO[np.ix_(act_hole[1], act_hole[1])]
    imds.FOV_act = FOV[np.ix_(act_hole[1], act_particle[1])]
    imds.Fov_act = Fov[np.ix_(act_hole[0], act_particle[0])]
    imds.Fvv_act = Fvv[np.ix_(act_particle[0], act_particle[0])]
    imds.FVV_act = FVV[np.ix_(act_particle[1], act_particle[1])]


    imds.Woovo_act = Woovo[np.ix_(inact_hole[0], act_hole[0], act_particle[0], act_hole[0])]
    imds.WooVO_act = WooVO[np.ix_(inact_hole[0], act_hole[0], act_particle[1], act_hole[1])]
    imds.WOOvo_act = WOOvo[np.ix_(inact_hole[1], act_hole[1], act_particle[0], act_hole[0])]
    imds.WOOVO_act = WOOVO[np.ix_(inact_hole[1], act_hole[1], act_particle[1], act_hole[1])]

    imds.Wvvvo_act = Wvvvo[np.ix_(inact_particle[0], act_particle[0], act_particle[0], act_hole[0])]
    imds.WvvVO_act = WvvVO[np.ix_(inact_particle[0], act_particle[0], act_particle[1], act_hole[1])]
    imds.WVVvo_act = WVVvo[np.ix_(inact_particle[1], act_particle[1], act_particle[0], act_hole[0])]
    imds.WVVVO_act = WVVVO[np.ix_(inact_particle[1], act_particle[1], act_particle[1], act_hole[1])]


    return imds 



def make_intermediates_energy():


    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(self.stdout, self.verbose)

    t1, t2, eris = self.t1, self.t2, self.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    inact_particle_a = np.delete(np.arange(nvira), act_particle[0])
    inact_particle_b = np.delete(np.arange(nvirb), act_particle[1])

    inact_hole_a = np.delete(np.arange(nocca), act_hole[0])
    inact_hole_b = np.delete(np.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    class _IMDS: pass
    imds = _IMDS()

    wvvov, wvvOV, wVVov, wVVOV = uintermediates.Wvvov(t1, t2, eris)


#Now sort the intermediates based on the active and inactive indices..


    imds.Wvvov_act = Wvvov[np.ix_(act_particle[0], act_particle[0], act_hole[0], act_particle[0])]
    imds.WvvOV_act = WvvOV[np.ix_(act_particle[0], act_particle[0], act_hole[1], act_particle[1])]
    imds.WVVov_act = WVVov[np.ix_(act_particle[1], act_particle[1], act_hole[0], act_particle[0])]
    imds.WVVOV_act = WVVOV[np.ix_(act_particle[1], act_particle[1], act_hole[1], act_particle[1])]


    return imds


def lhs_umpcc_t(mcc, t1, t2, eris) 
   '''    
     t1, t2 amplitudes will be used to build the lhs. later we will replace them by L1 and L2 amplitudes..
   ''' 

#aaa

    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.ovov).conj(), t1a)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5
    wvd = p6(w + v) / d3
    et = numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

#bbb

    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.OVOV).conj(), t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
    wvd = p6(w + v) / d3
    et += numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)
 

#baa

    v  = numpy.einsum('jbkc,IA->IjkAbc', numpy.asarray(eris.ovov).conj(), t1b)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += numpy.einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    r /= d3
    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)
 
#bba











    return w_lhs



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    t1a = t1b = mcc.t1
    t2ab = mcc.t2
    t2aa = t2bb = t2ab - t2ab.transpose(1,0,2,3)
    mcc = uccsd.UCCSD(scf.addons.convert_to_uhf(rhf))
    e3a = kernel(mcc, mcc.ao2mo(), (t1a,t1b), (t2aa,t2ab,t2bb))
    print(e3a - -0.00099642337843278096)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.spin = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    nao, nmo = mf.mo_coeff[0].shape
    numpy.random.seed(10)
    mf.mo_coeff = numpy.random.random((2,nao,nmo))

    numpy.random.seed(12)
    nocca, noccb = mol.nelec
    nmo = mf.mo_occ[0].size
    nvira = nmo - nocca
    nvirb = nmo - noccb
    t1a  = .1 * numpy.random.random((nocca,nvira))
    t1b  = .1 * numpy.random.random((noccb,nvirb))
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    mcc = uccsd.UCCSD(scf.addons.convert_to_uhf(mf))
    e3a = kernel(mcc, mcc.ao2mo(), [t1a,t1b], [t2aa, t2ab, t2bb])
    print(e3a - 9877.2780859693339)

    mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf))
    eris = mycc.ao2mo()
    t1 = mycc.spatial2spin(t1, eris.orbspin)
    t2 = mycc.spatial2spin(t2, eris.orbspin)
    from pyscf.cc import gccsd_t
    et = gccsd_t.kernel(mycc, eris, t1, t2)
    print(et - 9877.2780859693339)


    mol = gto.M()
    numpy.random.seed(12)
    nocca, noccb, nvira, nvirb = 3, 2, 4, 5
    nmo = nocca + nvira
    eris = cc.uccsd._ChemistsERIs()
    eri1 = (numpy.random.random((3,nmo,nmo,nmo,nmo)) +
            numpy.random.random((3,nmo,nmo,nmo,nmo)) * .8j - .5-.4j)
    eri1 = eri1 + eri1.transpose(0,2,1,4,3).conj()
    eri1[0] = eri1[0] + eri1[0].transpose(2,3,0,1)
    eri1[2] = eri1[2] + eri1[2].transpose(2,3,0,1)
    eri1 *= .1
    eris.ovov = eri1[0,nocca:,:nocca,nocca:,:nocca].transpose(1,0,3,2).conj()
    eris.ovvv = eri1[0,nocca:,:nocca,nocca:,nocca:].transpose(1,0,3,2).conj()
    eris.ovoo = eri1[0,nocca:,:nocca,:nocca,:nocca].transpose(1,0,3,2).conj()
    eris.OVOV = eri1[2,noccb:,:noccb,noccb:,:noccb].transpose(1,0,3,2).conj()
    eris.OVVV = eri1[2,noccb:,:noccb,noccb:,noccb:].transpose(1,0,3,2).conj()
    eris.OVOO = eri1[2,noccb:,:noccb,:noccb,:noccb].transpose(1,0,3,2).conj()
    eris.ovOV = eri1[1,nocca:,:nocca,noccb:,:noccb].transpose(1,0,3,2).conj()
    eris.ovVV = eri1[1,nocca:,:nocca,noccb:,noccb:].transpose(1,0,3,2).conj()
    eris.ovOO = eri1[1,nocca:,:nocca,:noccb,:noccb].transpose(1,0,3,2).conj()
    eris.OVov = eri1[1,nocca:,:nocca,noccb:,:noccb].transpose(3,2,1,0).conj()
    eris.OVvv = eri1[1,nocca:,nocca:,noccb:,:noccb].transpose(3,2,1,0).conj()
    eris.OVoo = eri1[1,:nocca,:nocca,noccb:,:noccb].transpose(3,2,1,0).conj()
    t1a  = .1 * numpy.random.random((nocca,nvira)) + numpy.random.random((nocca,nvira))*.1j
    t1b  = .1 * numpy.random.random((noccb,nvirb)) + numpy.random.random((noccb,nvirb))*.1j
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira)) + numpy.random.random((nocca,nocca,nvira,nvira))*.1j
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb)) + numpy.random.random((noccb,noccb,nvirb,nvirb))*.1j
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb)) + numpy.random.random((nocca,noccb,nvira,nvirb))*.1j
    f = (numpy.random.random((2,nmo,nmo)) * .4 +
         numpy.random.random((2,nmo,nmo)) * .4j)
    eris.focka = f[0]+f[0].T.conj() + numpy.diag(numpy.arange(nmo))
    eris.fockb = f[1]+f[1].T.conj() + numpy.diag(numpy.arange(nmo))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    mcc = cc.UCCSD(scf.UHF(mol))
    print(kernel(mcc, eris, t1, t2) - (-0.056092415718338388-0.011390417704868244j))
