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
from pyscf.lib import logger
from pyscf.cc import uccsd
from pyscf.cc import uintermediates
import time
import gc

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

    w = numpy.einsum('ijae,ebkc->ijkabc', t2bb, numpy.asarray(imds.wvvov_act))

    w-= numpy.einsum('imab,kcjm->ijkabc', t2bb, numpy.asarray(eris.OVOO).conj())
    r = r6(w)
    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.OVOV).conj(), t1b) # Why?
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
    wvd = p6(w + v) / d3
    et += numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

    # baa

    w  = numpy.einsum('jIeA,ebkc->IjkAbc', t2ab, numpy.asarray(imds.wvvov_act)) * 2


    w += numpy.einsum('jIbE,EAkc->IjkAbc', t2ab, numpy.asarray(imds.wVVov_act)) * 2

    w += numpy.einsum('jkbe,ecIA->IjkAbc', t2aa, numpy.asarray(imds.wvvOV_act))

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
    w  = numpy.einsum('ijae,ebkc->ijkabc', t2ab, numpy.asarray(imds.Wvvov)) * 2

    w += numpy.einsum('ijeb,eakc->ijkabc', t2ab, numpy.asarray(imds.Wvvov)) * 2

    w += numpy.einsum('jkbe,ecia->ijkabc', t2bb, numpy.asarray(imds.Wvvov))

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



def pert_w3(mcc, t1, t2, imds, eris, act_hole, act_particle):


    def cyclic_hole(u):
        return (u + u.transpose(1,2,0,3,4,5)+u.transpose(2,0,1,3,4,5))     

    def cyclic_particle(u):
        return (u + u.transpose(0,1,2,4,5,3)+u.transpose(0,1,2,5,3,4))     


    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea, mo_eb = eris.mo_energy

    mo_ea_o = numpy.diag(imds.Foo_act)
    mo_ea_v = numpy.diag(imds.Fvv_act)+mcc.level_shift
    mo_eb_o = numpy.diag(imds.FOO_act)
    mo_eb_v = numpy.diag(imds.FVV_act)+mcc.level_shift

    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    # aaa
    start = time.time()

    x  = lib.einsum('ijae,beck->ijkabc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], imds.Wvvvo_act)
    x -= lib.einsum('imab,mjck->ijkabc', t2aa[numpy.ix_(act_hole[0], numpy.arange(nocca), act_particle[0], act_particle[0])], imds.Woovo_act)

    end = time.time()
    print("time to make aaa contribution to t3", end-start)

    start = time.time()
    u3aaa = cyclic_hole(cyclic_particle(x)) 
    end = time.time()
    print("time to make aaa permutation", end-start)

    # bbb

    x = lib.einsum('ijae,beck->ijkabc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))], imds.WVVVO_act)
    x -= lib.einsum('imab,mjck->ijkabc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])], imds.WOOVO_act)

    u3bbb = cyclic_particle(cyclic_hole(x)) 

    # baa
    u3baa  = lib.einsum('jIeA,beck->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])], numpy.asarray(imds.Wvvvo_act))    # 2

   #P(jk)
    r = u3baa - u3baa.transpose(0,2,1,3,4,5)

    u3baa = lib.einsum('jIbE,AEck->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))], numpy.asarray(imds.WVVvo_act))     # 2

   #P(bc)p(jk)
    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = lib.einsum('jkbe,ceAI->IjkAbc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], numpy.asarray(imds.WvvVO_act))
    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mIbA,mjck->IjkAbc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.Woovo_act)) 

    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('jMbA,MIck->IjkAbc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])], numpy.asarray(imds.WOOvo_act)) 

   #P(bc)P(jk)

    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mjcb,mkAI->IjkAbc', t2aa[numpy.ix_(numpy.arange(nocca), act_hole[0], act_particle[0], act_particle[0])], numpy.asarray(imds.WooVO_act))

    #P(jk) 

    r += u3baa - u3baa.transpose(0,2,1,3,4,5)


    # bba

    u3bba  = lib.einsum('IJAE,BEck->IJkABc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))], numpy.asarray(imds.WVVvo_act)) 
#  P(AB)

    v = u3bba - u3bba.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('kJcE,BEAI->IJkABc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))], numpy.asarray(imds.WVVVO_act))  
#  P(IJ) 
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = lib.einsum('kIeA,ceBJ->IJkABc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])], numpy.asarray(imds.WvvVO_act))
 # P(IJ)P(AB)  

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = -lib.einsum('IMAB,MJck->IJkABc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])], numpy.asarray(imds.WOOvo_act)) 
#P(IJ)

    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = -lib.einsum('kMcB,MJAI->IJkABc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])], numpy.asarray(imds.WOOVO_act)) 

#P(AB)
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)
    u3bba = -lib.einsum('mJcB,mkAI->IJkABc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.WooVO_act))

#P(IJ)P(AB)

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)


    print("norm of u3aaa", numpy.linalg.norm(u3aaa))
    print("norm of u3bbb", numpy.linalg.norm(u3bbb))
    print("norm of u3baa", numpy.linalg.norm(r))
    print("norm of u3bba", numpy.linalg.norm(v))

    wtriples = u3aaa, u3bbb, r, v

    return wtriples 


def update_amps_t3(mcc, imds, wtriples, t3, eris, act_hole, act_particle):
    '''Update non-canonical MP2 amplitudes'''
    #assert (isinstance(eris, _ChemistsERIs))

    def cyclic_hole(u):
        return (u + u.transpose(1,2,0,3,4,5)+u.transpose(2,0,1,3,4,5))     

    def cyclic_particle(u):
        return (u + u.transpose(0,1,2,4,5,3)+u.transpose(0,1,2,5,3,4))     


    t3aaa, t3bbb, t3baa, t3bba = t3

    u3aaa, u3bbb, u3baa, u3bba = wtriples

    print("BEFORE UPDATE=======")

    print("norm of u3aaa", numpy.linalg.norm(u3aaa))
    print("norm of u3bbb", numpy.linalg.norm(u3bbb))
    print("norm of u3baa", numpy.linalg.norm(u3baa))
    print("norm of u3bba", numpy.linalg.norm(u3bba))
#    noccb, nocca, _, nvirb, nvira, _ = t3baa.shape

    print("=======")

    mo_ea_o = numpy.diag(imds.Foo_act)
    mo_ea_v = numpy.diag(imds.Fvv_act)+mcc.level_shift
    mo_eb_o = numpy.diag(imds.FOO_act)
    mo_eb_v = numpy.diag(imds.FVV_act)+mcc.level_shift


    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    # aaa
    d3aaa_active = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3aaa, imds.Fvv_act)
    u3aaa += cyclic_particle(x)
    x  = -lib.einsum('mjkabc,mi->ijkabc', t3aaa, imds.Foo_act)
    u3aaa += cyclic_hole(x)

    # bbb
    d3bbb_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3bbb, imds.FVV_act)
    u3bbb += cyclic_particle(x)
    x = -lib.einsum('mjkabc,mi->ijkabc', t3bbb, imds.FOO_act)
    u3bbb += cyclic_hole(x)


    print("norm of u3aaa, fock", numpy.linalg.norm(u3aaa))
    print("norm of u3bbb, fock", numpy.linalg.norm(u3bbb))

    print("fock_ooa", numpy.linalg.norm(imds.Foo_act))
    print("fock_oob", numpy.linalg.norm(imds.FOO_act))

    print("fock_vva", numpy.linalg.norm(imds.Fvv_act))
    print("fock_vvb", numpy.linalg.norm(imds.FVV_act))
    # baa
    d3baa_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)

    r = lib.einsum('IjkAec,be->IjkAbc', t3baa, imds.Fvv_act)  
    u3baa += r - r.transpose(0,1,2,3,5,4)
    u3baa += lib.einsum('IjkEbc,AE->IjkAbc', t3baa, imds.FVV_act)  

    r = -lib.einsum('ImkAbc,mj->IjkAbc', t3baa, imds.Foo_act)  
    u3baa += r - r.transpose(0,2,1,3,4,5)
    u3baa -= lib.einsum('MjkAbc,MI->IjkAbc', t3baa, imds.FOO_act)  

# fish out the energy contribution
#    temp_t3 = t3baa + u3baa/d3baa_active
#    print("energy contribution:t3baa", (1.0/4)*lib.einsum('ijkabc,ijkabc', temp_t3.conj(), u3baa))   

    # bba

    d3bba_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eia)

    v = lib.einsum('IJkEBc,AE->IJkABc', t3bba, imds.FVV_act)  
    u3bba += v - v.transpose(0,1,2,4,3,5)

    u3bba += lib.einsum('IJkABe,ce->IJkABc',t3bba, imds.Fvv_act)  

    v = -lib.einsum('MJkABc,MI->IJkABc', t3bba, imds.FOO_act)  
    u3bba += v - v.transpose(1,0,2,3,4,5)
    u3bba -= lib.einsum('IJmABc,mk->IJkABc',t3bba, imds.Foo_act)  

    print("norm of u3baa, fock", numpy.linalg.norm(u3baa))
    print("norm of u3bba, fock", numpy.linalg.norm(u3bba))
# fish out the energy contribution
#    temp_t3 = t3bba + u3bba/d3bba_active
#    print("energy contribution:t3bba", (1.0/4)*lib.einsum('ijkabc,ijkabc', temp_t3.conj(), u3bba))   


# divide by denominator..

#   u3aaa /=d3aaa_active
#   u3bbb /=d3bbb_active
#   u3bba /=d3bba_active
#   u3baa /=d3baa_active

#    t3aaa += u3aaa
#    t3bbb += u3bbb
#    t3baa += u3baa
#    t3bba += u3bba

    print("norm of u3aaa", numpy.linalg.norm(u3aaa))
    print("norm of u3bbb", numpy.linalg.norm(u3bbb))
    print("norm of u3baa", numpy.linalg.norm(u3baa))
    print("norm of u3bba", numpy.linalg.norm(u3bba))

    u3new = u3aaa/d3aaa_active, u3bbb/d3bbb_active, u3baa/d3baa_active, u3bba/d3bba_active

    return u3new  

def _iterative_kernel(mcc, t1, t2, l1, l2, eris, act_hole, act_particle): 

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

#    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.new_logger(mcc, verbose)

    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = numpy.result_type(t1a, t1b, t2aa, t2ab, t2bb)


    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)


    u3aaa = numpy.zeros((len(act_hole[0]),len(act_hole[0]),len(act_hole[0]),len(act_particle[0]),len(act_particle[0]),len(act_particle[0])), dtype=dtype)  
    u3bbb = numpy.zeros((len(act_hole[1]),len(act_hole[1]),len(act_hole[1]),len(act_particle[1]),len(act_particle[1]),len(act_particle[1])), dtype=dtype)
    u3bba = numpy.zeros((len(act_hole[1]),len(act_hole[1]),len(act_hole[0]),len(act_particle[1]),len(act_particle[1]),len(act_particle[0])), dtype=dtype)
    u3baa = numpy.zeros((len(act_hole[1]),len(act_hole[0]),len(act_hole[0]),len(act_particle[1]),len(act_particle[0]),len(act_particle[0])), dtype=dtype)
    t3 = u3aaa, u3bbb, u3baa, u3bba


    start = time.time()

    imds = make_intermediates(mcc, t1, t2, eris, act_hole, act_particle)
    w3triples = pert_w3(mcc, t1, t2, imds, eris, act_hole, act_particle)

    end = time.time()

    print("time to make intermediates", end-start)

    adiis = lib.diis.DIIS(mcc)

    conv = False
    for istep in range(mcc.max_cycle):
       
        error = update_amps_t3(mcc, imds, w3triples, t3, eris, act_hole, act_particle)

        normt = numpy.linalg.norm([numpy.linalg.norm(error[i])
                                  for i in range(4)])

#        normt = numpy.linalg.norm([numpy.linalg.norm(t3new[i] - t3[i])
#                                   for i in range(4)])

        t3new =  tuple(a + b for a, b in zip(t3, error))

        t3 = None
        t3shape = [x.shape for x in t3new]
        t3new = numpy.hstack([x.ravel() for x in t3new])
        t3new = adiis.update(t3new)
        t3new = lib.split_reshape(t3new, t3shape)

        t3, t3new = t3new, None
#       log.info('cycle = %d  norm(t3) = %.6g',
#                 istep+1, normt)

        print("cycle = ", istep+1, "norm(t3) = ", normt) 

        if normt < mcc.conv_tol_normt:
            conv = True
            break

    e_triples = lhs_umpcc_triples_active(mcc, t1, t2, l1, l2, t3, eris, act_hole, act_particle) 
    print("active contribution", e_triples)

    e_triples += lhs_umpcc_triples_inactive(mcc, t1, t2, l1, l2, t3, eris, act_hole, act_particle) 

 #  e_triples = lhs_umpcc_triples(mcc, t1, t2, l1, l2, t3, eris, act_hole, act_particle) 

    return e_triples
#    log.timer('MP2', *cput0)


def make_intermediates(mcc, t1, t2, eris, act_hole, act_particle):

#    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(self.stdout, self.verbose)

#    t1, t2, eris = self.t1, self.t2, self.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = numpy.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    class _IMDS: pass
    imds = _IMDS()

#   t1anew = t1a*0.0
#   t1bnew = t1b*0.0
#   t2aanew = t2aa*0.0
#   t2bbnew = t2bb*0.0
#   t2abnew = t2ab*0.0

#   t1new = t1anew, t1bnew
#   t2new = t2aanew, t2abnew, t2bbnew

# Foo, Fvv and Fov
    Foo, FOO = uintermediates.Foo(t1, t2, eris)
    Fvv, FVV = uintermediates.Fvv(t1, t2, eris)
    Fov, FOV = uintermediates.Fov(t1, t2, eris)
# Woovo 
    Woovo, WooVO, WOOvo, WOOVO = uintermediates.Woovo(t1, t2, eris)
#Wvvvo
    Wvvvo, WvvVO, WVVvo, WVVVO = uintermediates.Wvvvo_t3(t1, t2, eris)

    Wvvvo_act, WvvVO_act, WVVvo_act, WVVVO_act = get_vvvv_to_imds(mcc, t1, t2, eris, act_hole, act_particle)

#Wvvov and Wovoo

    Wvvov, WvvOV, WVVov, WVVOV = uintermediates.Wvvov(t1, t2, eris)
    Wovoo, WOVoo, WovOO, WOVOO = uintermediates.Wovoo(t1, eris)

#Wovov
#anti-symmetrize this array:
#==========================

    Wovov = numpy.asarray(eris.ovov) - numpy.asarray(eris.ovov).transpose(0,3,2,1)
    WOVOV = numpy.asarray(eris.OVOV) - numpy.asarray(eris.OVOV).transpose(0,3,2,1)
    WovOV = numpy.asarray(eris.ovOV)

    imds.Wovov_act = Wovov[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], act_particle[0])]
    imds.WOVOV_act = WOVOV[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], act_particle[1])]
    imds.WovOV_act = WovOV[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_particle[1])]


    imds.Foo_act = Foo[numpy.ix_(act_hole[0], act_hole[0])]
    imds.FOO_act = FOO[numpy.ix_(act_hole[1], act_hole[1])]
    imds.FOV_act = FOV[numpy.ix_(act_hole[1], act_particle[1])]
    imds.Fov_act = Fov[numpy.ix_(act_hole[0], act_particle[0])]
    imds.Fvv_act = Fvv[numpy.ix_(act_particle[0], act_particle[0])]
    imds.FVV_act = FVV[numpy.ix_(act_particle[1], act_particle[1])]

    imds.Woovo_act = Woovo[numpy.ix_(numpy.arange(nocca), act_hole[0], act_particle[0], act_hole[0])]
    imds.WooVO_act = WooVO[numpy.ix_(numpy.arange(nocca), act_hole[0], act_particle[1], act_hole[1])]
    imds.WOOvo_act = WOOvo[numpy.ix_(numpy.arange(noccb), act_hole[1], act_particle[0], act_hole[0])]
    imds.WOOVO_act = WOOVO[numpy.ix_(numpy.arange(noccb), act_hole[1], act_particle[1], act_hole[1])]

    imds.Wvvvo_act = Wvvvo[numpy.ix_(act_particle[0],numpy.arange(nvira), act_particle[0], act_hole[0])]  + Wvvvo_act
    imds.WvvVO_act = WvvVO[numpy.ix_(act_particle[0],numpy.arange(nvira), act_particle[1], act_hole[1])]  + WvvVO_act
    imds.WVVvo_act = WVVvo[numpy.ix_(act_particle[1],numpy.arange(nvirb), act_particle[0], act_hole[0])]  + WVVvo_act
    imds.WVVVO_act = WVVVO[numpy.ix_(act_particle[1],numpy.arange(nvirb), act_particle[1], act_hole[1])]  + WVVVO_act

    Wvvvo = WvvVO = WVVvo =  WVVVO = None

    imds.Wvvov_act = Wvvov[numpy.ix_(act_particle[0], act_particle[0], act_hole[0], act_particle[0])].copy()
    imds.WvvOV_act = WvvOV[numpy.ix_(act_particle[0], act_particle[0], act_hole[1], act_particle[1])].copy()
    imds.WVVov_act = WVVov[numpy.ix_(act_particle[1], act_particle[1], act_hole[0], act_particle[0])].copy()
    imds.WVVOV_act = WVVOV[numpy.ix_(act_particle[1], act_particle[1], act_hole[1], act_particle[1])].copy()

    Wvvov = WvvOV = WVVov =  WVVOV = None

# Wovoo
    imds.Wovoo_act = Wovoo[numpy.ix_(act_hole[0], act_particle[0], act_hole[0],act_hole[0])]
    imds.WOVoo_act = WOVoo[numpy.ix_(act_hole[1], act_particle[1], act_hole[0],act_hole[0])]
    imds.WovOO_act = WovOO[numpy.ix_(act_hole[0], act_particle[0], act_hole[1],act_hole[1])]
    imds.WOVOO_act = WOVOO[numpy.ix_(act_hole[1], act_particle[1], act_hole[1],act_hole[1])]
    return imds 

def make_intermediates_energy(mcc, t1, t2, eris, act_hole, act_particle):

    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(self.stdout, self.verbose)

#   t1, t2, eris = self.t1, self.t2, self.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    dtype = numpy.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    class _IMDS: pass
    imds = _IMDS()

#   t1anew = t1a*0.0
#   t1bnew = t1b*0.0
#   t2aanew = t2aa*0.0
#   t2bbnew = t2bb*0.0
#   t2abnew = t2ab*0.0

#   t1new = t1anew, t1bnew
#   t2new = t2aanew, t2abnew, t2bbnew

    Wvvov, WvvOV, WVVov, WVVOV = uintermediates.Wvvov(t1, t2, eris)
    Wovoo, WOVoo, WovOO, WOVOO = uintermediates.Wovoo(t1, eris)
    Fova, Fovb = uintermediates.Fov(t1, t2, eris)

    imds.Fova_active = Fova[numpy.ix_(act_hole[0], act_particle[0])]
    imds.Fovb_active = Fovb[numpy.ix_(act_hole[1], act_particle[1])]

#Now sort the intermediates numpy.ix_(based on the active and inactive indices..

# Wvvov

#   imds.Wvvov_act = Wvvov[numpy.ix_(numpy.arange(nvira), act_particle[0], act_hole[0], act_particle[0])]
#   imds.WvvOV_act = WvvOV[numpy.ix_(numpy.arange(nvira), act_particle[0], act_hole[1], act_particle[1])]
#   imds.WVVov_act = WVVov[numpy.ix_(numpy.arange(nvirb), act_particle[1], act_hole[0], act_particle[0])]
#   imds.WVVOV_act = WVVOV[numpy.ix_(numpy.arange(nvirb), act_particle[1], act_hole[1], act_particle[1])]

    imds.Wvvov_act = Wvvov[numpy.ix_(act_particle[0], act_particle[0], act_hole[0], act_particle[0])]
    imds.WvvOV_act = WvvOV[numpy.ix_(act_particle[0], act_particle[0], act_hole[1], act_particle[1])]
    imds.WVVov_act = WVVov[numpy.ix_(act_particle[1], act_particle[1], act_hole[0], act_particle[0])]
    imds.WVVOV_act = WVVOV[numpy.ix_(act_particle[1], act_particle[1], act_hole[1], act_particle[1])]

    imds.Wvvov_inact = Wvvov[numpy.ix_(inact_particle[0], act_particle[0], act_hole[0], act_particle[0])]
    imds.WvvOV_inact = WvvOV[numpy.ix_(inact_particle[0], act_particle[0], act_hole[1], act_particle[1])]
    imds.WVVov_inact = WVVov[numpy.ix_(inact_particle[1], act_particle[1], act_hole[0], act_particle[0])]
    imds.WVVOV_inact = WVVOV[numpy.ix_(inact_particle[1], act_particle[1], act_hole[1], act_particle[1])]

# Wovoo
#   imds.Wovoo_act = Wovoo[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], numpy.arange(nocca))]
#   imds.WOVoo_act = WOVoo[numpy.ix_(act_hole[1], act_particle[1], act_hole[0], numpy.arange(nocca))]
#   imds.WovOO_act = WovOO[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], numpy.arange(noccb))]
#   imds.WOVOO_act = WOVOO[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], numpy.arange(noccb))]

    imds.Wovoo_act = Wovoo[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], act_hole[0])]
    imds.WOVoo_act = WOVoo[numpy.ix_(act_hole[1], act_particle[1], act_hole[0], act_hole[0])]
    imds.WovOO_act = WovOO[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_hole[1])]
    imds.WOVOO_act = WOVOO[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], act_hole[1])]


    imds.Wovoo_inact = Wovoo[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], inact_hole[0])]
    imds.WOVoo_inact = WOVoo[numpy.ix_(act_hole[1], act_particle[1], act_hole[0], inact_hole[0])]
    imds.WovOO_inact = WovOO[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], inact_hole[1])]
    imds.WOVOO_inact = WOVOO[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], inact_hole[1])]

# Wovov

#anti-symmetrize this array:
#==========================

    Wovov = numpy.asarray(eris.ovov) - numpy.asarray(eris.ovov).transpose(0,3,2,1)
    WOVOV = numpy.asarray(eris.OVOV) - numpy.asarray(eris.OVOV).transpose(0,3,2,1)
    WovOV = numpy.asarray(eris.ovOV)

    imds.Wovov_act = Wovov[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], act_particle[0])]
    imds.WOVOV_act = WOVOV[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], act_particle[1])]
    imds.WovOV_act = WovOV[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_particle[1])]

    return imds


def get_vvvv_to_imds(mcc, t1, t2, eris, act_hole, act_particle):
    from pyscf import ao2mo

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    dtype = numpy.result_type(t1a, t1b)


    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    # get the active dimensions

    nact_vir_a = act_particle[0].size
    nact_vir_b = act_particle[1].size
    nact_occ_a = act_hole[0].size
    nact_occ_b = act_hole[1].size

    vir_a = eris.mo_coeff[0][:,nocca:]
    vir_b = eris.mo_coeff[1][:,noccb:]

    vir_act_a = vir_a[:,act_particle[0][0]:(act_particle[0][-1]+1)]
    vir_act_b = vir_b[:,act_particle[1][0]:(act_particle[1][-1]+1)]


    occ_a = eris.mo_coeff[0][:,:nocca]
    occ_b = eris.mo_coeff[1][:,:noccb]

#aaaa
    eris_vvov = ao2mo.general(mcc._scf._eri, (vir_act_a, vir_a, occ_a, vir_a),
                                 compact=False).reshape(nact_vir_a,nvira,nocca,nvira)


    eris_vvOV = ao2mo.general(mcc._scf._eri, (vir_act_a, vir_a, occ_b, vir_b),
                                 compact=False).reshape(nact_vir_a,nvira,noccb,nvirb)


    wvvvo = eris_vvov[numpy.ix_(numpy.arange(nact_vir_a), numpy.arange(nvira), act_hole[0], act_particle[0])].transpose(0,1,3,2) 

    vvov = eris_vvov - eris_vvov.transpose(0,3,2,1)


    wvvvo += lib.einsum('aemf,mifb->aebi', vvov, t2aa[numpy.ix_(numpy.arange(nocca), act_hole[0], numpy.arange(nvira), act_particle[0])])

    wvvvo += lib.einsum('aeMF,iMbF->aebi', eris_vvOV, t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], numpy.arange(nvirb))])


    wvvvo = wvvvo - wvvvo.transpose(2,1,0,3)


#bbbb

    eris_VVOV = ao2mo.general(mcc._scf._eri, (vir_act_b, vir_b, occ_b, vir_b),
                                 compact=False).reshape(nact_vir_b,nvirb,noccb,nvirb)

    wVVVO = eris_VVOV[numpy.ix_(numpy.arange(nact_vir_b), numpy.arange(nvirb), act_hole[1], act_particle[1])].transpose(0,1,3,2) 

    VVOV = eris_VVOV - eris_VVOV.transpose(0,3,2,1)


    eris_VVov = ao2mo.general(mcc._scf._eri, (vir_act_b, vir_b, occ_a, vir_a),
                                 compact=False).reshape(nact_vir_b,nvirb,nocca,nvira)


    wVVVO += lib.einsum('aemf,mifb->aebi', VVOV, t2bb[numpy.ix_(numpy.arange(noccb), act_hole[1], numpy.arange(nvirb), act_particle[1])])

    wVVVO += lib.einsum('AEmf,mIfB->AEBI', eris_VVov, t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], numpy.arange(nvira), act_particle[1])])

    wVVVO = wVVVO - wVVVO.transpose(2,1,0,3)

 #

    wVVvo = eris_VVov[numpy.ix_(numpy.arange(nact_vir_b), numpy.arange(nvirb), act_hole[0], act_particle[0])].transpose(0,1,3,2) 
     
    wVVvo += lib.einsum('AEmf,mifb->AEbi', eris_VVov, t2aa[numpy.ix_(numpy.arange(nocca), act_hole[0], numpy.arange(nvira), act_particle[0])])
    wVVvo += lib.einsum('AEMF,iMbF->AEbi', VVOV,t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], numpy.arange(nvirb))])
    wVVvo -= lib.einsum('bfME,iMfA->AEbi', eris_vvOV, t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), numpy.arange(nvira), act_particle[1])])


##

    wvvVO = eris_vvOV[numpy.ix_(numpy.arange(nact_vir_a), numpy.arange(nvira), act_hole[1], act_particle[1])].transpose(0,1,3,2) 


    wvvVO += lib.einsum('aemf,mIfB->aeBI', vvov, t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], numpy.arange(nvira), act_particle[1])])

    wvvVO += lib.einsum('aeMF,IMBF->aeBI', eris_vvOV, t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], numpy.arange(nvirb))])

    wvvVO -= lib.einsum('BFme,mIaF->aeBI', eris_VVov, t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], numpy.arange(nvirb))])
##

    vvov = VVOV = eris_VVov = eris_vvOV = eris_VVOV = eris_vvov = None


    vvvv_act = ao2mo.general(mcc._scf._eri, (vir_act_a, vir_a, vir_act_a, vir_a),
                                  compact=False).reshape(nact_vir_a, nvira, nact_vir_a, nvira)

    VVVV_act = ao2mo.general(mcc._scf._eri, (vir_act_b, vir_b, vir_act_b, vir_b),
                                 compact=False).reshape(nact_vir_b, nvirb, nact_vir_b, nvirb)

    vvVV_act = ao2mo.general(mcc._scf._eri, (vir_act_a, vir_a, vir_act_b, vir_b),
                                    compact=False).reshape(nact_vir_a, nvira, nact_vir_b, nvirb)

    vvvv_act_new = vvvv_act - vvvv_act.transpose(2,1,0,3)
    VVVV_act_new = VVVV_act - VVVV_act.transpose(2,1,0,3)

    wvvvo   += lib.einsum('aebf,if->aebi', vvvv_act_new, t1a[numpy.ix_(act_hole[0], numpy.arange(nvira))])

    wVVVO   += lib.einsum('aebf,if->aebi', VVVV_act_new, t1b[numpy.ix_(act_hole[1], numpy.arange(nvirb))])

    wVVvo   += lib.einsum('bfAE,if->AEbi', vvVV_act, t1a[numpy.ix_(act_hole[0], numpy.arange(nvira))])
    wvvVO   += lib.einsum('aeBF,IF->aeBI', vvVV_act, t1b[numpy.ix_(act_hole[1], numpy.arange(nvirb))])

    vvvv_act = vvvv_act_new = VVVV_act = VVVV_act_new = vvVV_act = None

    ovoo, OVoo, ovOO, OVOO = uintermediates.Wovoo(t1, eris)

    wvvvo += lib.einsum('meni,mnab->aebi', ovoo[numpy.ix_(numpy.arange(nocca), numpy.arange(nvira), numpy.arange(nocca), act_hole[0])],
             t2aa[numpy.ix_(numpy.arange(nocca), numpy.arange(nocca), act_particle[0], act_particle[0])])*0.5
    wVVVO += lib.einsum('meni,mnab->aebi', OVOO[numpy.ix_(numpy.arange(noccb), numpy.arange(nvirb), numpy.arange(noccb), act_hole[1])],
             t2bb[numpy.ix_(numpy.arange(noccb), numpy.arange(noccb), act_particle[1], act_particle[1])])*0.5
    wvvVO += lib.einsum('meni,mnab->aebi', ovOO[numpy.ix_(numpy.arange(nocca), numpy.arange(nvira), numpy.arange(noccb), act_hole[1])], t2ab[numpy.ix_(numpy.arange(nocca), numpy.arange(noccb), act_particle[0], act_particle[1])])
    wVVvo += lib.einsum('meni,nmba->aebi', OVoo[numpy.ix_(numpy.arange(noccb), numpy.arange(nvirb), numpy.arange(nocca), act_hole[0])], t2ab[numpy.ix_(numpy.arange(nocca), numpy.arange(noccb), act_particle[0], act_particle[1])])

    ovoo = OVoo = ovOO = OVOO = None




    return wvvvo, wvvVO, wVVvo, wVVVO


def get_t3_to_imds(mcc, t3, t1, eris, act_hole, act_particle):
    from pyscf import ao2mo

    t3aaa, t3bbb, t3baa, t3bba = t3

    t1a, t1b = t1
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    dtype = numpy.result_type(t3aaa, t3bbb)

    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    # get the active dimensions

    eris_ovov = numpy.asarray(eris.ovov)
    eris_OVOV = numpy.asarray(eris.OVOV)
    eris_ovOV = numpy.asarray(eris.ovOV)

    ovov = numpy.asarray(eris.ovov) - numpy.asarray(eris.ovov).transpose(0,3,2,1)
    OVOV = numpy.asarray(eris.OVOV) - numpy.asarray(eris.OVOV).transpose(0,3,2,1)


    wvvvo = -lib.einsum('mnjafb,menf->aebj', t3aaa, ovov[numpy.ix_(act_hole[0], numpy.arange(nvira), act_hole[0], act_particle[0])])*0.5
    wvvvo -= lib.einsum('NmjFab,meNF->aebj', t3baa, eris_ovOV[numpy.ix_(act_hole[0], numpy.arange(nvira), act_hole[1], act_particle[1])])


    wVVVO = -lib.einsum('mnjafb,menf->aebj', t3bbb, OVOV[numpy.ix_(act_hole[1], numpy.arange(nvirb), act_hole[1], act_particle[1])])*0.5
    wVVVO -= lib.einsum('JMnBAf,nfME->AEBJ', t3bba, eris_ovOV[numpy.ix_(act_hole[0], act_particle[0],act_hole[1],numpy.arange(nvirb))])


    wvvVO = -lib.einsum('JmnBaf,menf->aeBJ', t3baa, ovov[numpy.ix_(act_hole[0], numpy.arange(nvira), act_hole[0], act_particle[0])])*0.5
    wvvVO -= lib.einsum('JNmBFa,meNF->aeBJ', t3bba, eris_ovOV[numpy.ix_(act_hole[0], numpy.arange(nvira), act_hole[1], act_particle[1])])


    wVVvo = -lib.einsum('NMjFAb,MENF->AEbj', t3bba, OVOV[numpy.ix_(act_hole[1], numpy.arange(nvirb), act_hole[1], act_particle[1])])*0.5 
    wVVvo -= lib.einsum('MnjAfb,nfME->AEbj', t3baa, eris_ovOV[numpy.ix_(act_hole[0], act_particle[0],act_hole[1],numpy.arange(nvirb))])

############

    woovo = lib.einsum('ijnaef,menf->mjai', t3aaa, ovov[numpy.ix_(numpy.arange(nocca), act_particle[0],  act_hole[0], act_particle[0])])*0.5
    woovo += lib.einsum('NjiFea,meNF->mjai', t3baa, eris_ovOV[numpy.ix_(numpy.arange(nocca), act_particle[0],  act_hole[1], act_particle[1])])

    wOOVO = lib.einsum('ijnaef,menf->mjai', t3bbb, OVOV[numpy.ix_(numpy.arange(noccb), act_particle[1],  act_hole[1], act_particle[1])])*0.5
    wOOVO += lib.einsum('IJnAEf,nfME->MJAI', t3bba, eris_ovOV[numpy.ix_(numpy.arange(nocca), act_particle[0], act_hole[1], act_particle[1])])

    wooVO = lib.einsum('IjnAef,menf->mjAI', t3baa, ovov[numpy.ix_(numpy.arange(nocca), act_particle[0],  act_hole[0], act_particle[0])])*0.5
    wooVO += lib.einsum('INjAFe,meNF->mjAI', t3bba, eris_ovOV[numpy.ix_(numpy.arange(nocca), act_particle[0],  act_hole[1], act_particle[1])])


    wOOvo = lib.einsum('NJiFEa,MENF->MJai', t3bba, OVOV[numpy.ix_(numpy.arange(noccb), act_particle[1],  act_hole[1], act_particle[1])])*0.5
    wOOvo += lib.einsum('JniEfa,nfME->MJai', t3baa, eris_ovOV[numpy.ix_(numpy.arange(nocca), act_particle[0],  act_hole[1], act_particle[1])])


    return wvvvo, wvvVO, wVVvo, wVVVO, woovo, wOOVO, wooVO, wOOvo


def get_vvvv_imds(mcc, t1, t2, eris, act_hole, act_particle):
    from pyscf import ao2mo

    t1a, t1b = t1
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    dtype = numpy.result_type(t1a, t1b)


    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    # get the active dimensions

    nact_vir_a = act_particle[0].size
    nact_vir_b = act_particle[1].size
    nact_occ_a = act_hole[0].size
    nact_occ_b = act_hole[1].size

    occ_a = eris.mo_coeff[0][:,:nocca]
    occ_b = eris.mo_coeff[1][:,:noccb]

    vir_a = eris.mo_coeff[0][:,nocca:]
    vir_b = eris.mo_coeff[1][:,noccb:]

    vir_act_a = vir_a[:,act_particle[0][0]:(act_particle[0][-1]+1)]
    vir_act_b = vir_b[:,act_particle[1][0]:(act_particle[1][-1]+1)]

    vvvv = ao2mo.general(mcc._scf._eri, (vir_act_a, vir_act_a, vir_act_a, vir_act_a),
                                 compact=False).reshape(nact_vir_a, nact_vir_a, nact_vir_a, nact_vir_a)

    VVVV = ao2mo.general(mcc._scf._eri, (vir_act_b, vir_act_b, vir_act_b, vir_act_b),
                                 compact=False).reshape(nact_vir_b, nact_vir_b, nact_vir_b, nact_vir_b)

    vvVV_act = ao2mo.general(mcc._scf._eri, (vir_act_a, vir_act_a, vir_act_b, vir_act_b),
                                    compact=False).reshape(nact_vir_a, nact_vir_a, nact_vir_b, nact_vir_b)


    vvvv_act = vvvv - vvvv.transpose(0,3,2,1)
    VVVV_act = VVVV - VVVV.transpose(0,3,2,1)

    # ovvv*t1 -> vvvv 

    #(is it okay to do the transformation in this manner?)

    
    eris_ovvv = ao2mo.general(mcc._scf._eri, (occ_a, vir_act_a, vir_act_a, vir_act_a),
                                 compact=False).reshape(nocca, nact_vir_a, nact_vir_a, nact_vir_a)

    eris_OVVV = ao2mo.general(mcc._scf._eri, (occ_b, vir_act_b, vir_act_b, vir_act_b),
                                 compact=False).reshape(noccb, nact_vir_b, nact_vir_b, nact_vir_b)

    eris_ovVV = ao2mo.general(mcc._scf._eri, (occ_a, vir_act_a, vir_act_b, vir_act_b),
                                 compact=False).reshape(nocca, nact_vir_a, nact_vir_b, nact_vir_b)


    eris_OVvv = ao2mo.general(mcc._scf._eri, (occ_b, vir_act_b, vir_act_a, vir_act_a),
                                 compact=False).reshape(noccb, nact_vir_b, nact_vir_a, nact_vir_a)

    
    ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
    OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)


#    tmp = lib.einsum('mb,mfae->fbea', t1a[numpy.ix_(numpy.arange(nocca), act_particle[0])], ovvv)*-1.0
#    vvvv_act += tmp - tmp.transpose(0,3,2,1)  

    tmp = lib.einsum('mb,mfae->aebf', t1a[numpy.ix_(numpy.arange(nocca), act_particle[0])], ovvv)*-1.0
    vvvv_act += tmp - tmp.transpose(2,1,0,3)  

    tmp = lib.einsum('mb,mfae->aebf', t1b[numpy.ix_(numpy.arange(noccb), act_particle[1])], OVVV)*-1.0
    VVVV_act += tmp - tmp.transpose(2,1,0,3)  

    vvVV_act -= lib.einsum('mb,mfAE->bfAE', t1a[numpy.ix_(numpy.arange(nocca), act_particle[0])], eris_ovVV)   

    vvVV_act -= lib.einsum('MB,MFae->aeBF', t1b[numpy.ix_(numpy.arange(noccb), act_particle[1])], eris_OVvv)  #as the hamiltonian is not hermitian anymore. 

    #V*T2 -> vvvv

    eris_ovov = numpy.asarray(eris.ovov)
    eris_OVOV = numpy.asarray(eris.OVOV)
    eris_ovOV = numpy.asarray(eris.ovOV)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    tauaa, tauab, taubb = uintermediates.make_tau(t2, t1, t1)
    vvvv_act += 0.5*lib.einsum('mnab,menf->aebf', tauaa[numpy.ix_(numpy.arange(nocca), numpy.arange(nocca), act_particle[0], act_particle[0])], ovov[numpy.ix_(numpy.arange(nocca), act_particle[0], numpy.arange(nocca), act_particle[0])])
    VVVV_act += 0.5*lib.einsum('mnab,menf->aebf', taubb[numpy.ix_(numpy.arange(noccb), numpy.arange(noccb), act_particle[1], act_particle[1])], OVOV[numpy.ix_(numpy.arange(noccb), act_particle[1], numpy.arange(noccb), act_particle[1])])
    vvVV_act += lib.einsum('mNaB,meNF->aeBF', tauab[numpy.ix_(numpy.arange(nocca), numpy.arange(noccb), act_particle[0], act_particle[1])], eris_ovOV[numpy.ix_(numpy.arange(nocca), act_particle[0], numpy.arange(noccb), act_particle[1])])

    return vvvv_act, VVVV_act, vvVV_act


def lhs_umpcc_triples(mcc, t1, t2, l1, l2, t3, eris, act_hole, act_particle): 
    '''    
    t1, t2 amplitudes will be used to build the lhs. later we will replace them by L1 and L2 amplitudes..
    ''' 

    print("shape of t3", len(t3))


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

    l1a,l1b = l1
    l2aa,l2ab,l2bb = l2
    t3aaa,t3bbb,t3baa,t3bba = t3

    nocca, noccb, nvira, nvirb = l2ab.shape
    dtype = numpy.result_type(l1a, l1b, l2aa, l2ab, l2bb)


    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    imds = make_intermediates_energy(mcc, t1, t2, eris, act_hole, act_particle)

    l1a_active = l1a[numpy.ix_(act_hole[0], act_particle[0])]
    l1b_active = l1b[numpy.ix_(act_hole[1], act_particle[1])]

#aaa
    v = lib.einsum('ebkc,ijae->ijkabc', imds.Wvvov_act, l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],numpy.arange(nvira))]) 
    v -= lib.einsum('iajm,mkbc->ijkabc', imds.Wovoo_act, l2aa[numpy.ix_(numpy.arange(nocca), act_hole[0],act_particle[0],act_particle[0])])

    v += lib.einsum('jbkc,ia->ijkabc', imds.Wovov_act, l1a_active)
    v += lib.einsum('jkbc,ia->ijkabc', l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])], imds.Fova_active)

    wd = cyclic_particle(cyclic_hole(v)) 

    et = lib.einsum('ijkabc,ijkabc', wd.conj(), t3aaa)*(1.0/36)

    print("value of et, step 1:", et)

#bbb
    v = lib.einsum('ebkc,ijae->ijkabc', imds.WVVOV_act, l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],numpy.arange(nvirb))]) 
    v -= lib.einsum('iajm,mkbc->ijkabc',imds.WOVOO_act, l2bb[numpy.ix_(numpy.arange(noccb), act_hole[1],act_particle[1],act_particle[1])])
    v += lib.einsum('jbkc,ia->ijkabc', imds.WOVOV_act, l1b_active)
    v += lib.einsum('jkbc,ia->ijkabc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])], imds.Fovb_active)

    wd = cyclic_particle(cyclic_hole(v)) 

    et += lib.einsum('ijkabc,ijkabc', wd.conj(), t3bbb)*(1.0/36)

    print("value of et, step 2:", et)
#baa

  #  [6 terms to insert here..]

    w  = lib.einsum('ebkc,jIeA->IjkAbc', imds.Wvvov_act, l2ab[numpy.ix_(act_hole[0], act_hole[1],numpy.arange(nvira),act_particle[1])]) #done 
    w -= lib.einsum('mkbc,IAjm->IjkAbc', l2aa[numpy.ix_(numpy.arange(nocca), act_hole[0],act_particle[0],act_particle[0])], imds.WOVoo_act) #done #check if the defn of W can be changed
    #P(jk)
    r = w - w.transpose(0,2,1,3,4,5)

    w  = lib.einsum('ebIA,jkec->IjkAbc', imds.WvvOV_act, l2aa[numpy.ix_(act_hole[0], act_hole[0],numpy.arange(nvira),act_particle[0])]) #Done 
    w -= lib.einsum('mIbA,kcjm->IjkAbc', l2ab[numpy.ix_(numpy.arange(nocca), act_hole[1],act_particle[0],act_particle[1])], imds.Wovoo_act) 
    #P(bc)
    r += w - w.transpose(0,1,2,3,5,4)

    w  = lib.einsum('EAkc,jIbE->IjkAbc', imds.WVVov_act, l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],numpy.arange(nvirb))]) #done
    w -= lib.einsum('jMbA,kcIM->IjkAbc', l2ab[numpy.ix_(act_hole[0], numpy.arange(noccb),act_particle[0],act_particle[1])], imds.WovOO_act) 
    w += lib.einsum('kcIA,jb->IjkAbc', imds.WovOV_act, l1a_active)
    w += lib.einsum('kIcA,jb->IjkAbc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.Fova_active) 
    #P(jk)P(bc)

    y = w - w.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    r  += lib.einsum('jbkc,IA->IjkAbc', imds.Wovov_act, l1b_active)
    r  += lib.einsum('jkbc,IA->IjkAbc', l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])], imds.Fovb_active) 

    #P(None)

#    r = w - w.transpose(0,2,1,3,4,5)
#    r = r - r.transpose(0,1,2,3,5,4)

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3baa)*(1.0/4)
    print("value of et, step 3:", et)
 
#bba
  # [6 terms to insert here..]

    w  = lib.einsum('kJcE,EBIA->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],numpy.arange(nvirb))], imds.WVVOV_act) #done 
    w -= lib.einsum('IMAB,kcJM->IJkABc', l2bb[numpy.ix_(act_hole[1], numpy.arange(noccb),act_particle[1],act_particle[1])], imds.WovOO_act) #done

# P(IJ)

    r = w - w.transpose(1,0,2,3,4,5)

    w  = lib.einsum('kJeB,ecIA->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],numpy.arange(nvira),act_particle[1])], imds.WvvOV_act) #done
    w -= lib.einsum('mIcA,JBkm->IJkABc', l2ab[numpy.ix_(numpy.arange(nocca), act_hole[1],act_particle[0],act_particle[1])], imds.WOVoo_act)  #done
    w += lib.einsum('kcIA,JB->IJkABc', imds.WovOV_act, l1b_active)
    w += lib.einsum('kIcA,JB->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.Fovb_active) 

# P(IJ)P(AB)

    y = w - w.transpose(1,0,2,3,4,5)
    r += y - y.transpose(0,1,2,4,3,5)

    w = lib.einsum('IJAE,EBkc->IJkABc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],numpy.arange(nvirb))], imds.WVVov_act) 
    w -= lib.einsum('kMcB,IAJM->IJkABc', l2ab[numpy.ix_(act_hole[0], numpy.arange(noccb),act_particle[0],act_particle[1])], imds.WOVOO_act) 

# P(AB) 

    r += w - w.transpose(0,1,2,4,3,5)

    r  += lib.einsum('IAJB,kc->IJkABc', imds.WOVOV_act, l1a_active)
    r  += lib.einsum('IJAB,kc->IJkABc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])], imds.Fova_active) 

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3bba)*(1.0/4)

    print("value of et, step 4:", et)

#    et *= .25

    return et


def lhs_umpcc_triples_active(mcc, t1, t2, l1, l2, t3, eris, act_hole, act_particle): 
    '''    
    t1, t2 amplitudes will be used to build the lhs. later we will replace them by L1 and L2 amplitudes..
    ''' 

    print("shape of t3", len(t3))


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

    l1a,l1b = l1
    l2aa,l2ab,l2bb = l2
    t3aaa,t3bbb,t3baa,t3bba = t3

    nocca, noccb, nvira, nvirb = l2ab.shape
    dtype = numpy.result_type(l1a, l1b, l2aa, l2ab, l2bb)


    inact_particle_a = numpy.delete(act_particle[0], act_particle[0])
    inact_particle_b = numpy.delete(act_particle[1], act_particle[1])

    inact_hole_a = numpy.delete(act_hole[0], act_hole[0])
    inact_hole_b = numpy.delete(act_hole[1], act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    imds = make_intermediates_energy(mcc, t1, t2, eris, act_hole, act_particle)

    l1a_active = l1a[numpy.ix_(act_hole[0], act_particle[0])]
    l1b_active = l1b[numpy.ix_(act_hole[1], act_particle[1])]

#aaa
    v = lib.einsum('ebkc,ijae->ijkabc', imds.Wvvov_act, l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])]) 
    v -= lib.einsum('iajm,mkbc->ijkabc', imds.Wovoo_act, l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])])


    v += lib.einsum('jbkc,ia->ijkabc', imds.Wovov_act, l1a_active)
    v += lib.einsum('jkbc,ia->ijkabc', l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])], imds.Fova_active)

    wd = cyclic_particle(cyclic_hole(v)) 

    et = lib.einsum('ijkabc,ijkabc', wd.conj(), t3aaa)*(1.0/36)

    print("value of et, step 1:", et)

#bbb
    v = lib.einsum('ebkc,ijae->ijkabc', imds.WVVOV_act, l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])]) 
    v -= lib.einsum('iajm,mkbc->ijkabc',imds.WOVOO_act, l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])])
    v += lib.einsum('jbkc,ia->ijkabc', imds.WOVOV_act, l1b_active)
    v += lib.einsum('jkbc,ia->ijkabc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])], imds.Fovb_active)

    wd = cyclic_particle(cyclic_hole(v)) 

    et += lib.einsum('ijkabc,ijkabc', wd.conj(), t3bbb)*(1.0/36)

    print("value of et, step 2:", et)
#baa

  #  [6 terms to insert here..]

    w  = lib.einsum('ebkc,jIeA->IjkAbc', imds.Wvvov_act, l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])]) #done 
    w -= lib.einsum('mkbc,IAjm->IjkAbc', l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])], imds.WOVoo_act) #done #check if the defn of W can be changed
    #P(jk)
    r = w - w.transpose(0,2,1,3,4,5)

    w  = lib.einsum('ebIA,jkec->IjkAbc', imds.WvvOV_act, l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])]) #Done 
    w -= lib.einsum('mIbA,kcjm->IjkAbc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.Wovoo_act) 
    #P(bc)
    r += w - w.transpose(0,1,2,3,5,4)

    w  = lib.einsum('EAkc,jIbE->IjkAbc', imds.WVVov_act, l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])]) #done
    w -= lib.einsum('jMbA,kcIM->IjkAbc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.WovOO_act) 
    w += lib.einsum('kcIA,jb->IjkAbc', imds.WovOV_act, l1a_active)
    w += lib.einsum('kIcA,jb->IjkAbc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.Fova_active) 
    #P(jk)P(bc)

    y = w - w.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    r  += lib.einsum('jbkc,IA->IjkAbc', imds.Wovov_act, l1b_active)
    r  += lib.einsum('jkbc,IA->IjkAbc', l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],act_particle[0])], imds.Fovb_active) 

    #P(None)

#    r = w - w.transpose(0,2,1,3,4,5)
#    r = r - r.transpose(0,1,2,3,5,4)

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3baa)*(1.0/4)
    print("value of et, step 3:", et)
 
#bba
  # [6 terms to insert here..]

    w  = lib.einsum('kJcE,EBIA->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.WVVOV_act) #done 
    w -= lib.einsum('IMAB,kcJM->IJkABc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])], imds.WovOO_act) #done

# P(IJ)

    r = w - w.transpose(1,0,2,3,4,5)

    w  = lib.einsum('kJeB,ecIA->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.WvvOV_act) #done
    w -= lib.einsum('mIcA,JBkm->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.WOVoo_act)  #done
    w += lib.einsum('kcIA,JB->IJkABc', imds.WovOV_act, l1b_active)
    w += lib.einsum('kIcA,JB->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.Fovb_active) 

# P(IJ)P(AB)

    y = w - w.transpose(1,0,2,3,4,5)
    r += y - y.transpose(0,1,2,4,3,5)

    w = lib.einsum('IJAE,EBkc->IJkABc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])], imds.WVVov_act) 
    w -= lib.einsum('kMcB,IAJM->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.WOVOO_act) 

# P(AB) 

    r += w - w.transpose(0,1,2,4,3,5)

    r  += lib.einsum('IAJB,kc->IJkABc', imds.WOVOV_act, l1a_active)
    r  += lib.einsum('IJAB,kc->IJkABc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],act_particle[1])], imds.Fova_active) 

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3bba)*(1.0/4)

    print("value of et, step 4:", et)

#    et *= .25

    return et



def lhs_umpcc_triples_inactive(mcc, t1, t2, l1, l2, t3, eris, act_hole, act_particle): 
    '''    
    t1, t2 amplitudes will be used to build the lhs. later we will replace them by L1 and L2 amplitudes..
    ''' 

    print("shape of t3", len(t3))


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

    l1a,l1b = l1
    l2aa,l2ab,l2bb = l2
    t3aaa,t3bbb,t3baa,t3bba = t3

    nocca, noccb, nvira, nvirb = l2ab.shape
    dtype = numpy.result_type(l1a, l1b, l2aa, l2ab, l2bb)


    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])


    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    imds = make_intermediates_energy(mcc, t1, t2, eris, act_hole, act_particle)

    l1a_active = l1a[numpy.ix_(act_hole[0], act_particle[0])]
    l1b_active = l1b[numpy.ix_(act_hole[1], act_particle[1])]

#aaa
    v = lib.einsum('ebkc,ijae->ijkabc', imds.Wvvov_inact, l2aa[numpy.ix_(act_hole[0], act_hole[0],act_particle[0],inact_particle[0])]) 
    v -= lib.einsum('iajm,mkbc->ijkabc', imds.Wovoo_inact, l2aa[numpy.ix_(inact_hole[0], act_hole[0],act_particle[0],act_particle[0])])

    wd = cyclic_particle(cyclic_hole(v)) 

    et = lib.einsum('ijkabc,ijkabc', wd.conj(), t3aaa)*(1.0/36)

    print("value of et, step 1:", et)

#bbb
    v = lib.einsum('ebkc,ijae->ijkabc', imds.WVVOV_inact, l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],inact_particle[1])]) 
    v -= lib.einsum('iajm,mkbc->ijkabc',imds.WOVOO_inact, l2bb[numpy.ix_(inact_hole[1], act_hole[1],act_particle[1],act_particle[1])])

    wd = cyclic_particle(cyclic_hole(v)) 

    et += lib.einsum('ijkabc,ijkabc', wd.conj(), t3bbb)*(1.0/36)

    print("value of et, step 2:", et)
#baa
  #  [6 terms to insert here..]

    w  = lib.einsum('ebkc,jIeA->IjkAbc', imds.Wvvov_inact, l2ab[numpy.ix_(act_hole[0], act_hole[1],inact_particle[0],act_particle[1])]) #done 
    w -= lib.einsum('mkbc,IAjm->IjkAbc', l2aa[numpy.ix_(inact_hole[0], act_hole[0],act_particle[0],act_particle[0])], imds.WOVoo_inact) #done #check if the defn of W can be changed
    #P(jk)
    r = w - w.transpose(0,2,1,3,4,5)

    w  = lib.einsum('ebIA,jkec->IjkAbc', imds.WvvOV_inact, l2aa[numpy.ix_(act_hole[0], act_hole[0],inact_particle[0],act_particle[0])]) #Done 
    w -= lib.einsum('mIbA,kcjm->IjkAbc', l2ab[numpy.ix_(inact_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.Wovoo_inact) 
    #P(bc)
    r += w - w.transpose(0,1,2,3,5,4)

    w  = lib.einsum('EAkc,jIbE->IjkAbc', imds.WVVov_inact, l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],inact_particle[1])]) #done
    w -= lib.einsum('jMbA,kcIM->IjkAbc', l2ab[numpy.ix_(act_hole[0], inact_hole[1],act_particle[0],act_particle[1])], imds.WovOO_inact) 
    #P(jk)P(bc)

    y = w - w.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    #P(None)

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3baa)*(1.0/4)
    print("value of et, step 3:", et)
 
#bba
  # [6 terms to insert here..]

    w  = lib.einsum('kJcE,EBIA->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],act_particle[0],inact_particle[1])], imds.WVVOV_inact) #done 
    w -= lib.einsum('IMAB,kcJM->IJkABc', l2bb[numpy.ix_(act_hole[1], inact_hole[1],act_particle[1],act_particle[1])], imds.WovOO_inact) #done

# P(IJ)

    r = w - w.transpose(1,0,2,3,4,5)

    w  = lib.einsum('kJeB,ecIA->IJkABc', l2ab[numpy.ix_(act_hole[0], act_hole[1],inact_particle[0],act_particle[1])], imds.WvvOV_inact) #done
    w -= lib.einsum('mIcA,JBkm->IJkABc', l2ab[numpy.ix_(inact_hole[0], act_hole[1],act_particle[0],act_particle[1])], imds.WOVoo_inact)  #done

# P(IJ)P(AB)

    y = w - w.transpose(1,0,2,3,4,5)
    r += y - y.transpose(0,1,2,4,3,5)

    w = lib.einsum('IJAE,EBkc->IJkABc', l2bb[numpy.ix_(act_hole[1], act_hole[1],act_particle[1],inact_particle[1])], imds.WVVov_inact) 
    w -= lib.einsum('kMcB,IAJM->IJkABc', l2ab[numpy.ix_(act_hole[0], inact_hole[1],act_particle[0],act_particle[1])], imds.WOVOO_inact) 

# P(AB) 

    r += w - w.transpose(0,1,2,4,3,5)

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3bba)*(1.0/4)

    print("value of et, step 4:", et)

#    et *= .25

    return et



def pert_doubles(mcc, t1, t2, l1, l2, eris, act_hole, act_particle):
    from pyscf import ao2mo

    l1a,l1b = l1
    l2aa,l2ab,l2bb = l2


    t1a,t1b = l1
    t2aa,t2ab,t2bb = t2


    nocca, noccb, nvira, nvirb = l2ab.shape
    dtype = numpy.result_type(l1a, l1b, l2aa, l2ab, l2bb)

    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    vvvv = ao2mo.restore(1, numpy.asarray(eris.vvvv), nvira)
    VVVV = ao2mo.restore(1, numpy.asarray(eris.VVVV), nvirb)

    vvvv_new = vvvv - vvvv.transpose(0,3,2,1)
    VVVV_new = VVVV - VVVV.transpose(0,3,2,1)
    vvVV = uintermediates._get_vvVV(eris)

    et = 0.0

    #aa
    v  = lib.einsum("acbd, ijcd ->ijab", vvvv_new[numpy.ix_(inact_particle[0], act_particle[0], inact_particle[0], act_particle[0])], t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])])

    et += lib.einsum('ijab,ijab', l2aa[numpy.ix_(act_hole[0], act_hole[0], inact_particle[0], inact_particle[0])], v)*(1.0/4)

    #bb

    v = lib.einsum("acbd, ijcd ->ijab", VVVV_new[numpy.ix_(inact_particle[1], act_particle[1], inact_particle[1], act_particle[1])], t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])])
    et += lib.einsum('ijab,ijab', l2bb[numpy.ix_(act_hole[1], act_hole[1], inact_particle[1], inact_particle[1])], v)*(1.0/4)

    #ab

    v = lib.einsum("acbd, ijcd ->ijab", vvVV[numpy.ix_(inact_particle[0], act_particle[0], inact_particle[1], act_particle[1])], t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])])
    et += lib.einsum('ijab,ijab', l2ab[numpy.ix_(act_hole[0], act_hole[1], inact_particle[0], inact_particle[1])], v)

    print("two-body contribution:", et)

    wovvo  = numpy.asarray(eris.ovvo).transpose(0,2,1,3)
    wovvo -= numpy.asarray(eris.oovv).transpose(0,2,3,1)

    wOVVO  = numpy.asarray(eris.OVVO).transpose(0,2,1,3)
    wOVVO -= numpy.asarray(eris.OOVV).transpose(0,2,3,1)

    woVVo = -numpy.asarray(eris.ooVV).transpose(0,2,3,1)
    woVvO = +numpy.asarray(eris.ovVO).transpose(0,2,1,3)

    wOvvO = -numpy.asarray(eris.OOvv).transpose(0,2,3,1)
    wOvVo = +numpy.asarray(eris.OVvo).transpose(0,2,1,3)

    u2aa  = 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb  = 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab  = lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    v   = u2aa[numpy.ix_(act_hole[0], inact_hole[0], act_particle[0], inact_particle[0])]
    et += lib.einsum('ijab,ijab',l2aa[numpy.ix_(act_hole[0], inact_hole[0], act_particle[0], inact_particle[0])],v)*(1.0/4)  

    v   = u2bb[numpy.ix_(act_hole[1], inact_hole[1], act_particle[1], inact_particle[1])]
    et += lib.einsum('ijab,ijab',l2bb[numpy.ix_(act_hole[1], inact_hole[1], act_particle[1], inact_particle[1])],v)*(1.0/4)  

    v   = u2ab[numpy.ix_(act_hole[0], inact_hole[1], act_particle[0], inact_particle[1])]
    et += lib.einsum('ijab,ijab',l2ab[numpy.ix_(act_hole[0], inact_hole[1], act_particle[0], inact_particle[1])],v) 


    Woooo  = numpy.asarray(eris.oooo).transpose(0,2,1,3)
    WOOOO  = numpy.asarray(eris.OOOO).transpose(0,2,1,3)
    WoOoO  = numpy.asarray(eris.ooOO).transpose(0,2,1,3)

    u2aa  = lib.einsum('mnab,mnij->ijab', t2aa, Woooo*.5)
    u2bb  = lib.einsum('mnab,mnij->ijab', t2bb, WOOOO*.5)
    u2ab  = lib.einsum('mNaB,mNiJ->iJaB', t2ab, WoOoO)

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    v   = u2aa[numpy.ix_(inact_hole[0], inact_hole[0], act_particle[0], act_particle[0])]
    et += lib.einsum('ijab,ijab',l2aa[numpy.ix_(inact_hole[0], inact_hole[0], act_particle[0], act_particle[0])],v)*(1.0/4)  

    v   = u2bb[numpy.ix_(inact_hole[1], inact_hole[1], act_particle[1], act_particle[1])]
    et += lib.einsum('ijab,ijab',l2bb[numpy.ix_(inact_hole[1], inact_hole[1], act_particle[1], act_particle[1])],v)*(1.0/4)  

    v   = u2ab[numpy.ix_(inact_hole[0], inact_hole[1], act_particle[0], act_particle[1])]
    et += lib.einsum('ijab,ijab',l2ab[numpy.ix_(inact_hole[0], inact_hole[1], act_particle[0], act_particle[1])],v) 

    print("two-body contribution:", et)

    ovvv = eris.get_ovvv()  # ovvv = eris.ovvv[p0:p1]
    ovvv = ovvv - ovvv.transpose(0,3,2,1)
    OVVV = eris.get_OVVV()  # OVVV = eris.OVVV[p0:p1]
    OVVV = OVVV - OVVV.transpose(0,3,2,1)

    ovVV = eris.get_ovVV()  # ovVV = eris.ovVV[p0:p1]
    OVvv = eris.get_OVvv()  # OVvv = eris.OVvv[p0:p1]

    u1a  = 0.5*lib.einsum('mief,meaf->ia', t2aa, ovvv)
    u1b  = 0.5*lib.einsum('MIEF,MEAF->IA', t2bb, OVVV)
    u1b += lib.einsum('mIeF,meAF->IA', t2ab, ovVV)
    u1a += lib.einsum('iMfE,MEaf->ia', t2ab, OVvv)

    et += lib.einsum('ia,ia',l1a[numpy.ix_(act_hole[0], inact_particle[0])], u1a[numpy.ix_(act_hole[0], inact_particle[0])]) 
    et += lib.einsum('ia,ia',l1b[numpy.ix_(act_hole[1], inact_particle[1])], u1b[numpy.ix_(act_hole[1], inact_particle[1])]) 


    eris_ovoo = numpy.asarray(eris.ovoo)
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)

    eris_OVOO = numpy.asarray(eris.OVOO)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)

    eris_OVoo = numpy.asarray(eris.OVoo)
    eris_ovOO = numpy.asarray(eris.ovOO)

    u1a  = 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    u1b  = 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)

    et += lib.einsum('ia,ia',l1a[numpy.ix_(inact_hole[0], act_particle[0])], u1a[numpy.ix_(inact_hole[0], act_particle[0])]) 
    et += lib.einsum('ia,ia',l1b[numpy.ix_(inact_hole[1], act_particle[1])], u1b[numpy.ix_(inact_hole[1], act_particle[1])]) 

    return et


def iterative_update_amps_t3(mcc, t1, t2, t3, eris, act_hole, act_particle):
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


    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t3aaa, t3bbb, t3baa, t3bba = t3

    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea, mo_eb = eris.mo_energy
#   eia_full = mo_ea[:nocca,None] - mo_ea[nocca:]#+mcc.level_shift
#   eIA_full = mo_eb[:noccb,None] - mo_eb[noccb:]#+mcc.level_shift

#   eia = eia_full[numpy.ix_(act_hole[0], act_particle[0])]
#   eIA = eIA_full[numpy.ix_(act_hole[1], act_particle[1])]

    inact_particle_a = numpy.delete(numpy.arange(nvira), act_particle[0])
    inact_particle_b = numpy.delete(numpy.arange(nvirb), act_particle[1])

    inact_hole_a = numpy.delete(numpy.arange(nocca), act_hole[0])
    inact_hole_b = numpy.delete(numpy.arange(noccb), act_hole[1])

    inact_particle = (inact_particle_a, inact_particle_b)
    inact_hole = (inact_hole_a, inact_hole_b)

    start = time.time()

    imds = make_intermediates(mcc, t1, t2, eris, act_hole, act_particle)

    end = time.time()

    wvvvo_t3, wvvVO_t3, wVVvo_t3, wVVVO_t3, woovo_t3, wOOVO_t3, wooVO_t3, wOOvo_t3 = get_t3_to_imds(mcc, t3, t1, eris, act_hole, act_particle)

    print("time to make intermediates", end-start)


    imds.Wvvvo_act += wvvvo_t3
    imds.WvvVO_act += wvvVO_t3
    imds.WVVvo_act += wVVvo_t3
    imds.WVVVO_act += wVVVO_t3

    imds.Woovo_act += woovo_t3
    imds.WOOVO_act += wOOVO_t3
    imds.WooVO_act += wooVO_t3
    imds.WOOvo_act += wOOvo_t3

    wvvvo_t3 = wvvVO_t3 = wVVvo_t3 = wVVVO_t3 = None

    woovo_t3 = wOOVO_t3 = wooVO_t3 = wOOvo_t3 = None   

    foo_act = eris.focka[numpy.ix_(act_hole[0], act_hole[0])]
    fOO_act = eris.fockb[numpy.ix_(act_hole[1], act_hole[1])]

    fvv_act = eris.focka[numpy.ix_(nocca+act_particle[0], nocca+act_particle[0])]
    fVV_act = eris.fockb[numpy.ix_(noccb+act_particle[1], noccb+act_particle[1])]


#   mo_ea_o = numpy.diag(imds.Foo_act)
#   mo_ea_v = numpy.diag(imds.Fvv_act)+mcc.level_shift
#   mo_eb_o = numpy.diag(imds.FOO_act)
#   mo_eb_v = numpy.diag(imds.FVV_act)+mcc.level_shift


    mo_ea_o = numpy.diag(foo_act)
    mo_ea_v = numpy.diag(fvv_act)+mcc.level_shift
    mo_eb_o = numpy.diag(fOO_act)
    mo_eb_v = numpy.diag(fVV_act)+mcc.level_shift

    foo_act = fOO_act = fvv_act = fVV_act = None

    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    # aaa
    d3aaa_active = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    start = time.time()

    x  = lib.einsum('ijae,beck->ijkabc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], imds.Wvvvo_act)
    x -= lib.einsum('imab,mjck->ijkabc', t2aa[numpy.ix_(act_hole[0], numpy.arange(nocca), act_particle[0], act_particle[0])], imds.Woovo_act)

    end = time.time()
    print("time to make aaa contribution to t3", end-start)

    start = time.time()
    u3aaa = cyclic_hole(cyclic_particle(x)) 
    end = time.time()
    print("time to make aaa permutation", end-start)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3aaa, imds.Fvv_act)
    u3aaa += cyclic_particle(x)
    x  = -lib.einsum('mjkabc,mi->ijkabc', t3aaa, imds.Foo_act)
    u3aaa += cyclic_hole(x)


    # bbb
    d3bbb_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)

    x = lib.einsum('ijae,beck->ijkabc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))], imds.WVVVO_act)
    x -= lib.einsum('imab,mjck->ijkabc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])], imds.WOOVO_act)

    u3bbb = cyclic_particle(cyclic_hole(x)) 

    x  = lib.einsum('ijkabe,ce->ijkabc', t3bbb, imds.FVV_act)
    u3bbb += cyclic_particle(x)
    x = -lib.einsum('mjkabc,mi->ijkabc', t3bbb, imds.FOO_act)
    u3bbb += cyclic_hole(x)


    # baa
    d3baa_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    u3baa  = lib.einsum('jIeA,beck->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])], numpy.asarray(imds.Wvvvo_act))    # 2

   #P(jk)
    r = u3baa - u3baa.transpose(0,2,1,3,4,5)

    u3baa = lib.einsum('jIbE,AEck->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))], numpy.asarray(imds.WVVvo_act))     # 2

   #P(bc)p(jk)
    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = lib.einsum('jkbe,ceAI->IjkAbc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], numpy.asarray(imds.WvvVO_act))
    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mIbA,mjck->IjkAbc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.Woovo_act)) 

    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('jMbA,MIck->IjkAbc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])], numpy.asarray(imds.WOOvo_act)) 

   #P(bc)P(jk)

    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mjcb,mkAI->IjkAbc', t2aa[numpy.ix_(numpy.arange(nocca), act_hole[0], act_particle[0], act_particle[0])], numpy.asarray(imds.WooVO_act))

    #P(jk) 

    r += u3baa - u3baa.transpose(0,2,1,3,4,5)

# fish out the energy contribution

    u3baa = lib.einsum('IjkAec,be->IjkAbc', t3baa, imds.Fvv_act)  
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    r += lib.einsum('IjkEbc,AE->IjkAbc', t3baa, imds.FVV_act)  

    u3baa = -lib.einsum('ImkAbc,mj->IjkAbc', t3baa, imds.Foo_act)  
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)
    r -= lib.einsum('MjkAbc,MI->IjkAbc', t3baa, imds.FOO_act)  

    # bba

    d3bba_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eia)

    u3bba  = lib.einsum('IJAE,BEck->IJkABc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))], numpy.asarray(imds.WVVvo_act)) 
#  P(AB)

    v = u3bba - u3bba.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('kJcE,BEAI->IJkABc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))], numpy.asarray(imds.WVVVO_act))  
#  P(IJ) 
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = lib.einsum('kIeA,ceBJ->IJkABc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])], numpy.asarray(imds.WvvVO_act))
 # P(IJ)P(AB)  

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = -lib.einsum('IMAB,MJck->IJkABc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])], numpy.asarray(imds.WOOvo_act)) 
#P(IJ)

    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = -lib.einsum('kMcB,MJAI->IJkABc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])], numpy.asarray(imds.WOOVO_act)) 

#P(AB)
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)
    u3bba = -lib.einsum('mJcB,mkAI->IJkABc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.WooVO_act))

#P(IJ)P(AB)

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)


    u3bba = lib.einsum('IJkEBc,AE->IJkABc', t3bba, imds.FVV_act)  
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)

    v += lib.einsum('IJkABe,ce->IJkABc',t3bba, imds.Fvv_act)  

    u3bba = -lib.einsum('MJkABc,MI->IJkABc', t3bba, imds.FOO_act)  
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)
    v -= lib.einsum('IJmABc,mk->IJkABc',t3bba, imds.Foo_act)  

#Now add symmetrization of the u3 tensors:

    start = time.time()
    wvvvv, wVVVV, wvvVV = get_vvvv_imds(mcc, t1, t2, eris, act_hole, act_particle)

    end = time.time()

    print("time to make intermediate wvvvv", end-start)

    oooo, ooOO, OOoo, OOOO = uintermediates.Woooo(t1, t2, eris)

    woooo = oooo[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_hole[0])]
    wOOOO = OOOO[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_hole[1])]
    wooOO = ooOO[numpy.ix_(act_hole[0], act_hole[0], act_hole[1], act_hole[1])]

#   oovv, OOVV, OOvv, ooVV, ovVO, OVvo = uintermediates.Woovv(t1, t2, eris, factor=1.0)
    ovvo, ovVO, OVvo, OVVO, ooVV, OOvv = uintermediates.Wovvo_hp_ring(t1, t2, eris, factor=1.0)

    wovvo = ovvo[numpy.ix_(act_hole[0], act_particle[0], act_particle[0], act_hole[0])]
    wOVVO = OVVO[numpy.ix_(act_hole[1], act_particle[1], act_particle[1], act_hole[1])]
    wooVV = ooVV[numpy.ix_(act_hole[0], act_hole[0], act_particle[1], act_particle[1])]
    wOOvv = OOvv[numpy.ix_(act_hole[1], act_hole[1], act_particle[0], act_particle[0])]
    wovVO = ovVO[numpy.ix_(act_hole[0], act_particle[0],  act_particle[1], act_hole[1])]
    wOVvo = OVvo[numpy.ix_(act_hole[1], act_particle[1],  act_particle[0], act_hole[0])]

# Now evaluate vvvv contribution to the residue of T3: 

    start = time.time()
    tmp = lib.einsum('ijkefc,aebf->ijkabc', t3aaa, wvvvv) * .5 #P(c/ab)
    u3aaa += cyclic_particle(tmp) 

    tmp = lib.einsum('IJKEFC,AEBF->IJKABC', t3bbb, wVVVV) * .5

    u3bbb += cyclic_particle(tmp)

    tmp = lib.einsum('IjkEfc,bfAE->IjkAbc', t3baa, wvvVV) #P(bc)

    u3baa = tmp - tmp.transpose(0,1,2,3,5,4)
    u3bba = lib.einsum('IJkEFc,BFAE->IJkABc', t3bba, wVVVV) * .5

    u3baa += lib.einsum('IjkAef,becf->IjkAbc', t3baa, wvvvv) * .5
    tmp = lib.einsum('IJkAEf,cfBE->IJkABc', t3bba, wvvVV) #P(AB)

    u3bba += tmp - tmp.transpose(0,1,2,4,3,5) 

    wvvvv = wVVVV = wvvVV = None

    end = time.time()

    print("time to make most expensive N8 step:", end-start)

# Now evaluate oooo contribution to the residue of T3: 

    tmp = lib.einsum('imlabc,mjlk->ijkabc', t3aaa, woooo) * .5 # P(i/jk)
    u3aaa += cyclic_hole(tmp)

    tmp = lib.einsum('imlabc,mjlk->ijkabc', t3bbb, wOOOO) * .5 # P(i/jk)
    u3bbb += cyclic_hole(tmp)

    u3baa += lib.einsum('ImlAbc,mjlk->IjkAbc', t3baa, woooo) * .5 # 
    tmp = lib.einsum('IMlABc,lkMJ->IJkABc', t3bba, wooOO) # P(ij)
    u3bba += tmp - tmp.transpose(1,0,2,3,4,5)

    tmp = lib.einsum('MnkAbc,njMI->IjkAbc', t3baa, wooOO) # P(jk)
    u3baa += tmp - tmp.transpose(0,2,1,3,4,5)

    u3bba += lib.einsum('MNkABc,NJMI->IJkABc', t3bba, wOOOO) * .5 # 

# Evaluate (h-p) like terms 

#   tmp  = -lib.einsum('imkebc,mjae->ijkabc', t3aaa,woovv) #P(j/ik)P(a/bc)
#   tmp += lib.einsum('MikEac,MEbj->ijkabc', t3baa,wOVvo) #P(i/jk)P(a/bc)
#   u3aaa += cyclic_hole(cyclic_particle(tmp)) 

#   tmp  = -lib.einsum('imkebc,mjae->ijkabc', t3bbb,wOOVV) #P(j/ik)P(a/bc)
#   tmp += lib.einsum('KImCAe,meBJ->IJKABC', t3bba,wovVO) #P(I/JK)P(A/BC)
#   u3bbb += cyclic_hole(cyclic_particle(tmp)) 

#   tmp = -lib.einsum('ImkAbe,mjce->IjkAbc', t3baa,woovv) #P(jk,bc)
#   tmp += lib.einsum('IMkAEc,MEbj->IjkAbc', t3bba,wOVvo) 
#   tmp1 = tmp - tmp.transpose(0,2,1,3,4,5)
#   u3baa += tmp1 - tmp1.transpose(0,1,2,3,5,4)

#   tmp = -lib.einsum('ImkEbc,mjAE->IjkAbc', t3baa,wooVV) #p(jk)
#   u3baa += tmp - tmp.transpose(0,2,1,3,4,5) 

#   tmp = -lib.einsum('MjkAec,MIbe->IjkAbc', t3baa,wOOvv) #p(bc)
#   u3baa += tmp - tmp.transpose(0,1,2,3,5,4) 

#   u3baa += lib.einsum('mjkebc,meAI->IjkAbc', t3aaa,wovVO)
#   u3baa -= lib.einsum('MjkEbc,MIAE->IjkAbc', t3baa,wOOVV)

#   tmp = -lib.einsum('IMkEBc,MJAE->IJkABc', t3bba,wOOVV) #P(IJ,AB)
#   tmp += lib.einsum('JmkBec,meAI->IJkABc', t3baa,wovVO)#P(IJ)P(AB) 
#   tmp1 = tmp - tmp.transpose(1,0,2,3,4,5)
#   u3bba += tmp1 - tmp1.transpose(0,1,2,4,3,5)

#   tmp = -lib.einsum('IMkABe,MJce->IJkABc', t3bba,wOOvv) #P(IJ)
#   u3bba += tmp - tmp.transpose(1,0,2,3,4,5)

#   u3bba += lib.einsum('IJMABE,MEck->IJkABc', t3bbb,wOVvo) 
#   u3bba -= lib.einsum('IJmABe,mkce->IJkABc', t3bba,woovv)

#   tmp = -lib.einsum('IJmAEc,mkBE->IJkABc', t3bba,wooVV) #P(AB) 
#   u3bba += tmp - tmp.transpose(0,1,2,4,3,5)

#   woovv = wOOVV = wOVvo = wovVO = wooVV = wOOvv = None

#   print("norm of u3baa", numpy.linalg.norm(u3baa))
#   print("norm of u3bba", numpy.linalg.norm(u3bba))


    tmp = lib.einsum('mikeac,mebj->ijkabc', t3aaa,wovvo) #P(i/jk)P(a/bc)
    tmp += lib.einsum('MikEac,MEbj->ijkabc', t3baa,wOVvo) #P(i/jk)P(a/bc)
    u3aaa += cyclic_hole(cyclic_particle(tmp)) 

    tmp = lib.einsum('mikeac,mebj->ijkabc', t3bbb,wOVVO) #P(i/jk)P(a/bc)
    tmp += lib.einsum('KImCAe,meBJ->IJKABC', t3bba,wovVO) #P(I/JK)P(A/BC)
    u3bbb += cyclic_hole(cyclic_particle(tmp)) 

    tmp = lib.einsum('ImkAec,mebj->IjkAbc', t3baa,wovvo) #P(jk,bc)
    tmp += lib.einsum('IMkAEc,MEbj->IjkAbc', t3bba,wOVvo) 
    tmp1 = tmp - tmp.transpose(0,2,1,3,4,5)
    u3baa += tmp1 - tmp1.transpose(0,1,2,3,5,4)

#####

    tmp = -lib.einsum('ImkEbc,mjAE->IjkAbc', t3baa,wooVV) #p(jk)
    u3baa += tmp - tmp.transpose(0,2,1,3,4,5) 

    tmp = -lib.einsum('MjkAec,MIbe->IjkAbc', t3baa,wOOvv) #p(bc)
    u3baa += tmp - tmp.transpose(0,1,2,3,5,4) 

    u3baa += lib.einsum('mjkebc,meAI->IjkAbc', t3aaa,wovVO)

    u3baa += lib.einsum('MjkEbc,MEAI->IjkAbc', t3baa,wOVVO)

####
    tmp = lib.einsum('MIkEAc,MEBJ->IJkABc', t3bba,wOVVO) #P(IJ,AB)
    tmp += lib.einsum('JmkBec,meAI->IJkABc', t3baa,wovVO)#P(IJ)P(AB) 
    tmp1 = tmp - tmp.transpose(1,0,2,3,4,5)
    u3bba += tmp1 - tmp1.transpose(0,1,2,4,3,5)

    tmp = -lib.einsum('IMkABe,MJce->IJkABc', t3bba,wOOvv) #P(IJ)
    u3bba += tmp - tmp.transpose(1,0,2,3,4,5)

    u3bba += lib.einsum('IJMABE,MEck->IJkABc', t3bbb,wOVvo) 
    u3bba += lib.einsum('IJmABe,meck->IJkABc', t3bba,wovvo)

    tmp = -lib.einsum('IJmAEc,mkBE->IJkABc', t3bba,wooVV) #P(AB) 
    u3bba += tmp - tmp.transpose(0,1,2,4,3,5)
   
    v += u3bba 
    r += u3baa 

# divide by denominator..

    u3aaa /=d3aaa_active
    u3bbb /=d3bbb_active
    u3bba = v/d3bba_active
    u3baa = r/d3baa_active

#   print("norm of u3aaa", numpy.linalg.norm(u3aaa))
#   print("norm of u3bbb", numpy.linalg.norm(u3bbb))
#   print("norm of u3baa", numpy.linalg.norm(u3baa))
#   print("norm of u3bba", numpy.linalg.norm(u3bba))

#                    (1)                        ~        (1)
# contribution of T_3   to the residue of T_2: [F_ov, T_3   ]
     
    u2aa  = lib.einsum('ijmabe,me->ijab', t3aaa, imds.Fov_act)
    u2aa += lib.einsum('MijEab,ME->ijab', t3baa, imds.FOV_act)

    u2bb = lib.einsum('ijmabe,me->ijab', t3bbb, imds.FOV_act)
    u2bb += lib.einsum('IJmABe,me->IJAB', t3bba, imds.Fov_act)

    u2ab = lib.einsum('MIjEAb,ME->jIbA', t3bba, imds.FOV_act)
    u2ab += lib.einsum('IjmAbe,me->jIbA', t3baa, imds.Fov_act)

#                    (1)                                  (1)
# contribution of T_3   to the residue of T_2: [w_vvov, T_3   ]

    res  = lib.einsum('ijmace,bcme->ijab', t3aaa, imds.Wvvov_act)*0.5
    res  += lib.einsum('MijEac,bcME->ijab', t3baa, imds.WvvOV_act)

    u2aa += res - res.transpose(0,1,3,2) 


    res   = lib.einsum('IJMACE,BCME->IJAB', t3bbb, imds.WVVOV_act)*0.5
    res  += lib.einsum('IJmACe,BCme->IJAB', t3bba, imds.WVVov_act)

    u2bb += res - res.transpose(0,1,3,2) 

    u2ab += lib.einsum('MIjEAc,bcME->jIbA', t3bba, imds.WvvOV_act)
    u2ab += lib.einsum('JimCae,BCme->iJaB', t3baa, imds.WVVov_act)
    u2ab += lib.einsum('IjmAce,bcme->jIbA', t3baa, imds.Wvvov_act)*0.5
    u2ab += lib.einsum('MJiECa,BCME->iJaB', t3bba, imds.WVVOV_act)*0.5

#                    (1)                                  (1)
# contribution of T_3   to the residue of T_2: [w_ovoo, T_3   ]

    res   = -lib.einsum('inmabe,menj->ijab', t3aaa, imds.Wovoo_act)*0.5
    res  -= lib.einsum('MniEba,MEnj->ijab', t3baa, imds.WOVoo_act)

    u2aa += res - res.transpose(1,0,2,3)

    res   = -lib.einsum('INMABE,MENJ->IJAB', t3bbb, imds.WOVOO_act)*0.5
    res  -= lib.einsum('INmABe,meNJ->IJAB', t3bba, imds.WovOO_act)

    u2bb += res - res.transpose(1,0,2,3) 

    u2ab -= lib.einsum('MInEAb,MEnj->jIbA', t3bba, imds.WOVoo_act)
    u2ab -= lib.einsum('NimBae,meNJ->iJaB', t3baa, imds.WovOO_act)
    u2ab -= lib.einsum('InmAbe,menj->jIbA', t3baa, imds.Wovoo_act)*0.5
    u2ab -= lib.einsum('MNiEBa,MENJ->iJaB', t3bba, imds.WOVOO_act)*0.5

    print("norm of u2aa", numpy.linalg.norm(u2aa))
    print("norm of u2bb", numpy.linalg.norm(u2bb))


    u2_active = u2aa, u2ab, u2bb  

#                    (1)                                  (1)
# contribution of T_3   to the residue of T_1: [w_ovov, T_3   ]
     
    u1a  = lib.einsum('ijmabe,jbme->ia', t3aaa, imds.Wovov_act)*0.25
    u1a += lib.einsum('MijEab,jbME->ia', t3baa, imds.WovOV_act)
    u1a += lib.einsum('MIjEAb,MEIA->jb', t3bba, imds.WOVOV_act)*0.25

    u1b = lib.einsum('ijmabe,jbme->ia', t3bbb, imds.WOVOV_act)*0.25
    u1b += lib.einsum('IJmABe,meJB->IA',t3bba, imds.WovOV_act)
    u1b += lib.einsum('IjmAbe,jbme->IA',t3baa, imds.Wovov_act)*0.25

    print("norm of u1a", numpy.linalg.norm(u1a))
    print("norm of u1b", numpy.linalg.norm(u1b))

    u1_active = u1a, u1b  

    u3aaa  += t3aaa
    u3bbb  += t3bbb
    u3bba  += t3bba
    u3baa  += t3baa

    t3new = u3aaa, u3bbb, u3baa, u3bba
    
    return t3new, u2_active, u1_active 



def iterative_update_amps_ccsdt3(mcc, t1, t2, t3, eris, act_hole, act_particle):
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


    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t3aaa, t3bbb, t3baa, t3bba = t3

    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea, mo_eb = eris.mo_energy


    start = time.time()

    imds = make_intermediates(mcc, t1, t2, eris, act_hole, act_particle)

    end = time.time()

    print("time to make intermediates", end-start)

    mo_ea_o = numpy.diag(imds.Foo_act)
    mo_ea_v = numpy.diag(imds.Fvv_act)+mcc.level_shift
    mo_eb_o = numpy.diag(imds.FOO_act)
    mo_eb_v = numpy.diag(imds.FVV_act)+mcc.level_shift

    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    # aaa
    d3aaa_active = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    start = time.time()

    x  = lib.einsum('ijae,beck->ijkabc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], imds.Wvvvo_act)
    x -= lib.einsum('imab,mjck->ijkabc', t2aa[numpy.ix_(act_hole[0], numpy.arange(nocca), act_particle[0], act_particle[0])], imds.Woovo_act)

    end = time.time()
    print("time to make aaa contribution to t3", end-start)

    start = time.time()
    u3aaa = cyclic_hole(cyclic_particle(x)) 
    end = time.time()
    print("time to make aaa permutation", end-start)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3aaa, imds.Fvv_act)
    u3aaa += cyclic_particle(x)
    x  = -lib.einsum('mjkabc,mi->ijkabc', t3aaa, imds.Foo_act)
    u3aaa += cyclic_hole(x)


    # bbb
    d3bbb_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)

    x = lib.einsum('ijae,beck->ijkabc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))], imds.WVVVO_act)
    x -= lib.einsum('imab,mjck->ijkabc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])], imds.WOOVO_act)

    u3bbb = cyclic_particle(cyclic_hole(x)) 

    x  = lib.einsum('ijkabe,ce->ijkabc', t3bbb, imds.FVV_act)
    u3bbb += cyclic_particle(x)
    x = -lib.einsum('mjkabc,mi->ijkabc', t3bbb, imds.FOO_act)
    u3bbb += cyclic_hole(x)


    # baa
    d3baa_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    u3baa  = lib.einsum('jIeA,beck->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])], numpy.asarray(imds.Wvvvo_act))    # 2

   #P(jk)
    r = u3baa - u3baa.transpose(0,2,1,3,4,5)

    u3baa = lib.einsum('jIbE,AEck->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))], numpy.asarray(imds.WVVvo_act))     # 2

   #P(bc)p(jk)
    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = lib.einsum('jkbe,ceAI->IjkAbc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], numpy.asarray(imds.WvvVO_act))
    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mIbA,mjck->IjkAbc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.Woovo_act)) 

    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('jMbA,MIck->IjkAbc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])], numpy.asarray(imds.WOOvo_act)) 

   #P(bc)P(jk)

    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mjcb,mkAI->IjkAbc', t2aa[numpy.ix_(numpy.arange(nocca), act_hole[0], act_particle[0], act_particle[0])], numpy.asarray(imds.WooVO_act))

    #P(jk) 

    r += u3baa - u3baa.transpose(0,2,1,3,4,5)

# fish out the energy contribution

    u3baa = lib.einsum('IjkAec,be->IjkAbc', t3baa, imds.Fvv_act)  
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    r += lib.einsum('IjkEbc,AE->IjkAbc', t3baa, imds.FVV_act)  

    u3baa = -lib.einsum('ImkAbc,mj->IjkAbc', t3baa, imds.Foo_act)  
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)
    r -= lib.einsum('MjkAbc,MI->IjkAbc', t3baa, imds.FOO_act)  

    # bba

    d3bba_active = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eia)

    u3bba  = lib.einsum('IJAE,BEck->IJkABc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))], numpy.asarray(imds.WVVvo_act)) 
#  P(AB)

    v = u3bba - u3bba.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('kJcE,BEAI->IJkABc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))], numpy.asarray(imds.WVVVO_act))  
#  P(IJ) 
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = lib.einsum('kIeA,ceBJ->IJkABc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])], numpy.asarray(imds.WvvVO_act))
 # P(IJ)P(AB)  

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = -lib.einsum('IMAB,MJck->IJkABc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])], numpy.asarray(imds.WOOvo_act)) 
#P(IJ)

    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = -lib.einsum('kMcB,MJAI->IJkABc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])], numpy.asarray(imds.WOOVO_act)) 

#P(AB)
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)
    u3bba = -lib.einsum('mJcB,mkAI->IJkABc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])], numpy.asarray(imds.WooVO_act))

#P(IJ)P(AB)

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)


    u3bba = lib.einsum('IJkEBc,AE->IJkABc', t3bba, imds.FVV_act)  
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)

    v += lib.einsum('IJkABe,ce->IJkABc',t3bba, imds.Fvv_act)  

    u3bba = -lib.einsum('MJkABc,MI->IJkABc', t3bba, imds.FOO_act)  
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)
    v -= lib.einsum('IJmABc,mk->IJkABc',t3bba, imds.Foo_act)  

# divide by denominator..

    u3aaa /=d3aaa_active
    u3bbb /=d3bbb_active
    u3bba = v/d3bba_active
    u3baa = r/d3baa_active

#                    (1)                        ~        (1)
# contribution of T_3   to the residue of T_2: [F_ov, T_3   ]
     
    u2aa  = lib.einsum('ijmabe,me->ijab', t3aaa, imds.Fov_act)
    u2aa += lib.einsum('MijEab,ME->ijab', t3baa, imds.FOV_act)

    u2bb = lib.einsum('ijmabe,me->ijab', t3bbb, imds.FOV_act)
    u2bb += lib.einsum('IJmABe,me->IJAB', t3bba, imds.Fov_act)

    u2ab = lib.einsum('MIjEAb,ME->jIbA', t3bba, imds.FOV_act)
    u2ab += lib.einsum('IjmAbe,me->jIbA', t3baa, imds.Fov_act)

#                    (1)                                  (1)
# contribution of T_3   to the residue of T_2: [w_vvov, T_3   ]

    res  = lib.einsum('ijmace,bcme->ijab', t3aaa, imds.Wvvov_act)*0.5
    res  += lib.einsum('MijEac,bcME->ijab', t3baa, imds.WvvOV_act)

    u2aa += res - res.transpose(0,1,3,2) 


    res   = lib.einsum('IJMACE,BCME->IJAB', t3bbb, imds.WVVOV_act)*0.5
    res  += lib.einsum('IJmACe,BCme->IJAB', t3bba, imds.WVVov_act)

    u2bb += res - res.transpose(0,1,3,2) 

    u2ab += lib.einsum('MIjEAc,bcME->jIbA', t3bba, imds.WvvOV_act)
    u2ab += lib.einsum('JimCae,BCme->iJaB', t3baa, imds.WVVov_act)
    u2ab += lib.einsum('IjmAce,bcme->jIbA', t3baa, imds.Wvvov_act)*0.5
    u2ab += lib.einsum('MJiECa,BCME->iJaB', t3bba, imds.WVVOV_act)*0.5

#                    (1)                                  (1)
# contribution of T_3   to the residue of T_2: [w_ovoo, T_3   ]

    res   = -lib.einsum('inmabe,menj->ijab', t3aaa, imds.Wovoo_act)*0.5
    res  -= lib.einsum('MniEba,MEnj->ijab', t3baa, imds.WOVoo_act)

    u2aa += res - res.transpose(1,0,2,3)

    res   = -lib.einsum('INMABE,MENJ->IJAB', t3bbb, imds.WOVOO_act)*0.5
    res  -= lib.einsum('INmABe,meNJ->IJAB', t3bba, imds.WovOO_act)

    u2bb += res - res.transpose(1,0,2,3) 

    u2ab -= lib.einsum('MInEAb,MEnj->jIbA', t3bba, imds.WOVoo_act)
    u2ab -= lib.einsum('NimBae,meNJ->iJaB', t3baa, imds.WovOO_act)
    u2ab -= lib.einsum('InmAbe,menj->jIbA', t3baa, imds.Wovoo_act)*0.5
    u2ab -= lib.einsum('MNiEBa,MENJ->iJaB', t3bba, imds.WOVOO_act)*0.5

    print("norm of u2aa", numpy.linalg.norm(u2aa))
    print("norm of u2bb", numpy.linalg.norm(u2bb))


    u2_active = u2aa, u2ab, u2bb  

#                    (1)                                  (1)
# contribution of T_3   to the residue of T_1: [w_ovov, T_3   ]
     
    u1a  = lib.einsum('ijmabe,jbme->ia', t3aaa, imds.Wovov_act)*0.25
    u1a += lib.einsum('MijEab,jbME->ia', t3baa, imds.WovOV_act)
    u1a += lib.einsum('MIjEAb,MEIA->jb', t3bba, imds.WOVOV_act)*0.25

    u1b = lib.einsum('ijmabe,jbme->ia', t3bbb, imds.WOVOV_act)*0.25
    u1b += lib.einsum('IJmABe,meJB->IA',t3bba, imds.WovOV_act)
    u1b += lib.einsum('IjmAbe,jbme->IA',t3baa, imds.Wovov_act)*0.25

    print("norm of u1a", numpy.linalg.norm(u1a))
    print("norm of u1b", numpy.linalg.norm(u1b))

    u1_active = u1a, u1b  

    u3aaa  += t3aaa
    u3bbb  += t3bbb
    u3bba  += t3bba
    u3baa  += t3baa

    t3new = u3aaa, u3bbb, u3baa, u3bba

    return t3new, u2_active, u1_active 


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
