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

def kernel(mycc, eris, t1, t2, l1, l2, act_hole, act_particle): 

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

    mo_coeff = eris.mo_coeff
#generate all transformed integrals:

    ints = _make_4c_integrals(mycc, eris, t1, t2, mo_coeff)

    t3 = update_amps(ints)

    t3aaa, t3bbb, t3baa, t3bba = t3 

    t3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] = 0.0 

    t3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] = 0.0 

    t3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] = 0.0 

    t3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] = 0.0 

    t3 = t3aaa, t3bbb, t3baa, t3bba

    t1 = (ints.t1a, ints.t1b) 

    t2 = (ints.t2aa, ints.t2ab, ints.t2bb) 

    et = lhs_env_triples(ints, t1, t2, t3) 

    return et


def update_amps(ints, t3, act_hole, act_particle):
    '''Update non-canonical MP2 amplitudes'''

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

    mo_ea_o = numpy.diag(ints.Foo)
    mo_ea_v = numpy.diag(ints.Fvv)
    mo_eb_o = numpy.diag(ints.FOO)
    mo_eb_v = numpy.diag(ints.FVV)

    t3aaa, t3bbb, t3baa, t3bba = t3

    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    # aaa
    d3aaa  = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    start = time.time()

    x  = lib.einsum('ijae,beck->ijkabc', ints.t2aa, ints.Wvvvo )
    x -= lib.einsum('imab,mjck->ijkabc', ints.t2aa, ints.Woovo )

    end = time.time()
    print("time to make aaa contribution to t3", end-start)

    start = time.time()
    u3aaa = cyclic_hole(cyclic_particle(x)) 
    end = time.time()
    print("time to make aaa permutation", end-start)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3aaa, ints.Fvv)
    u3aaa += cyclic_particle(x)
    x  = -lib.einsum('mjkabc,mi->ijkabc', t3aaa, ints.Foo)
    u3aaa += cyclic_hole(x)

    # bbb
    d3bbb = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)

    x = lib.einsum('ijae,beck->ijkabc', ints.t2bb, ints.WVVVO )
    x -= lib.einsum('imab,mjck->ijkabc',ints.t2bb, ints.WOOVO )

    u3bbb = cyclic_particle(cyclic_hole(x)) 

    x  = lib.einsum('ijkabe,ce->ijkabc', t3bbb, ints.FVV)
    u3bbb += cyclic_particle(x)
    x = -lib.einsum('mjkabc,mi->ijkabc', t3bbb, ints.FOO)
    u3bbb += cyclic_hole(x)

    # baa
    d3baa = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    u3baa  = lib.einsum('jIeA,beck->IjkAbc', ints.t2ab, ints.Wvvvo)    # 2

   #P(jk)
    r = u3baa - u3baa.transpose(0,2,1,3,4,5)

    u3baa = lib.einsum('jIbE,AEck->IjkAbc', ints.t2ab, ints.WVVvo)     # 2

   #P(bc)p(jk)
    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = lib.einsum('jkbe,ceAI->IjkAbc', ints.t2aa, ints.WvvVO)
    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mIbA,mjck->IjkAbc', ints.t2ab, ints.Woovo) 

    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('jMbA,MIck->IjkAbc', ints.t2ab, ints.WOOvo ) 

   #P(bc)P(jk)

    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mjcb,mkAI->IjkAbc', ints.t2aa, ints.WooVO )

    #P(jk) 
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)


    u3baa = lib.einsum('IjkAec,be->IjkAbc', t3baa, ints.Fvv)  
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    r += lib.einsum('IjkEbc,AE->IjkAbc', t3baa, ints.FVV)  

    u3baa = -lib.einsum('ImkAbc,mj->IjkAbc', t3baa, ints.Foo)  
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)
    r -= lib.einsum('MjkAbc,MI->IjkAbc', t3baa, ints.FOO)  

    # bba

    d3bba = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eia)

    u3bba  = lib.einsum('IJAE,BEck->IJkABc', ints.t2bb, ints.WVVvo ) 
#  P(AB)

    v = u3bba - u3bba.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('kJcE,BEAI->IJkABc', ints.t2ab, ints.WVVVO )  
#  P(IJ) 
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = lib.einsum('kIeA,ceBJ->IJkABc', ints.t2ab, ints.WvvVO )
 # P(IJ)P(AB)  

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = -lib.einsum('IMAB,MJck->IJkABc', ints.t2bb, ints.WOOvo ) 
#P(IJ)

    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = -lib.einsum('kMcB,MJAI->IJkABc', ints.t2ab, ints.WOOVO ) 

#P(AB)
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)
    u3bba = -lib.einsum('mJcB,mkAI->IJkABc', ints.t2ab, ints.WooVO )

#P(IJ)P(AB)

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('IJkEBc,AE->IJkABc', t3bba, ints.FVV)  
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)

    v += lib.einsum('IJkABe,ce->IJkABc',t3bba, ints.Fvv)  

    u3bba = -lib.einsum('MJkABc,MI->IJkABc', t3bba, ints.FOO)  
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)
    v -= lib.einsum('IJmABc,mk->IJkABc',t3bba, ints.Foo)  

# divide by denominator..
    u3aaa /=d3aaa 
    u3bbb /=d3bbb 
    u3bba = v/d3bba 
    u3baa = r/d3baa 

    u3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] = 0.0 
    u3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] = 0.0 
    u3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] = 0.0 
    u3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] = 0.0 

    u3aaa  += t3aaa
    u3bbb  += t3bbb
    u3bba  += t3bba
    u3baa  += t3baa

    t3new = u3aaa, u3bbb, u3baa, u3bba

    return t3new 


def build_w(ints):
    '''Update non-canonical MP2 amplitudes'''

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

    # aaa

    start = time.time()
    x  = lib.einsum('ijae,beck->ijkabc', ints.t2aa, ints.Wvvvo )
    x -= lib.einsum('imab,mjck->ijkabc', ints.t2aa, ints.Woovo )

    end = time.time()
    print("time to make aaa contribution to t3", end-start)

    start = time.time()
    u3aaa = cyclic_hole(cyclic_particle(x)) 
    end = time.time()
    print("time to make aaa permutation", end-start)

    # bbb

    x = lib.einsum('ijae,beck->ijkabc', ints.t2bb, ints.WVVVO )
    x -= lib.einsum('imab,mjck->ijkabc',ints.t2bb, ints.WOOVO )

    u3bbb = cyclic_particle(cyclic_hole(x)) 

    # baa
    u3baa  = lib.einsum('jIeA,beck->IjkAbc', ints.t2ab, ints.Wvvvo)    # 2

   #P(jk)
    r = u3baa - u3baa.transpose(0,2,1,3,4,5)

    u3baa = lib.einsum('jIbE,AEck->IjkAbc', ints.t2ab, ints.WVVvo)     # 2

   #P(bc)p(jk)
    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = lib.einsum('jkbe,ceAI->IjkAbc', ints.t2aa, ints.WvvVO)
    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mIbA,mjck->IjkAbc', ints.t2ab, ints.Woovo) 

    #P(bc)
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('jMbA,MIck->IjkAbc', ints.t2ab, ints.WOOvo ) 

   #P(bc)P(jk)

    y = u3baa - u3baa.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    u3baa = -lib.einsum('mjcb,mkAI->IjkAbc', ints.t2aa, ints.WooVO )

    #P(jk) 
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)

    # bba

    u3bba  = lib.einsum('IJAE,BEck->IJkABc', ints.t2bb, ints.WVVvo ) 
#  P(AB)
    v = u3bba - u3bba.transpose(0,1,2,4,3,5)

    u3bba = lib.einsum('kJcE,BEAI->IJkABc', ints.t2ab, ints.WVVVO )  
#  P(IJ) 
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = lib.einsum('kIeA,ceBJ->IJkABc', ints.t2ab, ints.WvvVO )
 # P(IJ)P(AB)  

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

    u3bba = -lib.einsum('IMAB,MJck->IJkABc', ints.t2bb, ints.WOOvo ) 
#P(IJ)

    v += u3bba - u3bba.transpose(1,0,2,3,4,5)

    u3bba = -lib.einsum('kMcB,MJAI->IJkABc', ints.t2ab, ints.WOOVO ) 

#P(AB)
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)
    u3bba = -lib.einsum('mJcB,mkAI->IJkABc', ints.t2ab, ints.WooVO )

#P(IJ)P(AB)

    y = u3bba - u3bba.transpose(1,0,2,3,4,5)
    v += y - y.transpose(0,1,2,4,3,5)

# divide by denominator..
    u3bba = v 
    u3baa = r 

    w3 = u3aaa, u3bbb, u3baa, u3bba

    return w3


def update_amps_small(ints, w3, t3, act_hole, act_particle):
    '''Update non-canonical MP2 amplitudes'''

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

    mo_ea_o = numpy.diag(ints.Foo)
    mo_ea_v = numpy.diag(ints.Fvv)
    mo_eb_o = numpy.diag(ints.FOO)
    mo_eb_v = numpy.diag(ints.FVV)

    t3aaa, t3bbb, t3baa, t3bba = t3

#   u3aaa, u3bbb, r, v = w3

    u3aaa, u3bbb, r, v = (arr.copy() for arr in w3)

    eia = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    # aaa
    d3aaa  = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3aaa, ints.Fvv)
    u3aaa += cyclic_particle(x)
    x  = -lib.einsum('mjkabc,mi->ijkabc', t3aaa, ints.Foo)
    u3aaa += cyclic_hole(x)

    # bbb
    d3bbb = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)

    x  = lib.einsum('ijkabe,ce->ijkabc', t3bbb, ints.FVV)
    u3bbb += cyclic_particle(x)
    x = -lib.einsum('mjkabc,mi->ijkabc', t3bbb, ints.FOO)
    u3bbb += cyclic_hole(x)

    # baa
    d3baa = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)


    u3baa = lib.einsum('IjkAec,be->IjkAbc', t3baa, ints.Fvv)  
    r += u3baa - u3baa.transpose(0,1,2,3,5,4)

    r += lib.einsum('IjkEbc,AE->IjkAbc', t3baa, ints.FVV)  

    u3baa = -lib.einsum('ImkAbc,mj->IjkAbc', t3baa, ints.Foo)  
    r += u3baa - u3baa.transpose(0,2,1,3,4,5)
    r -= lib.einsum('MjkAbc,MI->IjkAbc', t3baa, ints.FOO)  

    # bba

    d3bba = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eia)

    u3bba = lib.einsum('IJkEBc,AE->IJkABc', t3bba, ints.FVV)  
    v += u3bba - u3bba.transpose(0,1,2,4,3,5)

    v += lib.einsum('IJkABe,ce->IJkABc',t3bba, ints.Fvv)  

    u3bba = -lib.einsum('MJkABc,MI->IJkABc', t3bba, ints.FOO)  
    v += u3bba - u3bba.transpose(1,0,2,3,4,5)
    v -= lib.einsum('IJmABc,mk->IJkABc',t3bba, ints.Foo)  

    R3 = u3aaa.copy(), u3bbb.copy(), r.copy(), v.copy()

# divide by denominator..
    u3aaa /=d3aaa 
    u3bbb /=d3bbb 
    u3bba = v/d3bba 
    u3baa = r/d3baa 

    u3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] = 0.0 
    u3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] = 0.0 
    u3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] = 0.0 
    u3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] = 0.0 

    u3aaa  += t3aaa
    u3bbb  += t3bbb
    u3bba  += t3bba
    u3baa  += t3baa

    t3new = u3aaa, u3bbb, u3baa, u3bba

    return t3new, R3 


def lhs_env_triples(ints, l1, l2, t3): 
    '''    
    t1, t2 amplitudes will be used to build the lhs. later we will replace them by L1 and L2 amplitudes..
    ''' 

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

#aaa
    v = lib.einsum('ebkc,ijae->ijkabc', ints.Wvvov , l2aa) 
    v -= lib.einsum('iajm,mkbc->ijkabc', ints.Wovoo , l2aa)

    v += lib.einsum('jbkc,ia->ijkabc', ints.Wovov, l1a)

    wd = cyclic_particle(cyclic_hole(v)) 

    et = lib.einsum('ijkabc,ijkabc', wd.conj(), t3aaa)*(1.0/36)

    print("value of et, step 1:", et)

#bbb
    v = lib.einsum('ebkc,ijae->ijkabc', ints.WVVOV , l2bb) 
    v -= lib.einsum('iajm,mkbc->ijkabc',ints.WOVOO , l2bb)
    v += lib.einsum('jbkc,ia->ijkabc', ints.WOVOV, l1b)

    wd = cyclic_particle(cyclic_hole(v)) 

    et += lib.einsum('ijkabc,ijkabc', wd.conj(), t3bbb)*(1.0/36)

    print("value of et, step 2:", et)
#baa
    w  = lib.einsum('ebkc,jIeA->IjkAbc', ints.Wvvov , l2ab)  
    w -= lib.einsum('mkbc,IAjm->IjkAbc', l2aa, ints.WOVoo ) 


    #P(jk)
    r = w - w.transpose(0,2,1,3,4,5)

    w  = lib.einsum('ebIA,jkec->IjkAbc', ints.WvvOV , l2aa) #Done 
    w -= lib.einsum('mIbA,kcjm->IjkAbc', l2ab, ints.Wovoo ) 
    #P(bc)
    r += w - w.transpose(0,1,2,3,5,4)

    w  = lib.einsum('EAkc,jIbE->IjkAbc', ints.WVVov , l2ab) #done
    w -= lib.einsum('jMbA,kcIM->IjkAbc', l2ab, ints.WovOO ) 
    w += lib.einsum('kcIA,jb->IjkAbc', ints.WovOV, l1a)
    #P(jk)P(bc)

    y = w - w.transpose(0,2,1,3,4,5)
    r += y - y.transpose(0,1,2,3,5,4)

    #P(None)

    r  += lib.einsum('jbkc,IA->IjkAbc', ints.Wovov, l1b)
    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3baa)*(1.0/4)
    print("value of et, step 3:", et)
 
#bba
    w  = lib.einsum('kJcE,EBIA->IJkABc', l2ab, ints.WVVOV ) #done 
    w -= lib.einsum('IMAB,kcJM->IJkABc', l2bb, ints.WovOO ) #done

# P(IJ)

    r = w - w.transpose(1,0,2,3,4,5)

    w  = lib.einsum('kJeB,ecIA->IJkABc', l2ab, ints.WvvOV ) #done
    w -= lib.einsum('mIcA,JBkm->IJkABc', l2ab, ints.WOVoo )  #done
    w += lib.einsum('kcIA,JB->IJkABc', ints.WovOV, l1b)

# P(IJ)P(AB)
    y = w - w.transpose(1,0,2,3,4,5)
    r += y - y.transpose(0,1,2,4,3,5)

    w = lib.einsum('IJAE,EBkc->IJkABc', l2bb, ints.WVVov ) 
    w -= lib.einsum('kMcB,IAJM->IJkABc',l2ab, ints.WOVOO ) 

# P(AB) 
    r += w - w.transpose(0,1,2,4,3,5)
    r  += lib.einsum('IAJB,kc->IJkABc', ints.WOVOV, l1a)

    et += lib.einsum('ijkabc,ijkabc', r.conj(), t3bba)*(1.0/4)

    print("value of et, step 4:", et)

#                    (1)                        ~        (1)
# contribution of T_3   to the residue of T_2: [F_ov, T_3   ]
     
    u2aa  = lib.einsum('ijmabe,me->ijab', t3aaa, ints.Fov)
    u2aa += lib.einsum('MijEab,ME->ijab', t3baa, ints.FOV)

    et += lib.einsum('ijab,ijab', u2aa.conj(), l2aa)*(1.0/4)

    u2bb = lib.einsum('ijmabe,me->ijab', t3bbb, ints.FOV)
    u2bb += lib.einsum('IJmABe,me->IJAB', t3bba, ints.Fov)

    et += lib.einsum('ijab,ijab', u2bb.conj(), l2bb)*(1.0/4)

    u2ab = lib.einsum('MIjEAb,ME->jIbA', t3bba, ints.FOV)
    u2ab += lib.einsum('IjmAbe,me->jIbA', t3baa, ints.Fov)

    et += lib.einsum('ijab,ijab', u2ab.conj(), l2ab)

    print("value of et, step 5:", et)

    return et

def iterative_kernel(mcc, eris, t1, t2, t3, act_hole, act_particle): 
    from pyscf.cc import uccsd_t_inactive_init

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2
    t3aaa,t3bbb,t3baa,t3bba = t3

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

#   t2aa_tmp = numpy.zeros((nocca,nocca,nvira,nvira), dtype=dtype)  
#   t2bb_tmp = numpy.zeros((noccb,noccb,nvirb,nvirb), dtype=dtype)  
#   t2ab_tmp = numpy.zeros((nocca,noccb,nvira,nvirb), dtype=dtype)  

#   t2aa_tmp[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])] +=  t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])] 
#   t2bb_tmp[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])] +=  t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])] 
#   t2ab_tmp[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])] +=  t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])] 

    u3aaa = numpy.zeros((nocca,nocca,nocca,nvira,nvira,nvira), dtype=dtype)  
    u3bbb = numpy.zeros((noccb,noccb,noccb,nvirb,nvirb,nvirb), dtype=dtype)  
    u3bba = numpy.zeros((noccb,noccb,nocca,nvirb,nvirb,nvira), dtype=dtype)  
    u3baa = numpy.zeros((noccb,nocca,nocca,nvirb,nvira,nvira), dtype=dtype)  

    u3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] += t3aaa 
    u3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] += t3bbb 
    u3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] += t3bba 
    u3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] += t3baa 

    t3 = u3aaa, u3bbb, u3baa, u3bba
#   t2_tmp = t2aa_tmp, t2ab_tmp, t2bb_tmp

    start = time.time()
#   ints = _make_4c_integrals(mcc, eris, t1, t2_tmp)
#   ints = _make_4c_integrals(mcc, eris, t1, t2)
    ints = _make_4c_integrals_bare(mcc, eris, t1, t2)

    w3 = build_w(ints)
    end = time.time()

    print("time to make intermediates", end-start)

    adiis = lib.diis.DIIS(mcc)

    conv = False
    for istep in range(mcc.max_cycle):
       
#       t3new = update_amps(ints, t3, act_hole, act_particle)
        t3new, _ = update_amps_small(ints, w3, t3, act_hole, act_particle)
        normt = numpy.linalg.norm([numpy.linalg.norm(t3new[i] - t3[i])
                                   for i in range(4)])

        t3shape = [x.shape for x in t3new]
        t3new = numpy.hstack([x.ravel() for x in t3new])
        t3new = adiis.update(t3new)
        t3new = lib.split_reshape(t3new, t3shape)

        t3, t3new = t3new, None

        print("cycle = ", istep+1, "norm(t3) = ", normt) 

        if normt < mcc.conv_tol_normt:
            conv = True
            break

    t3aaa, t3bbb, t3baa, t3bba = t3 

    t3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] = 0.0 
    t3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] = 0.0 
    t3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] = 0.0 
    t3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] = 0.0 

    t3 = t3aaa, t3bbb, t3baa, t3bba

    e_triples = lhs_env_triples(ints, t1, t2, t3) 
    print("Inactive contribution", e_triples)
    return e_triples



def noniterative_kernel(mcc, eris, t1, t2, t3, act_hole, act_particle): 
    from pyscf.cc import uccsd_t_inactive_init

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2
    t3aaa,t3bbb,t3baa,t3bba = t3

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

#   ints = _make_4c_integrals(mcc, eris, t1, t2_tmp)
    ints = _make_4c_integrals_bare(mcc, eris, t1, t2)


#   t2aa_tmp = numpy.zeros((nocca,nocca,nvira,nvira), dtype=dtype)  
#   t2bb_tmp = numpy.zeros((noccb,noccb,nvirb,nvirb), dtype=dtype)  
#   t2ab_tmp = numpy.zeros((nocca,noccb,nvira,nvirb), dtype=dtype)  

#   t2aa_tmp[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])] +=  t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])] 
#   t2bb_tmp[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])] +=  t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])] 
#   t2ab_tmp[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])] +=  t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])] 

#   u3aaa = numpy.zeros((nocca,nocca,nocca,nvira,nvira,nvira), dtype=dtype)  
#   u3bbb = numpy.zeros((noccb,noccb,noccb,nvirb,nvirb,nvirb), dtype=dtype)  
#   u3bba = numpy.zeros((noccb,noccb,nocca,nvirb,nvirb,nvira), dtype=dtype)  
#   u3baa = numpy.zeros((noccb,nocca,nocca,nvirb,nvira,nvira), dtype=dtype)  

    u3 = uccsd_t_inactive_init.kernel(mcc, eris, t1, t2, act_hole, act_particle) 

    u3aaa, u3bbb, u3baa, u3bba = u3

    t3aaa -= u3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])]  
    t3bbb -= u3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] 
    t3bba -= u3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] 
    t3baa -= u3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] 

    u3aaa = numpy.zeros((nocca,nocca,nocca,nvira,nvira,nvira), dtype=dtype)  
    u3bbb = numpy.zeros((noccb,noccb,noccb,nvirb,nvirb,nvirb), dtype=dtype)  
    u3bba = numpy.zeros((noccb,noccb,nocca,nvirb,nvirb,nvira), dtype=dtype)  
    u3baa = numpy.zeros((noccb,nocca,nocca,nvirb,nvira,nvira), dtype=dtype)  

    u3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] += t3aaa 
    u3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] += t3bbb 
    u3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] += t3bba 
    u3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] += t3baa 


    t3 = u3aaa, u3bbb, u3baa, u3bba
#   t2_tmp = t2aa_tmp, t2ab_tmp, t2bb_tmp

    start = time.time()

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


    w3 = build_w(ints)
    end = time.time()

    _, R3 = update_amps_small(ints, w3, t3, act_hole, act_particle)

    r3aaa, r3bbb, r3baa, r3bba = R3

#   r3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] *= 0.0 
#   r3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] *= 0.0 
#   r3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] *= 0.0 
#   r3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] *= 0.0 


    u3aaa_tr = numpy.einsum("ijkabc, iI, jJ, kK, aA, bB, cC -> IJKABC", r3aaa, ints.umat_occ_a,
                          ints.umat_occ_a, ints.umat_occ_a, ints.umat_vir_a, ints.umat_vir_a, ints.umat_vir_a,optimize=True)


    u3bbb_tr = numpy.einsum("ijkabc, iI, jJ, kK, aA, bB, cC -> IJKABC", r3bbb, ints.umat_occ_b,
                          ints.umat_occ_b, ints.umat_occ_b, ints.umat_vir_b, ints.umat_vir_b, ints.umat_vir_b,optimize=True)


    u3baa_tr = numpy.einsum("ijkabc, iI, jJ, kK, aA, bB, cC -> IJKABC", r3baa, ints.umat_occ_b,
                          ints.umat_occ_a, ints.umat_occ_a, ints.umat_vir_b, ints.umat_vir_a, ints.umat_vir_a,optimize=True)


    u3bba_tr = numpy.einsum("ijkabc, iI, jJ, kK, aA, bB, cC -> IJKABC", r3bba, ints.umat_occ_b,
                          ints.umat_occ_b, ints.umat_occ_a, ints.umat_vir_b, ints.umat_vir_b, ints.umat_vir_a,optimize=True)

    u3aaa_tr /=d3aaa 
    u3bbb_tr /=d3bbb 
    u3baa_tr /=d3baa 
    u3bba_tr /=d3bba 

    t3aaa = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3aaa_tr, ints.umat_occ_a,
                          ints.umat_occ_a, ints.umat_occ_a, ints.umat_vir_a, ints.umat_vir_a, ints.umat_vir_a,optimize=True)

    t3bbb = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3bbb_tr, ints.umat_occ_b,
                          ints.umat_occ_b, ints.umat_occ_b, ints.umat_vir_b, ints.umat_vir_b, ints.umat_vir_b,optimize=True)

    t3baa = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3baa_tr, ints.umat_occ_b,
                          ints.umat_occ_a, ints.umat_occ_a, ints.umat_vir_b, ints.umat_vir_a, ints.umat_vir_a,optimize=True)

    t3bba = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", u3bba_tr, ints.umat_occ_b,
                          ints.umat_occ_b, ints.umat_occ_a, ints.umat_vir_b, ints.umat_vir_b, ints.umat_vir_a,optimize=True)


    t3aaa[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] *= 0.0 
    t3bbb[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] *= 0.0 
    t3baa[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] *= 0.0 
    t3bba[numpy.ix_(act_hole[1], act_hole[1], act_hole[0], act_particle[1], act_particle[1], act_particle[0])] *= 0.0 

    t3 = t3aaa, t3bbb, t3baa, t3bba

    e_triples = lhs_env_triples(ints, t1, t2, t3) 
    print("Inactive contribution", e_triples)
    return e_triples


def kernel_bareV(mcc, eris, act_hole, act_particle, t1=None, t2=None):
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
#   mo_ea, mo_eb = eris.mo_energy
#   eia = mo_ea[:nocca,None] - mo_ea[nocca:]
#   eIA = mo_eb[:noccb,None] - mo_eb[noccb:]

    nocca, noccb, nvira, nvirb = t2ab.shape

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

# Wvvvo, WVVVO, WVVvo, WvvVO: untransformed.. 

    wovvv = numpy.asarray(eris.get_ovvv())
    wOVVV = numpy.asarray(eris.get_OVVV())
    wovVV = numpy.asarray(eris.get_ovVV())
    wOVvv = numpy.asarray(eris.get_OVvv())


    wovvv_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wovvv, umat_occ_a,umat_vir_a, umat_vir_a,  umat_vir_a,  optimize=True)
    wOVVV_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wOVVV,umat_occ_b, umat_vir_b, umat_vir_b,  umat_vir_b, optimize=True)
    wovVV_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wovVV, umat_occ_a,umat_vir_a, umat_vir_b,  umat_vir_b,  optimize=True)
    wOVvv_tr = numpy.einsum("abci,aA, bB, cC, iI -> ABCI", wOVvv, umat_occ_b, umat_vir_b, umat_vir_a,  umat_vir_a, optimize=True)


#   Wovoo, WOVoo, WovOO, WOVOO

    wovoo = numpy.asarray(eris.ovoo)
    wOVOO = numpy.asarray(eris.OVOO)
    wOVoo = numpy.asarray(eris.OVoo)
    wovOO = numpy.asarray(eris.ovOO)

    wovoo_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wovoo, umat_occ_a, umat_vir_a, umat_occ_a, umat_occ_a, optimize=True)
    wOVOO_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wOVOO, umat_occ_b, umat_vir_b, umat_occ_b, umat_occ_b, optimize=True)
    wOVoo_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wOVoo, umat_occ_b, umat_vir_b, umat_occ_a, umat_occ_a, optimize=True)
    wovOO_tr = numpy.einsum("iajk, iI, aA, jJ, kK -> IAJK", wovOO, umat_occ_a, umat_vir_a, umat_occ_b, umat_occ_b, optimize=True)


# wovov, wOVOV, wovOV

    wovov = numpy.asarray(eris.ovov)
    wOVOV = numpy.asarray(eris.OVOV)
    wovOV = numpy.asarray(eris.ovOV)


    wovov_tr = numpy.einsum("iajb, iI, aA, jJ, bB -> IAJB", wovov, umat_occ_a, umat_vir_a, umat_occ_a, umat_vir_a, optimize=True)
    wOVOV_tr = numpy.einsum("iajb, iI, aA, jJ, bB -> IAJB", wOVOV, umat_occ_b, umat_vir_b, umat_occ_b, umat_vir_b, optimize=True)
    wovOV_tr = numpy.einsum("iajb, iI, aA, jJ, bB -> IAJB", wovOV, umat_occ_a, umat_vir_a, umat_occ_b, umat_vir_b, optimize=True)


    t2aa_tr = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[0], umat_occ_a,
                         umat_occ_a, umat_vir_a, umat_vir_a, optimize=True)
           
    t2ab_tr = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[1], umat_occ_a,
                         umat_occ_b, umat_vir_a, umat_vir_b, optimize=True)
           
    t2bb_tr = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2[2], umat_occ_b,
                          umat_occ_b, umat_vir_b, umat_vir_b, optimize=True)

    t1aa_tr = lib.einsum("ia, iI, aA -> IA", t1[0], umat_occ_a, umat_vir_a)
    t1bb_tr = lib.einsum("ia, iI, aA -> IA", t1[1], umat_occ_b, umat_vir_b)

    Fov_tr = lib.einsum("ia, iI, aA -> IA", Fov, umat_occ_a, umat_vir_a)
    FOV_tr = lib.einsum("ia, iI, aA -> IA", FOV, umat_occ_b, umat_vir_b)


    eia = lib.direct_sum('i-a->ia', numpy.diag(eris.focka[:nocca,:nocca]), numpy.diag(eris.focka[nocca:,nocca:]))
    eIA = lib.direct_sum('i-a->ia', numpy.diag(eris.fockb[:noccb,:noccb]), numpy.diag(eris.fockb[noccb:,noccb:]))

    eia = lib.direct_sum('i-a->ia', e_occ_a, e_vir_a)
    eIA = lib.direct_sum('i-a->ia', e_occ_b, e_vir_b)

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = lib.einsum('ijae,kceb->ijkabc', t2aa_tr, wovvv_tr.conj())
    w-= lib.einsum('mkbc,iajm->ijkabc', t2aa_tr, wovoo_tr.conj())
    r = r6(w)
    v = lib.einsum('jbkc,ia->ijkabc', wovov_tr.conj(), t1aa_tr)
    v+= lib.einsum('jkbc,ia->ijkabc', t2aa_tr, Fov_tr) * .5
    wvd = p6(w + v) 
    r /= d3 
#    r[numpy.ix_(act_hole[0], act_hole[0], act_hole[0], act_particle[0], act_particle[0], act_particle[0])] *= 0.0 
    et = lib.einsum('ijkabc,ijkabc', wvd.conj(), r)


#extract the active contribution in original basis:
    r_act = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", r, umat_occ_a[numpy.ix_(act_hole[0], numpy.arange(nocca))],
                          umat_occ_a[numpy.ix_(act_hole[0], numpy.arange(nocca))],
                          umat_occ_a[numpy.ix_(act_hole[0], numpy.arange(nocca))],
                          umat_vir_a[numpy.ix_(act_particle[0], numpy.arange(nvira))],
                          umat_vir_a[numpy.ix_(act_particle[0], numpy.arange(nvira))],
                          umat_vir_a[numpy.ix_(act_particle[0], numpy.arange(nvira))] ,optimize=True)

    w_act = lib.einsum('ijae,kceb->ijkabc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))], 
                       wovvv[numpy.ix_(act_hole[0], act_particle[0], numpy.arange(nvira), act_particle[0])].conj())
    w_act-= lib.einsum('mkbc,iajm->ijkabc', t2aa[numpy.ix_(numpy.arange(nocca), act_hole[0], act_particle[0], act_particle[0])],
                     wovoo[numpy.ix_(act_hole[0], act_particle[0], act_hole[0],numpy.arange(nocca))].conj())
    v_act = lib.einsum('jbkc,ia->ijkabc', wovov[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], act_particle[0])].conj(),
                        t1a[numpy.ix_(act_hole[0], act_particle[0])])
    v_act+= lib.einsum('jkbc,ia->ijkabc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])], 
                       Fov[numpy.ix_(act_hole[0], act_particle[0])]) * .5
    wvd_act = p6(w_act + v_act) 

    et_act = lib.einsum('ijkabc,ijkabc', wvd_act.conj(), r_act)


    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,kceb->ijkabc', t2bb_tr, wOVVV_tr.conj())
    w-= numpy.einsum('imab,kcjm->ijkabc', t2bb_tr, wOVOO_tr.conj())
    r = r6(w)
    v = numpy.einsum('jbkc,ia->ijkabc', wOVOV_tr.conj(), t1bb_tr)
    v+= numpy.einsum('jkbc,ia->ijkabc', t2bb_tr, FOV_tr) * .5
    r /= d3 
#    r[numpy.ix_(act_hole[1], act_hole[1], act_hole[1], act_particle[1], act_particle[1], act_particle[1])] *= 0.0 
    wvd = p6(w + v) 
    et += numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

#extract the active contribution in original basis:
    r_act = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", r, umat_occ_a[numpy.ix_(act_hole[1], numpy.arange(noccb))],
                          umat_occ_b[numpy.ix_(act_hole[1], numpy.arange(noccb))], 
                          umat_occ_b[numpy.ix_(act_hole[1], numpy.arange(noccb))], 
                          umat_vir_b[numpy.ix_(act_particle[1], numpy.arange(nvirb))], 
                          umat_vir_b[numpy.ix_(act_particle[1], numpy.arange(nvirb))],
                          umat_vir_b[numpy.ix_(act_particle[1], numpy.arange(nvirb))],optimize=True)

    w_act = lib.einsum('ijae,kceb->ijkabc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))],
                        wOVVV[numpy.ix_(act_hole[1], act_particle[1], numpy.arange(nvirb), act_particle[1])].conj())
    w_act-= lib.einsum('mkbc,iajm->ijkabc', t2bb[numpy.ix_(numpy.arange(noccb), act_hole[1], act_particle[1], act_particle[1])],
                    wOVOO[numpy.ix_(act_hole[1], act_particle[1], act_hole[1],numpy.arange(noccb))].conj())
    v_act = lib.einsum('jbkc,ia->ijkabc', wOVOV[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], act_particle[1])].conj(), 
                       t1b[numpy.ix_(act_hole[1], act_particle[1])])
    v_act+= lib.einsum('jkbc,ia->ijkabc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])], 
                       FOV[numpy.ix_(act_hole[1], act_particle[1])]) * .5
    wvd_act = p6(w_act + v_act) 

    et_act += lib.einsum('ijkabc,ijkabc', wvd_act.conj(), r_act)

    # baa
    w  = numpy.einsum('jIeA,kceb->IjkAbc', t2ab_tr,wovvv_tr.conj()) * 2
    w += numpy.einsum('jIbE,kcEA->IjkAbc', t2ab_tr,wovVV_tr.conj()) * 2
    w += numpy.einsum('jkbe,IAec->IjkAbc', t2aa_tr,wOVvv_tr.conj())
    w -= numpy.einsum('mIbA,kcjm->IjkAbc', t2ab_tr,wovoo_tr.conj()) * 2
    w -= numpy.einsum('jMbA,kcIM->IjkAbc', t2ab_tr,wovOO_tr.conj()) * 2
    w -= numpy.einsum('jmbc,IAkm->IjkAbc', t2aa_tr,wOVoo_tr.conj())
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('jbkc,IA->IjkAbc',wovov_tr.conj(), t1bb_tr)
    v += numpy.einsum('kcIA,jb->IjkAbc',wovOV_tr.conj(), t1aa_tr)
    v += numpy.einsum('kcIA,jb->IjkAbc',wovOV_tr.conj(), t1aa_tr)
    v += numpy.einsum('jkbc,IA->IjkAbc', t2aa_tr, FOV_tr) * .5
    v += numpy.einsum('kIcA,jb->IjkAbc', t2ab_tr, Fov_tr) * 2
    w += v                                   
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    r /= d3
#    r[numpy.ix_(act_hole[1], act_hole[0], act_hole[0], act_particle[1], act_particle[0], act_particle[0])] *= 0.0 
    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)


    r_act = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", r, umat_occ_b[numpy.ix_(act_hole[1], numpy.arange(noccb))],
                          umat_occ_a[numpy.ix_(act_hole[0], numpy.arange(nocca))], 
                          umat_occ_a[numpy.ix_(act_hole[0], numpy.arange(nocca))], 
                          umat_vir_b[numpy.ix_(act_particle[1], numpy.arange(nvirb))], 
                          umat_vir_a[numpy.ix_(act_particle[0], numpy.arange(nvira))],
                          umat_vir_a[numpy.ix_(act_particle[0], numpy.arange(nvira))],optimize=True)
    
    w_act  = numpy.einsum('jIeA,kceb->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])],
                          wovvv[numpy.ix_(act_hole[0], act_particle[0], numpy.arange(nvira), act_particle[0])].conj()) * 2
    w_act += numpy.einsum('jIbE,kcEA->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))],
                            wovVV[numpy.ix_(act_hole[0], act_particle[0], numpy.arange(nvirb), act_particle[1])].conj()) * 2
    w_act += numpy.einsum('jkbe,IAec->IjkAbc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], numpy.arange(nvira))],
                            wOVvv[numpy.ix_(act_hole[1], act_particle[1], numpy.arange(nvira), act_particle[0])].conj())
    w_act -= numpy.einsum('mIbA,kcjm->IjkAbc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])],
                            wovoo[numpy.ix_(act_hole[0], act_particle[0], act_hole[0],numpy.arange(nocca))].conj()) * 2
    w_act -= numpy.einsum('jMbA,kcIM->IjkAbc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])],
                            wovOO[numpy.ix_(act_hole[0], act_particle[0], act_hole[1],numpy.arange(noccb))].conj()) * 2
    w_act -= numpy.einsum('jmbc,IAkm->IjkAbc', t2aa[numpy.ix_(act_hole[0], numpy.arange(nocca), act_particle[0], act_particle[0])],
                            wOVoo[numpy.ix_(act_hole[1], act_particle[1], act_hole[0],numpy.arange(nocca))].conj())
    

    v_act  = numpy.einsum('jbkc,IA->IjkAbc',wovov[numpy.ix_(act_hole[0], act_particle[0], act_hole[0], act_particle[0])].conj(), 
                          t1b[numpy.ix_(act_hole[1], act_particle[1])])   
    v_act += numpy.einsum('kcIA,jb->IjkAbc',wovOV[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_particle[1])].conj(), 
                          t1a[numpy.ix_(act_hole[0], act_particle[0])])
    v_act += numpy.einsum('kcIA,jb->IjkAbc',wovOV[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_particle[1])].conj(), 
                          t1a[numpy.ix_(act_hole[0], act_particle[0])])
    v_act += numpy.einsum('jkbc,IA->IjkAbc', t2aa[numpy.ix_(act_hole[0], act_hole[0], act_particle[0], act_particle[0])], 
                          FOV[numpy.ix_(act_hole[1], act_particle[1])]) * .5
    v_act += numpy.einsum('kIcA,jb->IjkAbc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])], 
                          Fov[numpy.ix_(act_hole[0], act_particle[0])]) * 2
    wvd_act = w_act + v_act 

    et_act += lib.einsum('ijkabc,ijkabc', wvd_act.conj(), r_act)    

    # abb
    w  = numpy.einsum('ijae,kceb->ijkabc', t2ab_tr, wOVVV_tr.conj()) * 2
    w += numpy.einsum('ijeb,kcea->ijkabc', t2ab_tr, wOVvv_tr.conj()) * 2
    w += numpy.einsum('jkbe,iaec->ijkabc', t2bb_tr, wovVV_tr.conj())
    w -= numpy.einsum('imab,kcjm->ijkabc', t2ab_tr, wOVOO_tr.conj()) * 2
    w -= numpy.einsum('mjab,kcim->ijkabc', t2ab_tr, wOVoo_tr.conj()) * 2
    w -= numpy.einsum('jmbc,iakm->ijkabc', t2bb_tr, wovOO_tr.conj())
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('jbkc,ia->ijkabc',wOVOV_tr.conj(), t1aa_tr)
    v += numpy.einsum('iakc,jb->ijkabc',wovOV_tr.conj(), t1bb_tr)
    v += numpy.einsum('iakc,jb->ijkabc',wovOV_tr.conj(), t1bb_tr)
    v += numpy.einsum('JKBC,ia->iJKaBC', t2bb_tr, Fov_tr) * .5
    v += numpy.einsum('iKaC,JB->iJKaBC', t2ab_tr, FOV_tr) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    r /= d3

#    r[numpy.ix_(act_hole[0], act_hole[1], act_hole[1], act_particle[0], act_particle[1], act_particle[1])] *= 0.0 

    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)

    r_act = numpy.einsum("IJKABC, iI, jJ, kK, aA, bB, cC -> ijkabc", r, umat_occ_a[numpy.ix_(act_hole[0], numpy.arange(nocca))],
                          umat_occ_b[numpy.ix_(act_hole[1], numpy.arange(noccb))], 
                          umat_occ_b[numpy.ix_(act_hole[1], numpy.arange(noccb))], 
                          umat_vir_a[numpy.ix_(act_particle[0], numpy.arange(nvira))], 
                          umat_vir_b[numpy.ix_(act_particle[1], numpy.arange(nvirb))],
                          umat_vir_b[numpy.ix_(act_particle[1], numpy.arange(nvirb))],optimize=True)
    
    w_act  = numpy.einsum('ijae,kceb->ijkabc', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], numpy.arange(nvirb))],
                          wOVVV[numpy.ix_(act_hole[1], act_particle[1], numpy.arange(nvirb), act_particle[1])].conj()) * 2
    w_act += numpy.einsum('ijeb,kcea->ijkabc', t2ab[numpy.ix_(act_hole[0], act_hole[1], numpy.arange(nvira), act_particle[1])],
                          wOVvv[numpy.ix_(act_hole[1], act_particle[1], numpy.arange(nvira), act_particle[0])].conj()) * 2
    w_act += numpy.einsum('jkbe,iaec->ijkabc', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], numpy.arange(nvirb))],
                          wovVV[numpy.ix_(act_hole[0], act_particle[0], numpy.arange(nvirb), act_particle[1])].conj())
    w_act -= numpy.einsum('imab,kcjm->ijkabc', t2ab[numpy.ix_(act_hole[0], numpy.arange(noccb), act_particle[0], act_particle[1])],
                          wOVOO[numpy.ix_(act_hole[1], act_particle[1], act_hole[1],numpy.arange(noccb))].conj()) * 2
    w_act -= numpy.einsum('mjab,kcim->ijkabc', t2ab[numpy.ix_(numpy.arange(nocca), act_hole[1], act_particle[0], act_particle[1])],
                          wOVoo[numpy.ix_(act_hole[1], act_particle[1], act_hole[0],numpy.arange(nocca))].conj()) * 2
    w_act -= numpy.einsum('jmbc,iakm->ijkabc', t2bb[numpy.ix_(act_hole[1], numpy.arange(noccb), act_particle[1], act_particle[1])],
                            wovOO[numpy.ix_(act_hole[0], act_particle[0], act_hole[1],numpy.arange(noccb))].conj())
    
    v_act  = numpy.einsum('jbkc,ia->ijkabc',wOVOV[numpy.ix_(act_hole[1], act_particle[1], act_hole[1], act_particle[1])].conj(), 
                          t1a[numpy.ix_(act_hole[0], act_particle[0])])   
    v_act += numpy.einsum('iakc,jb->ijkabc',wovOV[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_particle[1])].conj(), 
                          t1b[numpy.ix_(act_hole[1], act_particle[1])])
    v_act += numpy.einsum('iakc,jb->ijkabc',wovOV[numpy.ix_(act_hole[0], act_particle[0], act_hole[1], act_particle[1])].conj(), 
                          t1b[numpy.ix_(act_hole[1], act_particle[1])])
    v_act += numpy.einsum('JKBC,ia->iJKaBC', t2bb[numpy.ix_(act_hole[1], act_hole[1], act_particle[1], act_particle[1])], 
                          Fov[numpy.ix_(act_hole[0], act_particle[0])]) * .5
    v_act += numpy.einsum('iKaC,JB->iJKaBC', t2ab[numpy.ix_(act_hole[0], act_hole[1], act_particle[0], act_particle[1])], 
                          FOV[numpy.ix_(act_hole[1], act_particle[1])]) * 2
    wvd_act = w_act + v_act     

    et_act += lib.einsum('ijkabc,ijkabc', wvd_act.conj(), r_act)

    et *= .25

    et_act *= .25
    et_inact = et - et_act
    print("T3 (2) energy (UHF, DF) = %16.10f " % (et))

    print("T3 (2) energy active (UHF, DF) = %16.10f " % (et_act))
    print("T3 (2) energy inactive (UHF, DF) = %16.10f " % (et_inact))
   
    return et_inact


def _make_4c_integrals(mycc, eris, t1, t2):
    assert mycc._scf.istype('UHF')
#    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(mycc.stdout, mycc.verbose)
#   eris = _ChemistsERIs()
#   eris._common_init_(mycc, mo_coeff)

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

    mo_coeff = eris.mo_coeff
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

# Now construct all different integral types.. 

    e_occ_a, umat_occ_a = scipy.linalg.eig(Foo)     
    e_occ_b, umat_occ_b = scipy.linalg.eig(FOO)     

    e_vir_a, umat_vir_a = scipy.linalg.eig(Fvv)     
    e_vir_b, umat_vir_b = scipy.linalg.eig(FVV)     

    umat_occ_a = numpy.real(umat_occ_a)
    umat_occ_b = numpy.real(umat_occ_b)
                
    umat_vir_a = numpy.real(umat_vir_a)
    umat_vir_b = numpy.real(umat_vir_b)

# Wvvvo, WVVVO, WVVvo, WvvVO: 
    wvvvo = lib.einsum("Lae, Lbi -> aebi", Jvv, Jvo) 
    wvvvo = wvvvo - wvvvo.transpose(2,1,0,3)

    wVVVO = lib.einsum("Lae, Lbi -> aebi", JVV, JVO) 
    wVVVO = wVVVO - wVVVO.transpose(2,1,0,3)

    wVVvo = lib.einsum("Lae, Lbi -> aebi", JVV, Jvo) 
    wvvVO = lib.einsum("Lae, Lbi -> aebi", Jvv, JVO) 

#   wvvov, wvvOV, wVVov, wVVOV

    wvvov = lib.einsum("Lae, Lmf -> aemf",Jvv, Jov) 
    wvvov = wvvov - wvvov.transpose(0,3,2,1)

    wVVOV = lib.einsum("Lae, Lmf -> aemf",JVV, JOV) 
    wVVOV = wVVOV - wVVOV.transpose(0,3,2,1)

    wVVov = lib.einsum("Lae, Lmf -> aemf",JVV, Jov) 
    wvvOV = lib.einsum("Lae, Lmf -> aemf",Jvv, JOV) 

#   Wovoo, WOVoo, WovOO, WOVOO
    wovoo = lib.einsum("Lkc, Ljm -> kcjm",Jov, Joo) 
    wovoo = wovoo - wovoo.transpose(2,1,0,3)

    wOVOO = lib.einsum("Lkc, Ljm -> kcjm",JOV, JOO) 
    wOVOO = wOVOO - wOVOO.transpose(2,1,0,3)
    wOVoo = lib.einsum("Lkc, Ljm -> kcjm",JOV, Joo) 
    wovOO = lib.einsum("Lkc, Ljm -> kcjm",Jov, JOO) 


#   Woovo, WooVO, WOOvo, WOOVO 
    woovo = lib.einsum("Lmj, Lck -> mjck",Joo, Jvo) 
    woovo = woovo - woovo.transpose(0,3,2,1)

    wOOVO = lib.einsum("Lmj, Lck -> mjck",JOO, JVO) 
    wOOVO = wOOVO - wOOVO.transpose(0,3,2,1)

    wooVO = lib.einsum("Lmj, Lck -> mjck",Joo, JVO) 
    wOOvo = lib.einsum("Lmj, Lck -> mjck",JOO, Jvo) 

# wovov, wOVOV, wovOV
    wovov = lib.einsum("Lia, Ljb -> iajb",Jov, Jov) 
    wOVOV = lib.einsum("Lia, Ljb -> iajb",JOV, JOV) 
    wovOV = lib.einsum("Lia, Ljb -> iajb",Jov, JOV) 

    wovov = wovov - wovov.transpose(0,3,2,1)
    wOVOV = wOVOV - wOVOV.transpose(0,3,2,1)

    ints.Wvvvo, ints.WVVVO, ints.WVVvo, ints.WvvVO = wvvvo, wVVVO, wVVvo, wvvVO
    ints.Wvvov, ints.WvvOV, ints.WVVov, ints.WVVOV = wvvov, wvvOV, wVVov, wVVOV
    ints.Wovoo, ints.WOVoo, ints.WovOO, ints.WOVOO = wovoo, wOVoo, wovOO, wOVOO
    ints.Woovo, ints.WooVO, ints.WOOvo, ints.WOOVO = woovo, wooVO, wOOvo, wOOVO
    ints.Wovov, ints.WOVOV, ints.WovOV = wovov, wOVOV, wovOV
    
    ints.Fov, ints.FOV = Fov, FOV 
    ints.Foo, ints.FOO = Foo, FOO
    ints.Fvv, ints.FVV = Fvv, FVV 

    ints.ea_occ = numpy.real(e_occ_a) 
    ints.eb_occ = numpy.real(e_occ_b)
    ints.ea_vir = numpy.real(e_vir_a)
    ints.eb_vir = numpy.real(e_vir_b)

    ints.umat_occ_a = umat_occ_a
    ints.umat_vir_a = umat_vir_a
    ints.umat_occ_b = umat_occ_b
    ints.umat_vir_b = umat_vir_b


    ints.t2aa, ints.t2ab, ints.t2bb = t2aa, t2ab, t2bb
    ints.t1a, ints.t1b  = t1a, t1b

    return ints


def _make_4c_integrals_bare(mycc, eris, t1, t2):
    assert mycc._scf.istype('UHF')
#    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(mycc.stdout, mycc.verbose)
#   eris = _ChemistsERIs()
#   eris._common_init_(mycc, mo_coeff)

    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2

    mo_coeff = eris.mo_coeff
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

# Now construct all different integral types.. 

    e_occ_a, umat_occ_a = scipy.linalg.eigh(Foo)     
    e_occ_b, umat_occ_b = scipy.linalg.eigh(FOO)     

    e_vir_a, umat_vir_a = scipy.linalg.eigh(Fvv)     
    e_vir_b, umat_vir_b = scipy.linalg.eigh(FVV)     

    umat_occ_a = numpy.real(umat_occ_a)
    umat_occ_b = numpy.real(umat_occ_b)
                
    umat_vir_a = numpy.real(umat_vir_a)
    umat_vir_b = numpy.real(umat_vir_b)

# Wvvvo, WVVVO, WVVvo, WvvVO: 
    wvvvo = numpy.asarray(eris.get_ovvv()).transpose(3,2,1,0)
    wVVVO = numpy.asarray(eris.get_OVVV()).transpose(3,2,1,0)
    wVVvo = numpy.asarray(eris.get_ovVV()).transpose(3,2,1,0)
    wvvVO = numpy.asarray(eris.get_OVvv()).transpose(3,2,1,0)

    wvvvo = wvvvo - wvvvo.transpose(2,1,0,3)
    wVVVO = wVVVO - wVVVO.transpose(2,1,0,3)

#   wvvov, wvvOV, wVVov, wVVOV

    wvvov = numpy.asarray(eris.get_ovvv()).transpose(2,3,0,1)
    wVVOV = numpy.asarray(eris.get_OVVV()).transpose(2,3,0,1)
    wvvOV = numpy.asarray(eris.get_OVvv()).transpose(2,3,0,1)
    wVVov = numpy.asarray(eris.get_ovVV()).transpose(2,3,0,1)

    wvvov = wvvov - wvvov.transpose(0,3,2,1)
    wVVOV = wVVOV - wVVOV.transpose(0,3,2,1)

#   Wovoo, WOVoo, WovOO, WOVOO
    wovoo = numpy.asarray(eris.ovoo)
    wOVOO = numpy.asarray(eris.OVOO)
    wOVoo = numpy.asarray(eris.OVoo)
    wovOO = numpy.asarray(eris.ovOO)

    wovoo = wovoo - wovoo.transpose(2,1,0,3)
    wOVOO = wOVOO - wOVOO.transpose(2,1,0,3)

#   Woovo, WooVO, WOOvo, WOOVO 

    woovo = numpy.asarray(eris.ovoo).transpose(3,2,1,0)
    wOOVO = numpy.asarray(eris.OVOO).transpose(3,2,1,0)
    wooVO = numpy.asarray(eris.OVoo).transpose(3,2,1,0)
    wOOvo = numpy.asarray(eris.ovOO).transpose(3,2,1,0)

    woovo = woovo - woovo.transpose(0,3,2,1)
    wOOVO = wOOVO - wOOVO.transpose(0,3,2,1)

# wovov, wOVOV, wovOV
    wovov = numpy.asarray(eris.ovov)
    wOVOV = numpy.asarray(eris.OVOV)
    wovOV = numpy.asarray(eris.ovOV)

    wovov = wovov - wovov.transpose(0,3,2,1)
    wOVOV = wOVOV - wOVOV.transpose(0,3,2,1)

    ints.Wvvvo, ints.WVVVO, ints.WVVvo, ints.WvvVO = wvvvo, wVVVO, wVVvo, wvvVO
    ints.Wvvov, ints.WvvOV, ints.WVVov, ints.WVVOV = wvvov, wvvOV, wVVov, wVVOV
    ints.Wovoo, ints.WOVoo, ints.WovOO, ints.WOVOO = wovoo, wOVoo, wovOO, wOVOO
    ints.Woovo, ints.WooVO, ints.WOOvo, ints.WOOVO = woovo, wooVO, wOOvo, wOOVO
    ints.Wovov, ints.WOVOV, ints.WovOV = wovov, wOVOV, wovOV
    
    ints.Fov, ints.FOV = Fov, FOV 
    ints.Foo, ints.FOO = Foo, FOO
    ints.Fvv, ints.FVV = Fvv, FVV 

    ints.ea_occ = numpy.real(e_occ_a) 
    ints.eb_occ = numpy.real(e_occ_b)
    ints.ea_vir = numpy.real(e_vir_a)
    ints.eb_vir = numpy.real(e_vir_b)

    ints.umat_occ_a = umat_occ_a
    ints.umat_vir_a = umat_vir_a
    ints.umat_occ_b = umat_occ_b
    ints.umat_vir_b = umat_vir_b

    ints.t2aa, ints.t2ab, ints.t2bb = t2aa, t2ab, t2bb
    ints.t1a, ints.t1b  = t1a, t1b

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
        eri1 = lib.unpack_tril(eri1).reshape(-1, nao, nao)
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




def get_X(self, t1):

     Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
     Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
     X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

     return X, Xoo, Xvo

def get_J(self, Xoo, Xvo, t1):

    Joo = Xoo + self._eris.Loo
    Jvo = (
        Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
    )

    Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) #we don't need this here  

    return Joo, Jvo, Jvv

def get_F(self, t1, X, Xoo, Xvo):

    Foo = self._eris.foo.copy()
    Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
    Foo -= lib.einsum("Lmj,Lim->ij", self._eris.Loo,Xoo)

    Fvv = self._eris.fvv.copy()
    Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X)
    Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo)


    Fov  = self._eris.fov.copy()
    Fov += lib.einsum("Ljb,L->jb", self._eris.Lov, X)
    Fov -= lib.einsum("Lji,Lib->jb", Xoo, self._eris.Lov)

    return Foo, Fvv, Fov



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
    print(kernel(mcc, eris, t1, t2) - (-0.056092415718338388-0.011390417704868244j))
