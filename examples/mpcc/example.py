import sys
from pathlib import Path
import os
os.environ['PYSCF_TMPDIR'] = '/home/talha/pyscf_tmp'

import glob
import numpy as np
from pyscf import gto, scf, cc, mp, mpcc
import time
import psutil
from memory_profiler import profile

from helper_fun import build_molecule
from arg_parse import parse_arg

# ---------- SAFE LOGGER ----------

if __name__ == "__main__":

    species, basis, rank_reduced_option, rank_value, result_path = parse_arg()
    mol, mol_name = build_molecule(species, basis)

    if rank_reduced_option:
        suffix = f"CPD_Lov_Lvv_rank{rank_value}X"
    else:
        suffix = "DF"

    @profile
    def init_scf_and_mpcc(species, basis, rank_reduced_option, rank_value):
        mol, mol_name = build_molecule(species, basis)

    # ---------- SCF ----------
        mf = scf.RHF(mol).density_fit().run()
        mf.mol.max_memory = 30000

        c_lo = mf.mo_coeff

    # MPCC options
        frag_info = {'frag': [[[0], [0]]]}
        conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
        rank_control = {
            'rank_reduced': rank_reduced_option,
            'rank_opts': {'Loo': 1, 'Lov': rank_value, 'Lvv': rank_value}
        }
        kwargs = frag_info | conv_info | rank_control

    # ---------- Initialize low-level MPCC ----------
        mympcc = mpcc.RMPCC(mf, 'True', c_lo, **kwargs)

        return mol, mf, mympcc
    mol, mf, mympcc = init_scf_and_mpcc(
        species, basis, rank_reduced_option, rank_value
    )
    """
    # ---------- SCF ----------
    mf = scf.RHF(mol).density_fit().run()
    mf.mol.max_memory = 20000  # max memory for SCF in MB
    c_lo = mf.mo_coeff

    frag_info = {'frag': [[[0], [0]]]}
    conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
    rank_control = {
        'rank_reduced': rank_reduced_option,
        'rank_opts': {'Loo': 1, 'Lov': rank_value, 'Lvv': rank_value}
    }
    kwargs = frag_info | conv_info | rank_control

    # ---------- Initialize low-level MPCC ----------
    mympcc = mpcc.RMPCC(mf, 'True', c_lo, **kwargs)
    """
    # ---------- CCSD WITH MEMORY PROFILING ----------
    @profile
    def run_ccsd(mf):
        mycc = cc.CCSD(mf)
        #mycc.max_memory = 4000  # limit CCSD memory
        mycc.max_cycle = 6
        mycc.incore_complete = False
        mycc.kernel()
        mycc.t2 = None
        t1_init = mycc.t1.copy()
        del mycc
        print("CCSD done, t1 copied, mycc deleted")
        
        return t1_init
    t1_init = run_ccsd(mf)

    # ---------- Init low-level amplitudes WITH MEMORY PROFILING ----------
    @profile
    def init_lowlevel(mympcc):
        _, _, Y = mympcc.lowlevel.init_amps()
        return Y

    Y = init_lowlevel(mympcc)

    # ---------- LOW LEVEL SOLVER WITH MEMORY PROFILING ----------
    @profile
    def run_lowlevel(mympcc, t1_init, Y):
        t1, t2, Y_out = mympcc.lowlevel.kernel(t1_init, [0], Y)
        return t1,t2, Y_out

    t1,t2, Y = run_lowlevel(mympcc, t1_init, Y)

    print("Low-level kernel done, memory optimized")

