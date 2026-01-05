from pyscf import gto, scf, cc
from pyscf import mpcc

from pyscf.mpcc import mpcc_tools as mpt

from pyscf.mcscf import avas

import numpy as np
import time

import os 
import glob
import argparse

def get_inputs():

    parser = argparse.ArgumentParser()
    parser.add_argument("items", nargs="*", help="key=value pairs")
    args = parser.parse_args()

    kwargs = {}
    for item in args.items:
        k, v = item.split("=", 1)

        if v == 'None':
            kwargs[k] = None
        else:
            if k == 'rank': 
                kwargs[k] = int(v)
            elif k == 'tol':
                kwargs[k] = float(v)
            else:
                kwargs[k] = v
    
    return kwargs

def get_geometry(kwargs):

    sys = kwargs['system']
    n_sys = int(kwargs['n'])

    if sys == 'water':
        return tip4p(n_sys)
    elif sys == 'alkane':
        return alkane(n_sys)

def alkane(n_sys, TIP4P_DIR = "alkane"):
    pattern = os.path.join(TIP4P_DIR, f"alkane-{n_sys}.xyz")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern}")
    file_path = matches[0]
    return read_alkane_xyz_geom(file_path)

def read_alkane_xyz_geom(file_path):
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    # If it's a standard XYZ (first line is natoms, second is comment), use that.
    # Otherwise assume every non-empty line is "Sym x y z".
    try:
        natoms = int(lines[0].strip())
        coord_lines = lines[2:2 + natoms]
    except (ValueError, IndexError):
        coord_lines = [ln for ln in lines if ln.strip()]

    geom = []
    for line in coord_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        geom.append((sym, (x, y, z)))

    return geom


def tip4p(n_sys, TIP4P_DIR = "TIP4P"):
    pattern = os.path.join(TIP4P_DIR, f"TIP4P-{n_sys}.xyz")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern}")
    file_path = matches[0]
    return read_tip4p_xyz_geom(file_path)

def read_tip4p_xyz_geom(file_path, skip_dummy=True):
    
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    try:
        natoms = int(lines[0].strip())
        coord_lines = lines[2:2 + natoms]
    except ValueError:
        coord_lines = [ln for ln in lines if ln.strip()]

    geom = []
    for line in coord_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        if skip_dummy and sym.upper() in ("M", "X", "D"):
            continue
        x, y, z = map(float, parts[1:4])
        geom.append((sym, (x, y, z)))

    return geom

if __name__ == "__main__":

    inputs = get_inputs()
    
    basis = inputs['basis']
    atoms = get_geometry(inputs)
    chol_tol = inputs['tol']
    chol_rank = inputs['rank']

    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis 
    mol.build()
    mf = scf.RHF(mol).density_fit().run()

    # Generating LO basis 
    ao_labels = mpt.get_ao_labels(mol)
    minao="sto-3g"
    openshell_option = 3

    ncore = 0
    nelec_as = tuple(nelec - ncore for nelec in mol.nelec)
    n_cas = mol.nao - ncore
    active_orbs = [p for p in range(ncore, mol.nao)]
    frozen_orbs = [i for i in range(mol.nao) if i not in active_orbs]

    avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
    avas_obj.with_iao = True
    avas_obj.threshold = 1e-7 
    
    _, _, c_lo = avas_obj.kernel()

    act_hole = (
        np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
    )

    act_part = (
        np.where(avas_obj.vir_weights > avas_obj.threshold)[0])

    kwargs = {'frag'            : [[act_hole, act_part]],
                'll_con_tol'    : 1e-6, 
                'll_max_its'    : 80,
                'll_kernel_type': 'factorized',
                'lo_coeff'      : c_lo
            }

    mympcc = mpcc.MPCC(mf, **kwargs)
 
    # Initializaing
    t1, t2 = mympcc.lowlevel.init_amps()

    num_its = 2 
    for i in range(num_its):
        print(f'Macro Iteration :{i}\n')
        # run low-level solver
        if kwargs['ll_kernel_type'] == 'unfactorized':
            t1, t2 = mympcc.lowlevel.kernel(t1, t2)
        else:
            t1, t2 = mympcc.lowlevel.kernel(t1, t2, chol_tol = chol_tol, chol_rank = chol_rank)
       
        # runnning high-lelvel solver
        t1_act = []
        t2_act = []

        mympcc.screened.frag = mympcc.frags[0]
        mympcc.highlevel.frag = mympcc.frags[0]

        imds = mympcc.screened.kernel(t1, t2)
        t1_act_tmp, t2_act_tmp = mympcc.highlevel.kernel(imds, t1, t2)

        t1_act.append(t1_act_tmp)    
        t2_act.append(t2_act_tmp) 

        act_hole = mympcc.frags[0][0]
        act_particle = mympcc.frags[0][1]

        t1[np.ix_(act_hole, act_particle)] = t1_act[0]
        t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[0]

        mpcc_ene = mympcc.highlevel.get_cc_energy(t1, t2, t1_act, t2_act)
        print(f'DFMPCC It: {i}; correlation energy: {mpcc_ene}')

