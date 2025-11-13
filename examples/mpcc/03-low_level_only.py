import sys
import glob
import os
import time
import numpy as np
from pyscf import gto, scf, cc, mpcc

def read_xyz(filename):
    """Simple XYZ reader returning list of [atom, (x, y, z)]"""
    atoms = []
    with open(filename) as f:
        lines = f.readlines()[2:]  # skip first two lines
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                atom = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append([atom, (x, y, z)])
    return atoms

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Example: python example2.py water_cluster cc-pvdz True [cluster_index]")
        sys.exit(1)

    molecule = sys.argv[1].lower()
    basis = sys.argv[2]
    rank_reduced_option = sys.argv[3].lower() in ["true", "1", "yes"]

    cluster_index = None
    if len(sys.argv) == 5:
        cluster_index = int(sys.argv[4])

    print(f"\nRunning for: {molecule}, basis = {basis}, rank_reduced = {rank_reduced_option}")
    if cluster_index:
        print(f"Selected cluster/alkane index: {cluster_index}")

    molecules = []

    # --- Single water or ethane molecules ---
    if molecule in ["h2o", "water"]:
        mol = gto.Mole()
        mol.atom = [
            [8, (0.0, 0.0, 0.0)],
            [1, (0.0, -0.757, 0.587)],
            [1, (0.0, 0.757, 0.587)],
        ]
        mol.basis = basis
        mol.build()
        molecules = [mol]

    elif molecule in ["ethane", "c2h6"]:
        mol = gto.Mole()
        mol.atom = [
            [6, (-0.6695, 0.0000, 0.0000)],
            [6, (0.6695, 0.0000, 0.0000)],
            [1, (-1.2335, 0.9238, 0.0000)],
            [1, (-1.2335, -0.9238, 0.0000)],
            [1, (1.2335, 0.9238, 0.0000)],
            [1, (1.2335, -0.9238, 0.0000)],
        ]
        mol.basis = basis
        mol.build()
        molecules = [mol]

    # --- Water clusters ---
    elif molecule in ["water_cluster", "cluster"]:
        xyz_folder = "water_cluster"
        xyz_files = sorted(glob.glob(os.path.join(xyz_folder, "TIP4P-*.xyz")))
        if not xyz_files:
            raise FileNotFoundError(f"No TIP4P-*.xyz files found in {xyz_folder}/")
        if cluster_index:
            if cluster_index < 1 or cluster_index > len(xyz_files):
                raise ValueError(f"Cluster index out of range (1–{len(xyz_files)})")
            xyz_files = [xyz_files[cluster_index - 1]]
        for xyz_file in xyz_files:
            print(f"\n=== Found water cluster: {os.path.basename(xyz_file)} ===")
            mol = gto.Mole()
            mol.atom = read_xyz(xyz_file)
            mol.basis = basis
            mol.build()
            molecules.append(mol)

    # --- Alkanes ---
    elif molecule in ["alkane", "alkanes"]:
        xyz_folder = "alkanes"
        xyz_files = sorted(glob.glob(os.path.join(xyz_folder, "*.xyz")))
        if not xyz_files:
            raise FileNotFoundError(f"No .xyz files found in {xyz_folder}/")
        if cluster_index:
            if cluster_index < 1 or cluster_index > len(xyz_files):
                raise ValueError(f"Alkane index out of range (1–{len(xyz_files)})")
            xyz_files = [xyz_files[cluster_index - 1]]
        for xyz_file in xyz_files:
            print(f"\n=== Found alkane: {os.path.basename(xyz_file)} ===")
            mol = gto.Mole()
            mol.atom = read_xyz(xyz_file)
            mol.basis = basis
            mol.build()
            molecules.append(mol)

    else:
        raise ValueError(f"Unknown molecule '{molecule}'. Supported: h2o, water_cluster, ethane, alkanes")

    # --- Main computation loop ---
    for i, mol in enumerate(molecules, start=1):
        print(f"\n===== Running calculation {i}/{len(molecules)} =====")

        mf = scf.RHF(mol).density_fit().run()
        c_lo = mf.mo_coeff

        frag_info = {'frag': [[[0], [0]]]}  # default fragment
        conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
        rank_control = {'rank_reduced': rank_reduced_option}
        kwargs = frag_info | conv_info | rank_control

        mympcc = mpcc.RMPCC(mf, rank_reduced_option, c_lo, **kwargs)

        print('Initializing ...')
        st = time.time()

        mycc = cc.CCSD(mf)
        mycc.max_cycle = 6
        mycc.kernel()

        _, _, Y = mympcc.lowlevel.init_amps()
        print(f'Done! Elapsed time: {time.time() - st:.2f} sec')

        print('Starting Low-Level Solver')
        t1, t2, Y = mympcc.lowlevel.kernel(mycc.t1, [0], Y)
        print('Finished Low-Level solver!')

