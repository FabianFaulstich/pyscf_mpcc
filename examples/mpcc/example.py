import sys
import glob
import os
import numpy as np
from pyscf import gto, scf, cc, mpcc
from pyscf.mp.dfmp2_native import DFMP2
from pyscf.mcscf import avas
from pyscf.data.elements import chemcore


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
    # Allow optional cluster index
    if len(sys.argv) not in [4, 5]:
        print("Example: python example.py water_cluster cc-pvdz True [cluster_index]")
        sys.exit(1)

    molecule = sys.argv[1].lower()
    basis = sys.argv[2]
    rank_reduced_option = sys.argv[3].lower() in ["true", "1", "yes"]

    cluster_index = None
    if len(sys.argv) == 5:
        cluster_index = int(sys.argv[4])  # run only this cluster

    print(f"\nRunning for: {molecule}, basis = {basis}, rank_reduced = {rank_reduced_option}")
    if cluster_index:
        print(f"Selected cluster index: {cluster_index}")

    # === Single molecule cases ===
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

    elif molecule in ["water_cluster", "cluster"]:
        xyz_folder = "water_cluster"
        xyz_files = sorted(glob.glob(os.path.join(xyz_folder, "TIP4P-*.xyz")))
        if not xyz_files:
            raise FileNotFoundError(f"No TIP4P-*.xyz files found in {xyz_folder}/")

        # select specific cluster if index given
        if cluster_index:
            if cluster_index < 1 or cluster_index > len(xyz_files):
                raise ValueError(f"Cluster index out of range (1–{len(xyz_files)})")
            xyz_files = [xyz_files[cluster_index - 1]]

        molecules = []
        for xyz_file in xyz_files:
            print(f"\n=== Found water cluster: {os.path.basename(xyz_file)} ===")
            mol = gto.Mole()
            mol.atom = read_xyz(xyz_file)
            mol.basis = basis
            mol.build()
            molecules.append(mol)

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

    else:
        raise ValueError(f"Unknown molecule '{molecule}'. Supported: h2o, water_cluster, ethane")

    # === Main computation loop ===
    for i, mol in enumerate(molecules, start=1):
        print(f"\n===== Running calculation {i}/{len(molecules)} =====")

        mf = scf.RHF(mol).density_fit().run()
        mf.threshold = 1e-6

        mycc = cc.CCSD(mf)
        mycc.kernel()

        # AVAS localization
        ncore = 0
        ao_labels = ["O 2p", "O 2s", "H 1s"]
        minao = "sto-3g"
        openshell_option = 3

        avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
        avas_obj.with_iao = True
        avas_obj.threshold = 1e-7
        _, _, mocas = avas_obj.kernel()

        act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        act_part = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]

        print("Active hole dim:", len(act_hole))
        print("Active part dim:", len(act_part))

        frag_info = {'frag': [[act_hole, act_part]]}
        conv_info = {'ll_con_tol': 1e-6, 'll_max_its': 80}
        rank_control = {
            'rank_reduced': rank_reduced_option,
            'rank_opts': {'Loo': 0.5, 'Lov': 1.5, 'Lvv': 3.0},
        }

        kwargs = frag_info | conv_info | rank_control

        mympcc = mpcc.RMPCC(mf, rank_reduced_option, mf.mo_coeff, **kwargs)
        mympcc.kernel()

        print("Finished MPCC!")
        print(f'CCSD:\n  Total energy: {mycc.e_tot:.8f}  Correlation energy: {mycc.e_corr:.8f}')
        print(f'DF-MPCCSD:\n  Total energy: {mympcc.lowlevel.e_tot:.8f}  Correlation energy: {mympcc.lowlevel.e_corr:.8f}')
        print(f'Difference:\n  Total energy: {float(mympcc.lowlevel.e_tot - mycc.e_tot):.8f}  '
              f'Correlation energy: {mympcc.lowlevel.e_corr - mycc.e_corr:.8f}')

