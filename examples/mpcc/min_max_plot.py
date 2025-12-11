import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser(description="Plot Error stats of water cluster")
parser.add_argument("basis", type=str, help="e.g cc-pvdz")
parser.add_argument("--molecule", type=str, required=True,
                    choices=["water_clusters", "carbon_chains"])
parser.add_argument("--scan", type=str, required=True,
                    choices=["Lov", "Lvv", "Lov_Lvv",
                             "Lov_fix_Loo_1", "Lvv_fix_Loo_1", "Lov_Lvv_fix_Loo_1"])
args = parser.parse_args()

basis = args.basis
molecule = args.molecule
scan = args.scan
basis_to_mol = {
            "cc-pvdz": "DZ",
            "cc-pvtz": "TZ",
            "aug-cc-pvdz": "aVDZ",
            "aug-cc-pvtz": "aug-cc-pvtz"
            }
Basis = basis_to_mol[basis]

# ----------------------------
# User-defined settings
# ----------------------------
if molecule == "water_clusters":
    molecules = ["TIP4P-1","TIP4P-2", "TIP4P-4",
                 "TIP4P-6", "TIP4P-8", "TIP4P-10"]
else:
    molecules = ["c2h6", "c4h10", "c6h14","c8h18","c10h22"]

ranks = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
results = f"/home/talha/Documents/RR_MPCC/All_output_data/{basis}"

rank_to_diffs = {r: [] for r in ranks}

non_H_atoms = {
    "TIP4P-1": 1, "TIP4P-2": 2, "TIP4P-4": 4,
    "TIP4P-6": 6, "TIP4P-8": 8, "TIP4P-10": 10,
    "c2h6": 2, "c4h10": 4, "c6h14": 6,
    "c8h18": 8, "c10h22": 10
}
scan_str = str(scan).replace("_fix_Loo_1", "")

# ----------------------------
# Load data
# ----------------------------
for mol in molecules:

    n_atoms = non_H_atoms[mol]
    energy_folder = os.path.join(results, mol, f"energies_{scan}")

    if not os.path.exists(energy_folder):
        print(f"Skipping {mol}: folder not found")
        continue

    df_energy = np.loadtxt(os.path.join(energy_folder, "CC2_iter_energies_DF.txt"))

    for r in ranks:
        cpd_file = os.path.join(energy_folder, f"CC2_iter_energies_CPD_{scan_str}_rank{r}X.txt")
        iter_energies = np.loadtxt(cpd_file)

        diff = abs(df_energy - iter_energies) * 1000
        diff /= n_atoms

        rank_to_diffs[r].append(diff)

# -------------------------------------------------------------
# Plot statistics
# -------------------------------------------------------------
plt.figure(figsize=(11, 8))
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown"]

for i, r in enumerate(ranks):

    if len(rank_to_diffs[r]) == 0:
        continue

    lists = rank_to_diffs[r]
    min_len = min(len(x) for x in lists)
    trimmed = [x[:min_len] for x in lists]

    arr = np.array(trimmed)
    mean_diff = arr.mean(axis=0)
    min_diff = arr.min(axis=0)
    max_diff = arr.max(axis=0)

    iterations = np.arange(1, len(mean_diff) + 1)
    color = colors[i % len(colors)]
    yerr = np.vstack([mean_diff - min_diff, max_diff - mean_diff])

    label = f"{r}X".replace("1X", "X").replace(".0X", "X")

    plt.errorbar(iterations, mean_diff, yerr=yerr,
                 fmt='-o', linewidth=2, markersize=10,
                 color=color, capsize=12, capthick=1.5,
                 label=label)

# -------------------------------------------------------------
# Scan text logic
# -------------------------------------------------------------
if scan == "Lvv":
    if basis == "cc-pvdz":
        fixed_rank_text = r"$R_{\mathrm{oo}} = 0.5X,\; R_{\mathrm{ov}} = 1.5X$"
    else:
        fixed_rank_text = r"$R_{\mathrm{oo}} = 0.5X,\; R_{\mathrm{ov}} = 2X$"
    scan_rank = r"$R_{\mathrm{vv}}$"

elif scan == "Lov":
    fixed_rank_text = r"$R_{\mathrm{oo}} = 0.5X,\; R_{\mathrm{vv}} = 2.5X$"
    scan_rank = r"$R_{\mathrm{ov}}$"

elif scan == "Lov_fix_Loo_1":
    fixed_rank_text = r"$R_{\mathrm{oo}} = X,\; R_{\mathrm{vv}} = 2.5X$"
    scan_rank = r"$R_{\mathrm{ov}}$"

elif scan == "Lvv_fix_Loo_1":
    if basis == "cc-pvdz":
        fixed_rank_text = r"$R_{\mathrm{ov}} = 1.5X,\; R_{\mathrm{oo}} = X$"
    else:
        fixed_rank_text = r"$R_{\mathrm{ov}} = 2X,\; R_{\mathrm{oo}} = X$"
    scan_rank = r"$R_{\mathrm{vv}}$"

elif scan == "Lov_Lvv_fix_Loo_1":
    fixed_rank_text = r"$R_{\mathrm{oo}} = X$"
    scan_rank = r"CP rank"

else:
    fixed_rank_text = r"$R_{\mathrm{oo}} = 0.5X$"
    scan_rank = r"CP rank"

# -------------------------------------------------------------
# Plot formatting
# -------------------------------------------------------------
plt.ylim(1e-3, 1e0)
plt.yscale('log')
plt.yticks([1e-3, 1e-2, 1e-1, 1e0])
plt.xlabel("Iteration")
plt.ylabel("Error, mH/atom")

title_text = "CC2 Energy Error\n"
title_text += ("Water Clusters" if molecule == "water_clusters" else "Carbon Chains")
title_text += f", {Basis}"
title_text += f", {fixed_rank_text}"
plt.title(title_text)

plt.grid(True, linestyle="--", alpha=0.6)

# -------------------------------------------------------------
# LEGEND 1 (Ranks)
# -------------------------------------------------------------
rank_legend = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
                         title=f"{scan_rank}")
plt.gca().add_artist(rank_legend)

# -------------------------------------------------------------
# LEGEND 2 (Molecules)
# -------------------------------------------------------------
from matplotlib.lines import Line2D

mol_handles = [Line2D([0], [0], marker='o', linestyle='',
                      markersize=10, color='black') for _ in molecules]

plt.legend(mol_handles, molecules,
           loc='lower left', bbox_to_anchor=(0, 0),
           title="Molecules")

plt.tight_layout()

out_file = os.path.join(results, f"CC2_stats_{molecule}_{scan}.png")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()

print("\nSaved plot to:", out_file)

