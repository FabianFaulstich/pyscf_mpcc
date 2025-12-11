import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Plot CC2 iteration energies for a molecule and basis")
parser.add_argument("basis", type=str, help="Basis set, e.g., cc-pvdz")
parser.add_argument("molecule", type=str, help="Molecule name, e.g., H2O")
parser.add_argument("--scan",type=str,required=True, choices=["Lvv","Lov","Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],help="which tensor to scan:Lov etc")
parser.add_argument("--results", type=str, default=None, help="path to result folder")
args = parser.parse_args()

mol_name = args.molecule
basis = args.basis
scan = args.scan
basis_to_mol = {
            "cc-pvdz": "DZ",
            "cc-pvtz": "TZ",
            "aug-cc-pvdz": "aVDZ",
            "aug-cc-pvtz": "aug-cc-pvtz"
            }
Basis = basis_to_mol[basis]
import re

def mol_to_latex(mol_name):
    """
    Convert molecule name to LaTeX string for plot titles.
    """
    # --------------------------------------------------
    # TIP4P water clusters: TIP4P-1, TIP4P-2, ...
    # --------------------------------------------------
    m = re.match(r"TIP4P-(\d+)", mol_name)
    if m:
        n = int(m.group(1))
        if n == 1:
            return r"$\mathrm{H_2O}$"
        else:
            return rf"$\mathrm{{(H_2O)_{{{n}}}}}$"

    # --------------------------------------------------
    # Hydrocarbons: C2H6, C10H22
    # --------------------------------------------------
    m = re.match(r"C(\d+)H(\d+)$", mol_name)
    if m:
        return rf"$\mathrm{{C_{{{m.group(1)}}}H_{{{m.group(2)}}}}}$"

    # --------------------------------------------------
    # Generic fallback (safe default)
    # --------------------------------------------------
    return rf"$\mathrm{{{mol_name}}}$"
title_name = mol_to_latex(mol_name)

if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

energy_folder = os.path.join(results, basis, mol_name, f"energies_{scan}")

# LaTeX-friendly font sizes
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 10,
})

# Color-blind friendly markers
markers = ["o", "s", "D", "^", "v", "P", "X"]

# --------- LOAD FILES ---------
energy_files = sorted([
    f for f in os.listdir(energy_folder)
    if f.startswith("CC2_iter_energies")
])

plt.figure(figsize=(10, 7))
scan_str = str(scan).replace("_fix_Loo_1", "")

for i, f in enumerate(energy_files):
    file_path = os.path.join(energy_folder, f)
    iter_energies = np.loadtxt(file_path)
    iterations = np.arange(1, len(iter_energies) + 1)

    label = (
        f.replace("CC2_iter_energies_", "")
         .replace(f"CPD_{scan_str}_rank","") 
         .replace(".txt","")
         .replace("1X", "X")
         .replace(".0X", "X")
         )


    plt.plot(
        iterations,
        iter_energies,
        marker=markers[i % len(markers)],
        markersize=12,
        linewidth=2,
        label=label
    )
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
        fixed_rank_text = r"$R_{\mathrm{ov}} = 1.5X,\; R_{\mathrm{oo}} = 1X$"
    else:
        fixed_rank_text = r"$R_{\mathrm{ov}} = 2X,\; R_{\mathrm{oo}} = 1X$" 
    scan_rank = r"$R_{\mathrm{vv}}$"

elif scan == "Lov_Lvv_fix_Loo_1":
    fixed_rank_text = r"$R_{\mathrm{oo}} = 1X$"
    scan_rank = r"CP rank"
else:
    fixed_rank_text = r"$R_{\mathrm{oo}} = 0.5X$"
    scan_rank = r"CP rank"
# --------- LABELS ---------
plt.xlabel("Iteration")
plt.ylabel("Energy, Ha")
title_text = ("CC2 Energy Convergence\n" 
              f" {title_name}, {Basis}, ") 
if fixed_rank_text != "":
    title_text += ", " + fixed_rank_text

plt.title(title_text)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.legend(title= scan_rank, loc="lower right")
plt.savefig(os.path.join(energy_folder, f"CC2_iteration_plot_{basis}_{mol_name}_{scan}.png"),dpi=300)
plt.show()
##################
##################
"""
plt.figure(figsize=(10, 7))

file = os.path.join(energy_folder, "CC2_iter_energies_DF.txt")
df_energy = np.loadtxt(file)
energy_files.remove("CC2_iter_energies_DF.txt")

for i, f in enumerate(energy_files):
    file_path = os.path.join(energy_folder, f)
    iter_energies = np.loadtxt(file_path)

    diff = abs(df_energy - iter_energies) * 1000   # mHa
    diff = diff / n_atoms                           # per non-H atom

    iterations = np.arange(1, len(iter_energies) + 1)

    label = (
        f.replace(f"CC2_iter_energies_CPD_{scan}_rank", "")
         .replace(".txt", "")
         .replace("1X", "X")
         .replace(".0X", "X")
         )

    plt.plot(
        iterations,
        diff,
        marker=markers[i % len(markers)],
        markersize=12,
        linewidth=2,
        label=label
    )

# --------- LABELS ---------
plt.xlabel("Iteration")
plt.ylabel("Error, mHa/atom")
title_text = ("CC2 Error per Iteration\n"
          rf" $(H_{2}O)_{water_n}$, TZ, " 
          r"$R_{{\mathrm{oo}}}= X$")
if fixed_rank_text != "":
    title_text += ", " + fixed_rank_text
plt.title(title_text)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    title= scan_rank)

plt.tight_layout()
plt.savefig(
    os.path.join(energy_folder, f"CC2_energy_diff_iter_plot_{basis}_{mol_name}_{scan}.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
"""
