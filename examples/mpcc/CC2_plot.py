import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

parser = argparse.ArgumentParser(description="Plot CC2 iteration energies for a molecule and basis")
parser.add_argument("basis", type=str, help="Basis set, e.g., cc-pvdz")
parser.add_argument("molecule", type=str, help="Molecule name, e.g., H2O")
parser.add_argument("--scan",type=str,required=True, choices=["Lvv","Lov","Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],help="which tensor to scan:Lov etc")
parser.add_argument("--results", type=str, default=None, help="path to result folder")
parser.add_argument("method", type =str,help= "e.g., CC2")
#parser.add_argument("macro_it", type=int, help= "1 or 2")
args = parser.parse_args()

mol_name = args.molecule
basis = args.basis
scan = args.scan
method = args.method
#macro_it= args.macro_it
basis_to_mol = {
            "cc-pvdz": "DZ",
            "cc-pvtz": "TZ",
            "aug-cc-pvdz": "aVDZ",
            "aug-cc-pvtz": "aug-cc-pvtz"
            }
Basis = basis_to_mol[basis]
import re
non_H_atoms = {
    "TIP4P-1": 1, "TIP4P-2": 2, "TIP4P-3": 3,"TIP4P-4": 4,
    "TIP4P-5": 5,"TIP4P-6": 6, "TIP4P-8": 8, "TIP4P-10": 10,
    "ch4":1,"c2h6": 2,"c3h8":3, "c4h10": 4,"c5h12":5, "c6h14": 6,
    "c8h18": 8, "c10h22": 10
}
n_atoms = non_H_atoms[mol_name]
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
            return r"${H_2O}$"
        else:
            return rf"${{(H_2O)_{{{n}}}}}$"

    # --------------------------------------------------
    # Hydrocarbons: C2H6, C10H22
    # --------------------------------------------------
    m = re.match(r"c(\d+)h(\d+)$", mol_name)
    if m:
        return rf"${{C_{{{m.group(1)}}}H_{{{m.group(2)}}}}}$"

    # --------------------------------------------------
    # Generic fallback (safe default)
    # --------------------------------------------------
    return rf"${mol_name}$"
title_name = mol_to_latex(mol_name)

if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

energy_folder = os.path.join(results, basis, mol_name, f"energies_{method}")

energy_CCSD = os.path.join(results, basis, mol_name, f"energies_CC_SD")
file = os.path.join(energy_CCSD, "DF_CCSD_energy.txt")
ccsd_energy = np.loadtxt(file)
ccsd_energy = float(np.atleast_1d(ccsd_energy)[-1])

# LaTeX-friendly font sizes
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
})

# Color-blind friendly markers
markers = ["o", "s", "D", "^", "v", "P", "X"]

# --------- LOAD FILES ---------
energy_files = sorted([
    f for f in os.listdir(energy_folder)
    if f.startswith(f"{method}_iter_energy") and "rank1.0X" not in f])
    

plt.figure(figsize=(10, 7))
scan_str = str(scan).replace("_fix_Loo_1", "")

for i, f in enumerate(energy_files):
    file_path = os.path.join(energy_folder, f)
    iter_energies = np.loadtxt(file_path)
    iter_energies = iter_energies[1:]
    iterations = np.arange(1, len(iter_energies) + 1)

    label = (
        f.replace(f"{method}_iter_energy_", "")
         #.replace("CC2_iter_energies_", "")  
         .replace(f"CPD_{scan_str}_rank","") 
         .replace("X", r"$X$")
         .replace(".txt","")
         .replace("_2", "")
         .replace(".0X", r"$X$")
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
        fixed_rank_text = r"$R_{{oo}} = 0.5X,\; R_{{ov}} = 1.5X$"
    else: 
         fixed_rank_text = r"$R_{{oo}} = 0.5X,\; R_{{ov}} = 2X$"
    scan_rank = r"$R_{{vv}}$"

elif scan == "Lov":
    fixed_rank_text = r"$R_{{oo}} = 0.5X,\; R_{{vv}} = 2.5X$"
    scan_rank = r"$R_{{ov}}$"
elif scan == "Lov_fix_Loo_1":
    fixed_rank_text = r"$R_{{oo}} = X,\; R_{{vv}} = 2.5X$"
    scan_rank = r"$R_{{ov}}$"

elif scan == "Lvv_fix_Loo_1":
    if basis == "cc-pvdz":
        fixed_rank_text = r"$R_{{ov}} = 1.5X,\; R_{{oo}} = 1X$"
    else:
        fixed_rank_text = r"$R_{{ov}} = 2X,\; R_{{oo}} = 1X$" 
    scan_rank = r"$R_{{vv}}$"

elif scan == "Lov_Lvv_fix_Loo_1":
    fixed_rank_text = r"$R_{{oo}} = 1X$"
    scan_rank = r"CP rank"
else:
    fixed_rank_text = r"$R_{{oo}} = 0.5X$"
    scan_rank = r"CP rank"
# --------- LABELS ---------
#plt.axhline(y=ccsd_energy,linewidth=2.5,alpha=0.9,label="DF-CCSD")
plt.xlabel(r"Iteration")
plt.ylabel(r"Energy, Ha")
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
title_text = (rf"{method} Energy Convergence""\n" 
              rf" {title_name}, {Basis} ") 
#if fixed_rank_text != "":
    #title_text += ", " + fixed_rank_text

plt.title(title_text)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    title= scan_rank)
plt.tight_layout()
plt.savefig(os.path.join(energy_folder, f"{method}_iteration_plot_{basis}_{mol_name}_{scan}_minus_1st_iter_new.png"),dpi=300)
plt.show()
##################
##################
#"""
plt.figure(figsize=(10, 7))

file = os.path.join(energy_folder, f"{method}_iter_energy_DF_2.txt")
df_energy = np.loadtxt(file)
df_energy = df_energy[1:]

energy_files.remove(f"{method}_iter_energy_DF_2.txt")

for i, f in enumerate(energy_files):
    file_path = os.path.join(energy_folder, f)
    iter_energies = np.loadtxt(file_path)
    iter_energies = iter_energies[1:]
    n = min(len(iter_energies), len(df_energy))
    diff = abs(df_energy[:n] - iter_energies[:n]) * 1000   # mHa
    print(f"no of non H atom:{n_atoms}")
    diff = diff / n_atoms                           # per non-H atom

    iterations = np.arange(1, len(iter_energies) + 1)

    label = (
        f.replace(f"{method}_iter_energy_", "")
         #.replace("CC2_iter_energies_", "")  
         .replace(f"CPD_{scan_str}_rank","") 
         .replace("X", r"$X$")
         .replace(".txt","")
         .replace("_2", "")
         .replace(".0X", r"$X$")
         )
    plt.plot(
            iterations[:n],
        diff,
        marker=markers[i % len(markers)],
        markersize=12,
        linewidth=2,
        label=label
    )

# --------- LABELS ---------
plt.xlabel(r"Iteration")
#plt.yscale("log")
plt.ylabel(r"Error, mHa/atom")
ax = plt.gca()
#ax.yaxis.set_major_formatter(ScalarFormatter())
#ax.yaxis.get_major_formatter().set_scientific(False)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
title_text = (rf"{method} Error per Iteration""\n" 
              fr" {title_name}, {Basis} ") 
#if fixed_rank_text != "":
    #title_text += ", " + fixed_rank_text

plt.title(title_text)
plt.grid(True, which = "both", linestyle="--", alpha=0.6)
plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    title= scan_rank)

plt.tight_layout()
plt.savefig(
    os.path.join(energy_folder, f"{method}_energy_diff_iter_plot_{basis}_{mol_name}_{scan}_minus_1st_iter_new.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
#"""
