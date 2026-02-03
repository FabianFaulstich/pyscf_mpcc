import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
#import seaborn as sns

parser = argparse.ArgumentParser(description="Plot CC2 Y_amplitudes for a molecule and basis")
parser.add_argument("basis", type=str, help="Basis set, e.g., cc-pvdz")
parser.add_argument("molecules",nargs="+",type=str,help="One or more molecule names, e.g., H2O TIP4P-6")
parser.add_argument("--scan",type=str,required=True, choices=["Lvv","Lov","Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],help="which tensor to scan:Lov etc")
parser.add_argument("--results", type=str, default=None, help="path to result folder")
parser.add_argument("--tensor",type=str,required=True, choices=["Y","Ω","Foo","Fov","Fvv"],help="which tensor to scan:Y etc")

args = parser.parse_args()

molecules = args.molecules
basis = args.basis
scan = args.scan
tensor = args.tensor
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
    return rf"${{{mol_name}}}$"
#title_name = mol_to_latex(mol_name)

if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

if scan == "Lvv":
    if basis == "cc-pvdz":
        fixed_rank_text = r"$R_{{oo}} = 0.5X,\; R_{{ov}} = 1.5X$"
    else: 
         fixed_rank_text = r"$R_{{oo}} = 0.5X,\; R_{{ov}} = 2X$"
    scan_rank = r"$R_{\mathrm{vv}}$"

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
    scan_rank = r"$R_{ov}$, $R_{vv}$"
else:
    fixed_rank_text = r"$R_{{oo}} = 0.5X$"
    scan_rank = r"CP rank"


def compute_l2_errors(mol_name):
    title_name = mol_to_latex(mol_name)

    if tensor == "Y":
        Y_folder = os.path.join(results, basis, mol_name, f"Y_amp_{scan}")
    else:
        Y_folder = os.path.join(results, basis, mol_name, f"{tensor}_{scan}")

    # DF reference
    df_file = os.path.join(Y_folder, f"CC2_{tensor}_DF_2.npy")
    Y_DF = np.load(df_file)

    cpd_files = sorted(
        f for f in os.listdir(Y_folder)
        if f.startswith(f"CC2_{tensor}_CPD") and f.endswith("2.npy")
    )

    ranks = []
    errors = []

    for f in cpd_files:
        Y_CPD = np.load(os.path.join(Y_folder, f))

        diff_norm = np.linalg.norm(Y_DF - Y_CPD)
        diff_norm /= np.linalg.norm(Y_DF)
        diff_norm *= 100  # percent

        m = re.search(r"rank([0-9.]+)X", f)
        rank = float(m.group(1)) if m else None

        ranks.append(rank)
        errors.append(diff_norm)

    ranks, errors = zip(*sorted(zip(ranks, errors)))
    return ranks, errors, title_name
# LaTeX-friendly font sizes
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 22,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ----------------- PLOT -----------------
plt.figure(figsize=(6.5, 6))
markers = ["o", "s", "D", "^", "v", "P", "X"]

for i,mol in enumerate(molecules):
    ranks, errors, title_name = compute_l2_errors(mol)
    plt.plot(
        ranks,
        errors,
        marker=markers[i % len(markers)],
        markersize = 8,
        linewidth=2,
        label=title_name
    )

plt.xlabel(rf"{scan_rank}")
ticks = [ 1.5, 2, 2.5, 3, 3.5]
labels = [rf"{t:g}X" for t in ticks]

plt.xticks(ticks, labels)
yticks = [0.0, 0.5, 1,1.5, 2, 2.5,3,3.5]
ylabels = [rf"{t:g}" for t in yticks]
plt.yticks(yticks, ylabels)
ylabel = r"Relative Percent Error"

plt.ylabel(ylabel)
#plt.yscale("log")
#plt.yticks([0.6,0.8,1,1.2,1.4,1.6,1.8,2])

title_text = (
    rf"{tensor} $L_2$ Percent Error vs Rank"
    "\n"
    rf" {Basis}/{Basis}-RI"
)
#if fixed_rank_text != "":
    #title_text += ", " + fixed_rank_text

plt.title(title_text)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

out_file = os.path.join(results,f"{basis}", f"{tensor}_L2_percent_error_{basis}_{scan}.png")
plt.savefig(out_file, dpi=300)
plt.show()

