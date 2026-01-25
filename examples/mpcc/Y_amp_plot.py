import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
#import seaborn as sns

parser = argparse.ArgumentParser(description="Plot CC2 Y_amplitudes for a molecule and basis")
parser.add_argument("basis", type=str, help="Basis set, e.g., cc-pvdz")
parser.add_argument("molecule", type=str, help="Molecule name, e.g., H2O")
parser.add_argument("--scan",type=str,required=True, choices=["Lvv","Lov","Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],help="which tensor to scan:Lov etc")
parser.add_argument("--results", type=str, default=None, help="path to result folder")
parser.add_argument("--tensor",type=str,required=True, choices=["Y","Ω","Foo","Fov","Fvv"],help="which tensor to scan:Y etc")

args = parser.parse_args()

mol_name = args.molecule
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
title_name = mol_to_latex(mol_name)

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
    scan_rank = r"CP rank"
else:
    fixed_rank_text = r"$R_{{oo}} = 0.5X$"
    scan_rank = r"CP rank"


##############################
#   HISTOGRAM FOR Y VALUES  #
##############################
if tensor == "Y":
    Y_folder = os.path.join(results, basis, mol_name, f"Y_amp_{scan}")
else:
    Y_folder = os.path.join(results, basis, mol_name, f"{tensor}_{scan}")
scan_str = str(scan).replace("_fix_Loo_1", "")

# Load all Y files
Y_files = sorted([
    f for f in os.listdir(Y_folder)
    if (f.startswith(f"CC2_{tensor}_CPD") or f.startswith(f"CC2_{tensor}_DF")) and f.endswith(".npy")
])
# LaTeX-friendly font sizes
plt.rcParams.update({
    "font.family": "serif",
    #"font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 10,
})

plt.figure(figsize=(10, 7))

for i, f in enumerate(Y_files):
    file_path = os.path.join(Y_folder, f)
    Y = np.load(file_path)         # load tensor
    #print(f"shape of Y:{Y.shape}")
    Y_flat = Y.flatten()           # flatten to 1D
    Y_abs = np.abs(Y_flat)
    Y_abs = Y_abs[Y_abs > 0]
    #print(f"shape of flaten Y:{Y_flat.shape}")
    # clean label
    label = (
        f.replace(f"CC2_{tensor}_", "")
         .replace(f"CPD_{scan_str}_rank", "")
         .replace(".npy", "")
         .replace("1X", r"X")
         .replace(".0X", r"X")
    )

    # Plot histogram of values
    #bins = np.logspace(np.log10(Y_abs.min()),np.log10(Y_abs.max()),120)
    bins = np.logspace(-16, -1.2, 120)


    plt.hist(Y_abs,bins=bins,label=label,histtype='step',linewidth=3)
plt.xlim(1e-7, 1e-1)
plt.xticks([ 1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"{tensor} Value")
plt.ylabel("Count")
title_text = (f"Histogram of Flattened {tensor}\n"
          f" {title_name}, {Basis}, ")
if fixed_rank_text != "":
    title_text += "," + fixed_rank_text
plt.title(title_text)

plt.grid(True,which="both", linestyle="--", alpha=0.4)
plt.legend(title=f"{scan_rank}")
plt.tight_layout()

# Save histogram
plt.savefig(
    os.path.join(Y_folder, f"Histogram_{tensor}_values_{mol_name}_{basis}.png"),
    dpi=300
)
plt.show()
# ----------------------------
# L2 error plotting
# ----------------------------
df_file = os.path.join(Y_folder,f"CC2_{tensor}_DF_2.npy")
Y_DF = np.load(df_file)
print(f"norm of omega DF:{np.linalg.norm(Y_DF)}")

cpd_files = sorted([f for f in os.listdir(Y_folder) if f.startswith(f"CC2_{tensor}_CPD") and f.endswith("2.npy")])

ranks =[]
errors = []

for f in cpd_files:
    file_path = os.path.join(Y_folder,f)
    Y_CPD = np.load(file_path)
    #print(f"norm of omega CP:{np.linalg.norm(Y_CPD)}")
    diff_norm = np.linalg.norm(Y_DF - Y_CPD)
    #print(f"Absolute error {f}:{diff_norm}")
    #print(f"Percent absolute error:{diff_norm*100}")
    diff_norm /= np.linalg.norm(Y_DF)
    print(f"Relative error {f}:{diff_norm}")
    diff_norm *= 100

    m = re.search(r"rank([0-9.]+)X",f)
    rank = float(m.group(1)) if m else None
    
    print(f"Percent relative error {f}: {diff_norm}")
    ranks.append(rank)
    errors.append(diff_norm)

ranks, errors = zip(*sorted(zip(ranks,errors)))

# ----------------- PLOT -----------------
plt.figure(figsize=(10, 7))
plt.plot(ranks, errors, marker="o", linewidth=2)

plt.xlabel(rf"{scan_rank}")
ticks = [1, 1.5, 2, 2.5, 3, 3.5]
labels = [rf"${t:g}X$" for t in ticks]

plt.xticks(ticks, labels)
#yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#ylabels = [rf"${t:g}$" for t in yticks]
#plt.yticks(yticks, ylabels)
ylabel = r"Relative Percent Error"

plt.ylabel(ylabel)
#plt.yscale("log")
#plt.yticks([0.6,0.8,1,1.2,1.4,1.6,1.8,2])

title_text = (
    rf"{tensor} $L_2$ Percent Error vs Rank"
    "\n"
    rf"{title_name}, {Basis}"
)
if fixed_rank_text != "":
    title_text += ", " + fixed_rank_text

plt.title(title_text)


plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

out_file = os.path.join(
    Y_folder, f"{tensor}_L2_error_{basis}_{mol_name}_{scan}.png"
)
plt.savefig(out_file, dpi=300)
plt.show()


