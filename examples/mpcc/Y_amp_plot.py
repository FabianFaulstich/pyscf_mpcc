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
parser.add_argument("--scan",type=str,required=True, choices=["Lvv","Lov","Lov_Lvv"],help="which tensor to scan:Lov etc")
parser.add_argument("--results", type=str, default=None, help="path to result folder")
parser.add_argument("--relative", action="store_true",
                    help="Plot relative L2 error")
parser.add_argument("--tensor",type=str,required=True, choices=["Y","Foo","Fov","Fvv"],help="which tensor to scan:Y etc")

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



##############################
#   HISTOGRAM FOR Y VALUES  #
##############################
if tensor == "Y":
    Y_folder = os.path.join(results, basis, mol_name, f"Y_amp_{scan}")
else:
    Y_folder = os.path.join(results, basis, mol_name, f"{tensor}_{scan}")

# Load all Y files
Y_files = sorted([
    f for f in os.listdir(Y_folder)
    if (f.startswith(f"CC2_{tensor}_CPD") or f.startswith(f"CC2_{tensor}_DF")) and f.endswith(".npy")
])
# LaTeX-friendly font sizes
plt.rcParams.update({
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
    print(f"shape of Y:{Y.shape}")
    Y_flat = Y.flatten()           # flatten to 1D
    Y_abs = np.abs(Y_flat)
    Y_abs = Y_abs[Y_abs > 0]
    print(f"shape of flaten Y:{Y_flat.shape}")
    # clean label
    label = (
        f.replace(f"CC2_{tensor}_", "")
         .replace(f"CPD_{scan}_rank", "")
         .replace(".npy", "")
         .replace("1X", "X")
         .replace(".0X", "X")
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
plt.title(f"Histogram of Flattened {tensor}\n"
          f" {title_name}, {Basis}, "
          r"$R_{\mathrm{oo}} = X$")


plt.grid(True,which="both", linestyle="--", alpha=0.4)
plt.legend(title="CP Rank")
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
df_file = os.path.join(Y_folder,f"CC2_{tensor}_DF.npy")
Y_DF = np.load(df_file)

cpd_files = sorted([f for f in os.listdir(Y_folder) if f.startswith(f"CC2_{tensor}_CPD") and f.endswith(".npy")])

ranks =[]
errors = []

for f in cpd_files:
    file_path = os.path.join(Y_folder,f)
    Y_CPD = np.load(file_path)

    diff_norm = np.linalg.norm(Y_DF - Y_CPD) * 100
    if args.relative:
        diff_norm /= np.linalg.norm(Y_DF)

    m = re.search(r"rank([0-9.]+)X",f)
    rank = float(m.group(1)) if m else None
    
    print(f"error {f}: {diff_norm}")
    ranks.append(rank)
    errors.append(diff_norm)

ranks, errors = zip(*sorted(zip(ranks,errors)))

# ----------------- PLOT -----------------
plt.figure(figsize=(10, 7))
plt.plot(ranks, errors, marker="o", linewidth=2)

plt.xlabel("CP rank")
ticks = [1, 1.5, 2, 2.5, 3, 3.5]
labels = [f"{t:g}X" for t in ticks]

plt.xticks(ticks, labels)

ylabel = "Absolute Percent Error"
if args.relative:
    ylabel = "Relative Percent Error"

plt.ylabel(ylabel)
#plt.yscale("log")

plt.title(
    rf"{tensor} $L_2$ Percent Error vs Rank"
    "\n"
    rf"{title_name}, {Basis}"
)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

out_file = os.path.join(
    Y_folder, f"{tensor}_L2_error_{basis}_{mol_name}_{scan}.png"
)
plt.savefig(out_file, dpi=300)
plt.show()


