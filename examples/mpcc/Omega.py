import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

parser = argparse.ArgumentParser(description="L2 error of Omega tensor vs CP rank")
parser.add_argument("basis", type=str)
parser.add_argument("molecule", type=str)
parser.add_argument("--scan", type=str, required=True,
                    choices=["Lvv", "Lov", "Lov_Lvv"])
parser.add_argument("--results", type=str, default=None)
parser.add_argument("--relative", action="store_true",
                    help="Plot relative L2 error")
args = parser.parse_args()

basis = args.basis
mol_name = args.molecule
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

# ----------------- PATHS -----------------
if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

omega_folder = os.path.join(results, basis, mol_name, f"Ω_{scan}")

df_file = os.path.join(omega_folder,"CC2_Ω_DF.npy")
Omega_DF = np.load(df_file)

cpd_files = sorted([f for f in os.listdir(omega_folder) if f.startswith("CC2_Ω_CPD") and f.endswith(".npy")])

ranks =[]
errors = []

for f in cpd_files:
    file_path = os.path.join(omega_folder,f)
    Omega_CPD = np.load(file_path)

    diff_norm = np.linalg.norm(Omega_DF - Omega_CPD) * 100
    if args.relative:
        diff_norm /= np.linalg.norm(Omega_DF)

    m = re.search(r"rank([0-9.]+)X",f)
    rank = float(m.group(1)) if m else None
    
    print(f"error {f}: {diff_norm}")
    ranks.append(rank)
    errors.append(diff_norm)

ranks, errors = zip(*sorted(zip(ranks,errors)))

# ----------------- PLOT -----------------
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
    r"Ω $L_2$ Percent Error vs Rank"
    "\n"
    rf"{title_name}, {Basis}"
)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

out_file = os.path.join(
    omega_folder, f"Omega_L2_error_{basis}_{mol_name}_{scan}.png"
)
plt.savefig(out_file, dpi=300)
plt.show()

# ----------------------------
# DEbuging
# ----------------------------


abs_Omega = np.abs(Omega_DF)
print("max |Ω| =", abs_Omega.max())
print("min |Ω| (nonzero) =", abs_Omega[abs_Omega > 0].min())
log_Omega = np.log10(abs_Omega + 1e-16)

print("log10 max =", log_Omega.max())
print("log10 min =", log_Omega.min())


##############################
#   HISTOGRAM FOR Y VALUES  #
##############################

# Load all Y files
Omega_files = sorted([
    f for f in os.listdir(omega_folder)
    if (f.startswith("CC2_Ω_CPD") or f.startswith("CC2_Ω_DF")) and f.endswith(".npy")
])

plt.figure(figsize=(10, 7))

for i, f in enumerate(Omega_files):
    file_path = os.path.join(omega_folder, f)
    Ω = np.load(file_path)         # load tensor
    print(f"shape of Ω:{Ω.shape}")
    Ω_flat = Ω.flatten()           # flatten to 1D
    Ω_abs = np.abs(Ω_flat)
    Ω_abs = Ω_abs[Ω_abs > 0]
    print(f"shape of flaten Ω:{Ω_flat.shape}")
    # clean label
    label = (
        f.replace("CC2_Ω_", "")
         .replace(f"CPD_{scan}_rank", "")
         .replace(".npy", "")
         .replace("1X", "X")
         .replace(".0X", "X")
    )

    # Plot histogram of values
    #bins = np.logspace(np.log10(Y_abs.min()),np.log10(Y_abs.max()),120)
    bins = np.logspace(-16, -1.2, 120)


    plt.hist(Ω_abs,bins=bins,label=label,histtype='step',linewidth=3)
plt.xlim(1e-7, 1e-1)
plt.xticks([ 1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ω Value")
plt.ylabel("Frequency")
plt.title("Histogram of Ω-values\n"
          f" {title_name}, {Basis}, "
          r"$R_{\mathrm{oo}} = 0.5X$")


plt.grid(True,which="both", linestyle="--", alpha=0.4)
plt.legend(title="CP Rank")
plt.tight_layout()

# Save histogram
plt.savefig(
    os.path.join(omega_folder, f"Histogram_Ω_values_{mol_name}_{basis}.png"),
    dpi=300
)
plt.show()

