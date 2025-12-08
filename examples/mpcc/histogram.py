import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Histogram of CC2 Y amplitudes at fixed CP rank across water clusters"
)
parser.add_argument("basis", type=str, help="Basis set, e.g. cc-pvdz")
parser.add_argument(
    "--scan", type=str, required=True,
    choices=["Lvv", "Lov", "Lov_Lvv"],
    help="Which tensor to scan"
)
parser.add_argument(
    "--results", type=str, default=None,
    help="Path to results folder"
)
parser.add_argument(
    "--rank", type=str, default="2.5X",
    help="CP rank to plot (default: 2.5X)"
)

args = parser.parse_args()

basis = args.basis
scan = args.scan
target_rank = args.rank

# --------------------------------------------------
# Basis names for plotting
# --------------------------------------------------
basis_to_mol = {
    "cc-pvdz": "DZ",
    "cc-pvtz": "TZ",
    "aug-cc-pvdz": "aVDZ",
    "aug-cc-pvtz": "aVTZ",
}
Basis = basis_to_mol.get(basis, basis)

# --------------------------------------------------
# Results path
# --------------------------------------------------
if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

# --------------------------------------------------
# Molecule list: TIP4P-2 ... TIP4P-10
# --------------------------------------------------
#molecules = [f"TIP4P-{i}" for i in range(1, 11)]
molecules = ["c2h6", "c4h10", "c6h14","c8h18","c10h22"] 
# --------------------------------------------------
# Molecule → LaTeX
# --------------------------------------------------
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
# --------------------------------------------------
# Matplotlib style
# --------------------------------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 11,
})

plt.figure(figsize=(10, 7))

# --------------------------------------------------
# Fixed bins for all histograms
# --------------------------------------------------
bins = np.logspace(-16, -1.2, 120)

# --------------------------------------------------
# Main loop: fixed rank, varying molecule size
# --------------------------------------------------
for mol_name in molecules:

    Y_folder = results / basis / mol_name / f"Y_amp_{scan}"
    if not Y_folder.is_dir():
        print(f"[skip] {mol_name}: folder not found")
        continue

    # Pick CPD or DF file with the desired rank
    Y_files = [
        f for f in os.listdir(Y_folder)
        if f.endswith(".npy")
        and target_rank in f
        and (f.startswith("CC2_Y_CPD") or f.startswith("CC2_Y_DF"))
    ]

    if len(Y_files) == 0:
        print(f"[skip] {mol_name}: no rank {target_rank} file")
        continue

    # Assume one file per molecule per rank
    file_path = Y_folder / Y_files[0]
    Y = np.load(file_path)

    Y_abs = np.abs(Y.ravel())
    Y_abs = Y_abs[Y_abs > 0]

    if Y_abs.size == 0:
        print(f"[skip] {mol_name}: empty tensor")
        continue

    plt.hist(
        Y_abs,
        bins=bins,
        histtype="step",
        linewidth=3,
        label=mol_to_latex(mol_name),
    )

    print(f"[ok] {mol_name} | {Y_abs.size} entries")

# --------------------------------------------------
# Plot formatting
# --------------------------------------------------
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-7, 1e-1)
plt.xticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

plt.xlabel("Y Value")
plt.ylabel("Frequency")

plt.title(
    "Histogram of CC2 Y-amplitude at Fixed Rank\n"
    rf"{Basis}, CP rank={target_rank}, $R_{{oo}}=0.5X$"
)

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(title="Carbon Chains", ncol=2)
plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------
outfile = results / basis / f"Histogram_carbon_Y_rank_{target_rank}_{scan}_{basis}.png"
plt.savefig(outfile, dpi=300)
print(f"\nSaved figure: {outfile}")

plt.show()

