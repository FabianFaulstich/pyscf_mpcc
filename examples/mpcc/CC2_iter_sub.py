import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot CC2 iteration energies (2x2: basis × molecule)"
)
parser.add_argument(
    "--bases", nargs=2, required=True,
    help="Two basis sets, e.g. cc-pvdz cc-pvtz"
)
parser.add_argument(
    "--molecules", nargs=2, required=True,
    help="Two molecules, e.g. TIP4P-2 C2H6"
)
parser.add_argument(
    "--scan", type=str, required=True,
    choices=["Lvv", "Lov", "Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],
    help="Which tensor to scan"
)
parser.add_argument(
    "--results", type=str, default=None,
    help="Path to results folder"
)
args = parser.parse_args()

bases = args.bases
molecules = args.molecules
scan = args.scan

basis_to_label = {
    "cc-pvdz": "DZ",
    "cc-pvtz": "TZ",
    "aug-cc-pvdz": "aVDZ",
    "aug-cc-pvtz": "aVTZ",
}

# --------------------------------------------------
# Paths
# --------------------------------------------------
if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

# --------------------------------------------------
# Molecule → LaTeX
# --------------------------------------------------
def mol_to_latex(mol):
    m = re.match(r"TIP4P-(\d+)", mol)
    if m:
        n = int(m.group(1))
        return r"$\mathrm{H_2O}$" if n == 1 else rf"$\mathrm{{(H_2O)_{{{n}}}}}$"

    m = re.match(r"c(\d+)h(\d+)$", mol)
    if m:
        return rf"$\mathrm{{C_{{{m.group(1)}}}H_{{{m.group(2)}}}}}$"

    return rf"$\mathrm{{{mol}}}$"

# --------------------------------------------------
# Matplotlib style
# --------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})
scan_str = str(scan).replace("_fix_Loo_1", "")

markers = ["o", "s", "D", "^", "v", "P", "X"]

fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

# --------------------------------------------------
# Main loop: basis (rows) × molecule (cols)
# --------------------------------------------------
for i, basis in enumerate(bases):
    for j, mol in enumerate(molecules):
        ax = axs[i, j]
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

        energy_folder = results / basis / mol / f"energies_{scan}"
        if not energy_folder.is_dir():
            ax.set_visible(False)
            continue

        energy_files = sorted([
            f for f in os.listdir(energy_folder)
            if f.startswith("CC2_iter_energies")
        ])

        for k, f in enumerate(energy_files):
            data = np.loadtxt(energy_folder / f)
            iterations = np.arange(1, len(data) + 1)

            label = (
                f.replace("CC2_iter_energies_", "")
                 .replace(f"CPD_{scan_str}_rank", "")
                 .replace(".txt", "")
                 .replace(".0X", "X")
            )

            ax.plot(
                iterations,
                data,
                marker=markers[k % len(markers)],
                linewidth=2,
                markersize=8,
                label=label
            )

        # Titles
        ax.set_title(
    rf"{mol_to_latex(mol)}, {basis_to_label[basis]}" + "\n" + fixed_rank_text
)


        ax.grid(True, linestyle="--", alpha=0.6)

        if i == 1:
            ax.set_xlabel("Iteration")
        if j == 0:
            ax.set_ylabel("Energy (Ha)")

# --------------------------------------------------
# Legend & super title
# --------------------------------------------------
axs[0, 1].legend(title=f"{scan_rank}", loc="best")

fig.suptitle("CC2 Energy Convergence vs Iteration" + "\n",
    fontsize=18
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

out_file = results / f"CC2_iteration_2x2_{scan}.png"
plt.savefig(out_file, dpi=300)
plt.show()

print(f"Saved figure: {out_file}")

