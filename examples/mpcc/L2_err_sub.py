import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# --------------------------------------------------
# Argument parsing (kept same spirit as your code)
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot CC2 Y-amplitude L2 error (2x2: molecule × basis)"
)
parser.add_argument(
    "--scan", type=str, required=True,
    choices=["Lvv", "Lov", "Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],
    help="Which tensor to scan"
)
parser.add_argument("--tensor",type=str,required=True, choices=["Y", "Ω"], help="tensor Y or Omega")

parser.add_argument(
    "--results", type=str, default=None,
    help="Path to results folder"
)
parser.add_argument(
    "--relative", action="store_true",
    help="Plot relative L2 error"
)

args = parser.parse_args()

scan = args.scan
tensor = args.tensor
relative = args.relative

molecules = ["TIP4P-6", "c6h14"]
bases = ["cc-pvdz", "cc-pvtz"]

basis_label = {
    "cc-pvdz": "DZ",
    "cc-pvtz": "TZ",
}
if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"
# --------------------------------------------------
# Molecule → LaTeX
# --------------------------------------------------
def mol_to_latex(mol):
    if mol.startswith("TIP4P-2"):
        return r"$\mathrm{(H_2O)_2}$"
    m = re.match(r"c(\d+)h(\d+)", mol)
    if m:
        return rf"$\mathrm{{C_{{{m.group(1)}}}H_{{{m.group(2)}}}}}$"
    return mol

# --------------------------------------------------
# Matplotlib style (journal friendly)
# --------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

# --------------------------------------------------
# Main loops
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

        if tensor == "Y":
            Y_folder = results / basis / mol / f"Y_amp_{scan}"
        else:
            Y_folder = results / basis / mol / f"{tensor}_{scan}"

        if not Y_folder.is_dir():
            ax.set_visible(False)
            continue

        # --- Load DF reference ---
        Y_DF = np.load(Y_folder / f"CC2_{tensor}_DF.npy")

        cpd_files = sorted([
            f for f in os.listdir(Y_folder)
            if f.startswith(f"CC2_{tensor}_CPD") and f.endswith(".npy")
        ])

        ranks = []
        errors = []

        for f in cpd_files:
            Y_CPD = np.load(Y_folder / f)

            err = np.linalg.norm(Y_DF - Y_CPD) * 100
            if relative:
                err /= np.linalg.norm(Y_DF)

            m = re.search(r"rank([0-9.]+)X", f)
            if m:
                ranks.append(float(m.group(1)))
                errors.append(err)

        if len(ranks) == 0:
            ax.set_visible(False)
            continue

        ranks, errors = zip(*sorted(zip(ranks, errors)))

        # --- Plot ---
        ax.plot(ranks, errors, marker="o", linewidth=2)

# --- Apply per-row scaling ---
        if tensor == "Y":
            ax.set_yscale("linear")
            #ax.set_yticks([0.6,0.8,1,1.2,1.4,1.6,1.8,2]) 
        else:
            ax.set_yscale("linear")
            #ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) 

        ax.grid(True, linestyle="--", alpha=0.6)

        ax.set_title(
            rf"{mol_to_latex(mol)}, {basis_label[basis]}" + "\n" + fixed_rank_text
        )

# --------------------------------------------------
# Axis labels
# --------------------------------------------------
for ax in axs[-1, :]:
    ticks = [1, 1.5, 2, 2.5, 3, 3.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}X" for t in ticks])
    ax.set_xlabel(f"{scan_rank}")

ylabel = "Relative Percent Error" if relative else "Absolute Percent Error"
for ax in axs[:, 0]:
    ax.set_ylabel(ylabel)

# --------------------------------------------------
# Global title
# --------------------------------------------------
fig.suptitle(
    rf"$L_2$ Error of CC2 {tensor} vs {scan_rank}",
    fontsize=18,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_file = results / f"L2_{tensor}_error_2x2_{scan}.png"
plt.savefig(out_file, dpi=300)
plt.show()

print(f"Saved figure: {out_file}")

