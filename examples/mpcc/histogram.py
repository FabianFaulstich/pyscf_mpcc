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
parser.add_argument("--tensor",type=str,required=True, choices=["Y", "Ω"], help="tensor Y or Omega")
parser.add_argument(
    "--molecule", type=str, required=True,
    choices=["water_clusters", "carbon_chains"],
    help="Which systems to pick e.g water_clusters"
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
parser.add_argument(
    "--rank", type=str, default="2.5X",
    help="CP rank to plot (default: 2.5X)"
)

args = parser.parse_args()

basis = args.basis
molecule = args.molecule
tensor = args.tensor
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
if molecule == "water_clusters":
    molecules = [f"TIP4P-{i}" for i in range(1, 11)]
else:
    molecules = ["c2h6", "c3h8","c4h10", "c5h12","c6h14","c8h18","c10h22"] 
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
    m = re.match(r"c(\d+)h(\d+)$", mol_name)
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

L_rank =  r"$R_{{oo}}= X$" if "_fix_Loo_1" in scan else  r"$R_{{oo}}=0.5X$"
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
    fixed_rank_text = ""
    scan_rank = r"CP rank"

plt.figure(figsize=(10, 7))

# --------------------------------------------------
# Fixed bins for all histograms
# --------------------------------------------------
bins = np.logspace(-16, -1.2, 120)

# --------------------------------------------------
# Main loop: fixed rank, varying molecule size
# --------------------------------------------------
for mol_name in molecules:
    if tensor == "Y":
        Y_folder = results / basis / mol_name / f"Y_amp_{scan}"
    else:
        Y_folder = results / basis / mol_name / f"{tensor}_{scan}"

    if not Y_folder.is_dir():
        print(f"[skip] {mol_name}: folder not found")
        continue

    # Pick CPD or DF file with the desired rank
    Y_df_files = [
        f for f in os.listdir(Y_folder)
        if f.startswith(f"CC2_{tensor}_DF") and f.endswith(".npy")
    ]

    Y_cpd_files = [
        f for f in os.listdir(Y_folder)
        if f.startswith(f"CC2_{tensor}_CPD")
        and target_rank in f
        and f.endswith(".npy")
    ]

    if len(Y_df_files) == 0 or len(Y_cpd_files) == 0:
        print(f"[skip] {mol_name}: missing DF or CPD data")
        continue    
    Y_df = np.load(Y_folder / Y_df_files[0])
    Y_cpd = np.load(Y_folder / Y_cpd_files[0])
    if Y_df.shape != Y_cpd.shape:
        print(f"[skip] {mol_name}: shape mismatch DF {Y_df.shape} vs CPD {Y_cpd.shape}")
        continue

    # --- Difference ---
    Y_diff = Y_df - Y_cpd
    Y_abs = np.abs(Y_diff.ravel())
    Y_abs = Y_abs[Y_abs > 0]

    if Y_abs.size == 0:
        print(f"[skip] {mol_name}: empty difference")
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
plt.xlim(1e-9, 1e-4)
plt.xticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])

plt.xlabel(f"Approx. {tensor} Value")
plt.ylabel("Frequency")

plt.title(
    f"Histogram of CC2 {tensor} at Fixed Rank\n"
    rf"{Basis}, {scan_rank}={target_rank}, " + fixed_rank_text
)

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(title=f"{molecule}", ncol=2)
plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------
outfile = results / basis / f"Histogram_{molecule}_{tensor}_rank_{target_rank}_{scan}_{basis}.png"
plt.savefig(outfile, dpi=300)
print(f"\nSaved figure: {outfile}")

plt.show()

