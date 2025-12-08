import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser(description="Plot Error stats of water cluster")
parser.add_argument("--scan", type=str,required=True, choices=["Lov","Lvv","Lov_Lvv"],help="which tensor to scan:Lov etc")
args = parser.parse_args()

scan = args.scan
# ----------------------------
# User-defined settings
# ----------------------------
#molecules = ["TIP4P-1","TIP4P-2", "TIP4P-4","TIP4P-6", "TIP4P-8", "TIP4P-10"]
molecules = ["c2h6", "c4h10", "c6h14"] 
ranks = [1.0, 1.5, 2.0,2.5, 3.0,3.5]          # Adjust according to your data
results = "/home/talha/Documents/RR_MPCC/All_output_data/cc-pvtz"
# ----------------------------

# Dictionary: rank → list of error vectors (one per molecule)
rank_to_diffs = {r: [] for r in ranks}
non_H_atoms = {
    "TIP4P-1": 1,
    "TIP4P-2": 2,
    "TIP4P-4": 4,
    "TIP4P-6": 6,
    "TIP4P-8": 8,
    "TIP4P-10": 10,
    "c2h6": 2,
    "c4h10": 4,
    "c6h14": 6,
    "c8h18": 8,
    "c10h22": 10

}
# Loop over molecules
for mol in molecules:
    
    n_atoms = non_H_atoms[mol]
    print(f"no of non-H-atom:{n_atoms}")
    energy_folder = os.path.join(results, mol, f"energies_{scan}")

    # --- Load DF reference file ---
    df_file = os.path.join(energy_folder, "CC2_iter_energies_DF.txt")
    df_energy = np.loadtxt(df_file)

    # --- Loop over ranks ---
    for r in ranks:
        cpd_file = os.path.join(energy_folder,f"CC2_iter_energies_CPD_{scan}_rank{r}X.txt")

        iter_energies = np.loadtxt(cpd_file)
        print(f"no of iterations:{iter_energies.shape}")
        #min_len = min(len(df_energy), len(iter_energies))
        #diff = np.abs(df_energy[:min_len] - iter_energies[:min_len]) * 1000
        diff = abs(df_energy - iter_energies) * 1000  
        diff /= n_atoms

        rank_to_diffs[r].append(diff)

# -------------------------------------------------------------
# Compute statistics across molecules for each rank
# -------------------------------------------------------------
plt.figure(figsize=(11, 8))
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

colors = ["tab:blue", "tab:orange", "tab:green", 
          "tab:red", "tab:purple", "tab:brown"]

for i,r in enumerate(ranks):
    if len(rank_to_diffs[r]) == 0:
        continue

    # list of lists for all molecules at this rank
    lists = rank_to_diffs[r]
    
    # find minimum number of iterations among all molecules
    min_len = min(len(x) for x in lists)

    # trim all to min_len
    trimmed = [x[:min_len] for x in lists]

    arr = np.array(trimmed)   # now shape = (num_molecules, min_len)
    print(arr.shape)
    mean_diff = arr.mean(axis=0)
    min_diff  = arr.min(axis=0)
    max_diff  = arr.max(axis=0)

    # then plot mean/min/max for this rank
    iterations = np.arange(1, len(mean_diff) + 1)
    color = colors[i % len(colors)]
    yerr = np.vstack([mean_diff - min_diff, max_diff - mean_diff])
    
    label=f"{r}X".replace("1X", "X").replace(".0X", "X")

    plt.errorbar(iterations, mean_diff, yerr=yerr,
                 fmt='-o', linewidth=2, markersize=10,
                 color=color, capsize=12,capthick=1.5, label=label) 


    #plt.plot(iterations, mean_diff, linewidth=2, marker='o',color=color, label=f"Rank {r}")
    #plt.plot(iterations, max_diff, label="Max", linestyle=":",color=color,linewidth=2)
    #plt.plot(iterations, min_diff, label="Min", linestyle="--",color=color,linewidth=2)
    #plt.fill_between(iterations, min_diff, max_diff, alpha=0.2)
if scan == "Lvv":     
    fixed_rank_text = r"$R_{\mathrm{ov}} = 1.5X$"
    scan_rank = r"$R_{\mathrm{vv}}$"
elif scan == "Lov":    
    fixed_rank_text = r"$R_{\mathrm{vv}} = 2.5X$"
    scan_rank = r"$R_{\mathrm{ov}}$"
else:
    fixed_rank_text = ""
    scan_rank = f"CP_rank"

# Plot formatting
plt.ylim(1e-3, 1e-0)
plt.yticks([1e-3, 1e-2, 1e-1, 1e0])
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Error, mH/atom")
title_text =("CC2 Energy Error"
          "\nCarbon Chains, DZ, "
          r"$R_{\mathrm{oo}} = 0.5X$")
if fixed_rank_text != "":
    title_text += ", " + fixed_rank_text

plt.title(title_text)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title= scan_rank)
plt.tight_layout()

out_file = os.path.join(results, f"CC2_carbon_chains_stats_{scan}.png")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()

print("\nSaved plot to:", out_file)

