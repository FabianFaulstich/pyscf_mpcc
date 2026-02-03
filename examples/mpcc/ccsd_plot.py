import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

# ---------------- ARGPARSE ----------------
parser = argparse.ArgumentParser(
    description="Complete dissociation energy error vs CPD rank (DF reference)"
)
parser.add_argument("basis", type=str)
parser.add_argument("--scan", required=True,
    choices=["Lvv","Lov","Lov_Lvv","Lov_fix_Loo_1",
             "Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"])
parser.add_argument("--results", type=str, default=None)
parser.add_argument("--nmax", type=int, default=10)
parser.add_argument("method", type=str)
args = parser.parse_args()

basis = args.basis
scan = args.scan
nmax = args.nmax
method = args.method
# ---------------- PATH ----------------
if args.results:
    results = Path(os.path.expanduser(args.results)).resolve()
else:
    results = Path(__file__).parent / "output_data"

HARTREE_TO_KCAL = 627.509474
basis_to_mol = {
            "cc-pvdz": "DZ",
            "cc-pvtz": "TZ",
            "aug-cc-pvdz": "aVDZ",
            "aug-cc-pvtz": "aug-cc-pvtz"
            }
Basis = basis_to_mol[basis]
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

# ---------------- FIND ALL CPD RANKS FROM MONOMER ----------------
mono_folder = results / basis / "TIP4P-1" / f"energies_{method}"
if not mono_folder.exists():
    raise RuntimeError("Monomer folder missing — cannot detect CPD ranks")

rank_pattern = re.compile(r"rank([\d\.]+X)")
rank_files = [
    f for f in os.listdir(mono_folder)
    if f.startswith(f"{method}_iter_energies") and "rank1.0X" not in f
]

cpd_ranks = sorted({
    rank_pattern.search(f).group(1)
    for f in rank_files
    if rank_pattern.search(f)
})

print("Detected CPD ranks:", cpd_ranks)

# ============================================================
# NEW: LOAD DF ENERGIES
# ============================================================
df_energy = {}

for n in range(1, nmax + 1):
    mol = f"TIP4P-{n}"
    folder = results / basis / mol / f"energies_{method}"

    if not folder.exists():
        continue

    df_files = [
        f for f in os.listdir(folder)
        if f.startswith(f"{method}_iter_energies") and ("DF") in f
    ]

    if not df_files:
        continue

    data = np.loadtxt(folder / df_files[0])
    df_energy[n] = data[-1]

if 1 not in df_energy:
    raise RuntimeError("DF monomer energy missing — cannot compute reference")
# ---------------- DF DISSOCIATION ENERGY ----------------
D_df = {}
E1_df = df_energy[1]

for n, En in df_energy.items():
    D_df[n] = (En - n * E1_df) * HARTREE_TO_KCAL

# ============================================================
# NEW: LOAD CCSD ENERGIES
# ============================================================


ccsd_energy = {}

for n in range(1, nmax + 1):
    mol = f"TIP4P-{n}"
    folder = results / basis / mol / f"energies_CC_SD"

    if not folder.exists():
        continue

    ccsd_files = [
        f for f in os.listdir(folder)
        if ("DF") in f
    ]

    if not ccsd_files:
        continue

    data = np.loadtxt(folder / ccsd_files[0])
   
    ccsd_energy[n] = float(np.atleast_1d(data)[-1])
if 1 not in ccsd_energy:
    raise RuntimeError("DF monomer energy missing — cannot compute reference")
# ---------------- CCSD DISSOCIATION ENERGY ----------------
D_ccsd = {}
E1_ccsd = ccsd_energy[1]

for n, En in ccsd_energy.items():
    D_ccsd[n] = (En - n * E1_ccsd) * HARTREE_TO_KCAL

# ============================================================
# LOAD CPD ENERGIES
# ============================================================
final_energy = {rank: {} for rank in cpd_ranks}

for rank in cpd_ranks:
    for n in range(1, nmax + 1):
        mol = f"TIP4P-{n}"
        folder = results / basis / mol / f"energies_{method}"

        if not folder.exists():
            continue

        files = [
                f for f in os.listdir(folder)
                if f.startswith(f"{method}_iter_energies_CPD_Lvv")
                and f"rank{rank}" in f
                #and not any(r in f for r in ("rank2.0", "rank3.0"))
                ]


        if not files:
            continue

        data = np.loadtxt(folder / files[0])
        final_energy[rank][n] = data[-1]

# ---------------- CPD DISSOCIATION ENERGY ----------------
D_complete = {}

for rank, energies in final_energy.items():
    if 1 not in energies:
        continue

    E1 = energies[1]
    D_complete[rank] = {}

    for n, En in energies.items():
        D_complete[rank][n] = (En - n * E1) * HARTREE_TO_KCAL

# ============================================================
# NEW: ERROR RELATIVE TO DF
# ============================================================
"""
DF_error = {
    n: abs(D_ccsd[n] - D_df[n])
    for n in D_df if n in D_ccsd
}
"""

CPD_error = {}

for rank, Dn in D_complete.items():
    CPD_error[rank] = {
        n: (D_df[n] - Dn[n])
        for n in Dn if n in D_df
    }
# ---------------- PLOT ERROR ----------------
plt.figure(figsize=(6.5, 6))

markers = ["s", "^", "D", "v", "P", "X"]
"""
# ---- DF baseline ----
n_vals = sorted(DF_error.keys())
err_vals = [DF_error[n] for n in n_vals]

plt.plot(
    n_vals,
    err_vals,
    color = "black",
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=8,
    label=r"MPCC"
)
#"""
# ---- CPD ranks ----
for i, (rank, Derr) in enumerate(CPD_error.items()):
    n_vals = sorted(Derr.keys())
    err_vals = [Derr[n] for n in n_vals]
    label = (
        rank.replace(".0X", "X")
         )
    plt.plot(
        n_vals,
        err_vals,
        marker=markers[i % len(markers)],
        linewidth=2,
        markersize=8,
        label=rf" {label}"
        #label=r"$R_{vv}=$" rf" {label}"
    )
#plt.axhline(0.0, color="black", linestyle="--", linewidth=1)

plt.xlabel(r"$n, (H_{2}O)_n$")
#plt.yticks([1.5,1.2,0.9,0.6,0.3,0,-0.3,-0.6])
#plt.yticks([0.6,0.3,0,-0.3,-0.6])
plt.yticks([3,2.5,2,1.5,1,0.5,0,-0.5])

plt.ylabel(r"Error (kcal/mol)")
plt.title(rf"Error in Dissociation Energy""\n"
          rf" Reference = {method}, {Basis}/{Basis}-RI")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc='best',labelspacing=0.1,handletextpad=0.5,title= r"$R_{vv}$")
#plt.legend(loc='best',labelspacing=0.1,handletextpad=0.5)
plt.tight_layout()

out = results / basis / f"dissociation_error_{method}_DF_all_ranks_{basis}_{scan}_signed.png"
plt.savefig(out, dpi=300)
plt.show()

print(f"Saved plot to: {out}")
###############
