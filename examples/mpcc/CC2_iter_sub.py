import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.optimize import curve_fit

# ---------------- ARGPARSE ----------------
parser = argparse.ArgumentParser(description="Final CPD energy and error vs system size")
parser.add_argument("basis", type=str, help="e.g. cc-pvdz")
parser.add_argument("--scan", type=str, required=True,
                    choices=["Lov", "Lvv", "Lov_Lvv",
                             "Lov_fix_Loo_1", "Lvv_fix_Loo_1", "Lov_Lvv_fix_Loo_1"])
parser.add_argument("method", type=str, help="e.g. CC2")
parser.add_argument("macro_it", type=int, help="1 or 2")
args = parser.parse_args()

basis = args.basis
scan = args.scan
method = args.method
macro_it = args.macro_it

# ---------------- GLOBALS ----------------
ERROR_TOL = 0.5  # mH / atom
results = f"/home/talha/Documents/RR_MPCC/All_output_data/{basis}"
scan_str = scan.replace("_fix_Loo_1", "")
basis_to_mol = {
            "cc-pvdz": "DZ",
            "cc-pvtz": "TZ",
            "aug-cc-pvdz": "aVDZ",
            "aug-cc-pvtz": "aug-cc-pvtz"
            }
Basis = basis_to_mol[basis]

ranks = [1.5, 2.5, 3.5]
# LaTeX-friendly font sizes
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

# ---------------- MOLECULE SETS ----------------
SYSTEMS = {
    "water_clusters": {
        "molecules": [
            "TIP4P-2","TIP4P-3","TIP4P-4",
            "TIP4P-5","TIP4P-6","TIP4P-8","TIP4P-10"
        ]},
    "carbon_chains": {
        "molecules": [
            "c2h6","c3h8","c4h10",
            "c5h12","c6h14","c8h18","c10h22"
        ]}
}

# ============================================================
# FUNCTIONS
# ============================================================
def system_size(mol, system):
    if system == "water_clusters":
        return int(mol.split("-")[-1])
    else:
        return int("".join(filter(str.isdigit, mol.split("h")[0])) or 1)


def load_data(system_name):
    molecules = SYSTEMS[system_name]["molecules"]

    rank_to_energy = {r: {} for r in ranks}
    rank_to_error  = {r: {} for r in ranks}
    df_energy = {}

    for mol in molecules:
        n_system = system_size(mol, system_name)

        energy_folder = os.path.join(results, mol, f"energies_{method}")
        if not os.path.exists(energy_folder):
            continue

        df_file = os.path.join(
            energy_folder,
            #f"{method}_iter_energies_DF.txt"
            f"{method}_iter_energies_DF_{macro_it}.txt"
        )
        if not os.path.exists(df_file):
            continue

        E_df = np.loadtxt(df_file)[-1]
        df_energy[n_system] = E_df

        for r in ranks:
            cpd_file = os.path.join(
                energy_folder,
                #f"{method}_iter_energies_CPD_{scan_str}_rank{r}X.txt"
                f"{method}_iter_energies_CPD_{scan_str}_rank{r}X_{macro_it}.txt"
            )
            if not os.path.exists(cpd_file):
                continue

            E_cpd = np.loadtxt(cpd_file)[-1]
            rank_to_energy[r][n_system] = E_cpd
            rank_to_error[r][n_system] = abs(E_df - E_cpd) * 1000 / n_system

    return df_energy, rank_to_energy, rank_to_error
def get_nao_naux(basis, molecules, n):
    """
    Return (nao, naux) for given basis, molecule type, and system size n
    """

    if basis == "cc-pvdz":
        if molecules == "water_clusters":
            nao = 24 * n
            naux = 116 * n

        elif molecules == "carbon_chains":
            nao = 34 + 24 * (n - 1)
            naux = 162 + 116 * (n - 1)

        else:
            raise ValueError("Unknown molecule type")

    elif basis == "cc-pvtz":
        if molecules == "water_clusters":
            nao = 58 * n
            naux = 139 * n

        elif molecules == "carbon_chains":
            nao = 86 + 58 * (n - 1)
            naux = 199 + 139 * (n - 1)

        else:
            raise ValueError("Unknown molecules type")

    else:
        raise ValueError(f"Unsupported basis: {basis}")

    return nao, naux


def compute_rank_scaling(rank_to_error,molecules):
    all_system_sizes = sorted(
        set().union(*[set(d.keys()) for d in rank_to_error.values()])
    )

    required_ranks = {}
    for n in all_system_sizes:
        eligible = []
        for r in sorted(rank_to_error.keys()):
            if n in rank_to_error[r]:
                if abs(rank_to_error[r][n]) <= ERROR_TOL:
                    eligible.append(r)
        if eligible:
            required_ranks[n] = min(eligible)
            print(f"rank:{required_ranks}")
    n_val = np.array(sorted(required_ranks.keys()))

    nao_vals = []
    naux_vals = []

    for n in n_val:
        nao, naux = get_nao_naux(basis, molecules, n)
        nao_vals.append(nao)
        naux_vals.append(naux)

    nao_vals = np.array(nao_vals)
    naux_vals = np.array(naux_vals)

    n_vals = nao_vals
    #print(f"n values:{n_vals}")
    R_vals = naux_vals * np.array([required_ranks[n] for n in n_val])
    #print(f"R values:{R_vals}")
    log_n = np.log10(n_vals)
    log_R = np.log10(R_vals)
    print(f"n values:{log_n}")
    print(f"R values:{log_R}")

    return log_n, log_R


def linear_model(x, a, p):
    return a + p * x


# ============================================================
# LOAD BOTH SYSTEMS
# ============================================================
data = {}
for system in SYSTEMS:
    data[system] = load_data(system)

# ============================================================
# PLOT 1 & 2 (PER SYSTEM)
# ============================================================
    for system, (df_energy, rank_to_energy, rank_to_error) in data.items():
        if system == "water_clusters":
            molecule = "Water clusters"
            x_title = r"$n, (H_{2}O)_n$"
        else:
            molecule = "Carbon chains"
            x_title = r"$n, C_{n}H_{2n+2}$"
        # ---------- PLOT 1 ----------
        plt.figure(figsize=(8,6))
        xs_df = sorted(df_energy.keys())
        ys_df = [df_energy[n] for n in xs_df]
    
        plt.plot(xs_df, ys_df, linestyle=":", linewidth=4, label=f"DF-{method}")
    
        for i, (r, d) in enumerate(rank_to_energy.items()):
            if not d:
                continue
            xs = sorted(d.keys())
            ys = [d[n] for n in xs]
            plt.plot(xs, ys, marker="o", linewidth=2, label=rf"$R_{{vv}}={r}$")
    
        plt.xlabel(rf"{x_title}")
        plt.ylabel(r"Energy (Hartree)")
        plt.title(rf"{method} energy vs system size""\n"
        rf"{molecule}, {Basis}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results, f"{method}_energy_vs_size_{system}_{basis}_{scan}.png"), dpi=300)
        plt.show()
    
        # ---------- PLOT 2 ----------
        plt.figure(figsize=(8,6))

        # Collect all existing system sizes from the data
        all_xs = set()
        
        for r, d in rank_to_error.items():
            if not d:
                continue
            xs = sorted(d.keys())
            ys = [d[n] for n in xs]
            all_xs.update(xs)
            plt.plot(xs, ys, marker="o", linewidth=2, label=rf"$R_{{vv}}={r}$")
        
        # Use ONLY existing system sizes as x-ticks
        all_xs = sorted(all_xs)
        plt.xticks(all_xs)
        
        plt.xlabel(rf"{x_title}")
        plt.yticks([0, 0.1, 0.2, 0.3])
        plt.ylabel(r"Error mH/atom")
        plt.title(rf"{method} error vs system size""\n"
        rf"{molecule}, {Basis}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(results, f"{method}_error_vs_size_{system}_{basis}_{scan}.png"),
            dpi=300
        )
        plt.show()
            
# ============================================================
# PLOT 3: COMBINED RANK SCALING
# ============================================================
plt.figure(figsize=(8,6))

markers = {"o",  "s"}

for system in SYSTEMS:
    _, _, rank_to_error = data[system]
    #molecules = SYSTEMS[system]["molecules"]
    print(f"molecule:{system}")
    log_n, log_R = compute_rank_scaling(rank_to_error, system)
    popt, _ = curve_fit(linear_model, log_n, log_R)
    a, p = popt

    plt.scatter(
        log_n, log_R,
        s=80,
        label=None
    )

    plt.plot(
        log_n, a + p * log_n,
        linewidth=2,
        label= rf"{system.replace('_',' ')}""\n"
                 rf"$y = {int(10**a)} x^{p:.1f}$"
                 
    )

plt.xlabel(r"Orbital basis set")
plt.ylabel(r"CP rank")
plt.title(
    rf"{method}: Rank scaling at fixed error""\n"
    rf"${ERROR_TOL}$ mH/atom, {Basis}"
)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig( os.path.join( results, f"{method}_rank_scaling_{ERROR_TOL}mH_{basis}_{scan}.png" ), dpi=300 )
plt.show()

