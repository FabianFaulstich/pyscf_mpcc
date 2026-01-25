from pathlib import Path
from pyscf import gto
import io
import sys
import re


def parse_iteration_energies(solver_output,pattern_str):
    """
    Extracts CC2 correlation energy values from MPCC solver output.
    
    Returns:
        iter_energies (list of float): energy at each iteration
        num_iterations (int): number of iterations
    """
    energy_pattern = re.compile(pattern_str)
    iter_energies = []

    for line in solver_output.splitlines():
        m = energy_pattern.search(line)
        if m:
            iter_energies.append(float(m.group(1)))

    return iter_energies, len(iter_energies)

def capture_output(func, *args, **kwargs):
    """
    Runs a function while capturing its printed output.
    Returns:
        output (str): full printed output
        result: return value of the function
    """
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()
    return output, result


def read_xyz(filepath: Path):
    """Read a standard XYZ file and return atom list for PySCF."""
    atoms = []
    with filepath.open() as f:
        lines = f.readlines()[2:]  # Skip natom + comment
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                atom = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append([atom, (x, y, z)])
    return atoms


def build_molecule(species, basis):
    """
    species can be:
    - name:  'h2o', 'water'
    - path to xyz file
    - path to directory containing multiple xyz files
    """

    species = str(species)
    path = Path(species)

    # ---------------------------------------------------
    # CASE 1: If path exists → file or folder
    # ---------------------------------------------------
    if path.exists():
        # ---- xyz file ----
        if path.is_file() and path.suffix.lower() == ".xyz":
            atoms = read_xyz(path)
            mol = gto.Mole()
            mol.atom = atoms
            mol.basis = basis
            mol.build()
            return mol, path.stem

        # ---- directory with many xyz files ----
        if path.is_dir():
            xyz_files = sorted(path.glob("*.xyz"))
            if not xyz_files:
                raise FileNotFoundError(f"No XYZ files in folder {path}")
            return xyz_files  # returned for looping later

    # ---------------------------------------------------
    # CASE 2: Predefined small molecules
    # ---------------------------------------------------
    name = species.lower()

    if name in ["h2o","water"]:
        mol = gto.Mole()
        mol.atom = [
            [8, (0.0, 0.0, 0.0)],
            [1, (0.0, -0.757, 0.587)],
            [1, (0.0, 0.757, 0.587)],
        ]
        mol.basis = basis
        mol.build()
        return mol, "TIP4P-1"
    
    if name in ["ch4", "methane"]:
        mol = gto.Mole()
        mol.atom = [
            [6, (-0.6695, 0.0, 0.0)],
            [6, (0.6695, 0.0, 0.0)],
            [1, (-1.2335, 0.9238, 0.0)],
            [1, (-1.2335, -0.9238, 0.0)],
            [1, (1.2335, 0.9238, 0.0)],
            [1, (1.2335, -0.9238, 0.0)],
        ]
        mol.basis = basis
        mol.build()
        return mol, "ch4"
    if name in ["c3h8", "propane"]:
        mol = gto.Mole()
        mol.atom = [
            [6	,(0	        ,0	    ,0.5891	)],
            [6	,(0	        ,1.2676	,-0.2605)],
            [6	,(0	        ,-1.2676,-0.2605)],
            [1	,(0.8749    ,0	    ,1.243	)],
            [1	,(-0.8749   ,0	    ,1.243	)],
            [1	,(0	        ,2.1642	,0.3602	)],
            [1	,(0	        ,-2.1642,0.3602	)],
            [1	,(0.8811	,1.3045	,-0.9037)],
            [1	,(-0.8811	,1.3045	,-0.9037)],
            [1	,(-0.8811	,-1.3045,-0.9037)],
            [1	,(0.8811	,-1.3045,-0.9037)]
    ]

        mol.basis = basis
        mol.build()
        return mol, "c3h8"


    # ---------------------------------------------------
    raise ValueError(f"Unknown species or invalid path: {species}")

