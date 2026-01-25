import re
import numpy as np

# Store omega values
omega = {
    "DF": None,
    "CPD_1.5": None,
    "CPD_2": None,
}

current_run = None
current_omega = None

with open("output.txt", "r") as f:
    for line in f:

        # ---------------- Detect which run we are in ----------------
        if "Running DF (rank_reduced = false)" in line:
            current_run = "DF"

        elif "CPD Run: Lov=Lvv = 1.5" in line:
            current_run = "CPD_1.5"

        elif "CPD Run: Lov=Lvv = 2" in line:
            current_run = "CPD_2"

        # ---------------- Capture omega values ----------------
        if "omega value:" in line:
            current_omega = float(line.split(":")[1].strip())

        # ---------------- Capture macro iteration ----------------
        if line.startswith("DFMPCC It:"):
            m = re.search(r"DFMPCC It:\s*(\d+)", line)
            if m and current_run is not None and current_omega is not None:
                macro_it = int(m.group(1))

                # 👇 ONLY macro iteration 2
                if macro_it == 2 and omega[current_run] is None:
                    omega[current_run] = current_omega
print(omega)

