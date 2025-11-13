import re
import matplotlib.pyplot as plt

# Lists for first macro iteration only
ccsd_corr_iter_macro1 = []
cc2_corr_iter_macro1 = []

# Track current macro iteration
current_macro = 0

with open("output.txt", "r") as f:
    for line in f:
        line = line.strip()
        
        # Update current macro iteration
        if line.startswith("MPCC macro iteration:"):
            current_macro = int(line.split(":")[-1].strip())
        
        # Only record data if we are in macro iteration 1
        if current_macro == 1:
            if "CCSD correlation energy:" in line:
                value = float(line.split(":")[-1])
                ccsd_corr_iter_macro1.append(value)
            
            if "CC2 correlation energy:" in line:
                value = float(line.split(":")[-1])
                cc2_corr_iter_macro1.append(value)

print("CCSD energies in macro iteration 1:", ccsd_corr_iter_macro1)
print("CC2 energies in macro iteration 1:", cc2_corr_iter_macro1)


# Example: energies from first macro iteration
# ccsd_corr_iter_macro1 and cc2_corr_iter_macro1 are lists from previous code

# Iteration numbers
iterations_ccsd = list(range(1, len(ccsd_corr_iter_macro1) + 1))
iterations_cc2 = list(range(1, len(cc2_corr_iter_macro1) + 1))

# Plot
plt.plot(iterations_ccsd, ccsd_corr_iter_macro1, marker='o', label="CCSD")
#plt.plot(iterations_cc2, cc2_corr_iter_macro1, marker='x', label="CC2")
plt.xlabel("Iteration")
plt.ylabel("Correlation Energy ")
plt.title("Energies vs No of iteration")
plt.legend()
plt.grid(True)
plt.show()

