#!/bin/bash

# ===============================
#   Bash Script for CC2 Plots
# ===============================
# Usage:
#   ./run_all_plots.sh BASIS MOLECULE SCAN RESULTS_FOLDER
#
# Example:
#   ./run_all_plots.sh cc-pvdz TIP4P-4 Lvv /home/talha/output
#
# This will run plots for:
#   Y, Ω, Foo, Fov, Fvv
# ===============================

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 BASIS MOLECULE SCAN"
    exit 1
fi

BASIS="$1"
MOLECULE="$2"
SCAN="$3"

# All tensors to plot
TENSORS=("Y" "Ω" "Foo" "Fov" "Fvv")
results_dir="$HOME/Documents/RR_MPCC/All_output_data" 
# Python script name
SCRIPT="Y_amp_plot.py"      # <-- Change if your python file has a different name

echo "-------------------------------------------"
echo " Running CC2 plots"
echo " Molecule: $MOLECULE"
echo " Basis:    $BASIS"
echo " Scan:     $SCAN"
echo "-------------------------------------------"

for T in "${TENSORS[@]}"; do
    echo ">>> Plotting tensor: $T"
    python "$SCRIPT" "$BASIS" "$MOLECULE" --scan "$SCAN" --results "$results_dir" --relative --tensor "$T"
    echo ""
done

echo "-------------------------------------------"
echo " All plots finished successfully."
echo "-------------------------------------------"

