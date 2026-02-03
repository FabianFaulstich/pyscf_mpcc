#!/bin/bash

# ===========================================
# USAGE:
#   ./run_all_molecules.sh aug-cc-pvtz Lov_Lvv
# ===========================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <basis> {Lvv|Lov|Lov_Lvv|Lov_fix_Loo_1|Lvv_fix_Loo_1|Lov_Lvv_fix_Loo_1}"
    exit 1
fi

basis=$1
scan_mode=$2

molecules=(
    	Molecule/c10h22.xyz
)

ranks=(1.5 2 2.5 3 3.5)

results_dir="$HOME/Documents/Nov_11_MPCC/All_output_data"

echo "=========================================="
echo " STARTING OVERNIGHT MPCC RUN "
echo "=========================================="

for species in "${molecules[@]}"; do
    echo ""
    echo "=========================================="
    echo " Molecule: $species"
    echo "=========================================="

    echo "---- DF run ----"
    python 04-single-shot-mpcc.py \
        "$species" \
        "$basis" \
        false \
        --results "$results_dir" \
        --scan "$scan_mode"

    echo "---- CPD runs ----"
    for r in "${ranks[@]}"; do
        echo "   CPD rank = $r"
        python 04-single-shot-mpcc.py \
            "$species" \
            "$basis" \
            true \
            --rank "$r" \
            --results "$results_dir" \
            --scan "$scan_mode"
    done

    echo " Completed $species"
done

echo "=========================================="
echo " ALL MOLECULES COMPLETED "
echo "=========================================="

