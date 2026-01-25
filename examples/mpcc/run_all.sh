#!/bin/bash

# ===========================================
# USAGE:
#   ./run_all.sh Molecule/c4h10.xyz aug-cc-pvtz Lov_Lvv
# ===========================================

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <species.xyz> <basis> {Lvv|Lov|Lov_Lvv|Lov_fix_Loo_1|Lvv_fix_Loo_1|Lov_Lvv_fix_Loo_1}"
    exit 1
fi

species=$1
basis=$2
scan_mode=$3   # required string

ranks=(1.5 2 2.5 3 3.5)
results_dir="$HOME/Documents/RR_MPCC/All_output_data"

echo "=========================================="
echo " Running DF (rank_reduced = false)"
echo "=========================================="

python 04-single-shot-mpcc.py \
    "$species" \
    "$basis" \
    false \
    --results "$results_dir" \
    --scan "$scan_mode"

echo ""
echo "=========================================="
echo " Running CPD (rank_reduced = true)"
echo "=========================================="

for r in "${ranks[@]}"; do
    echo "------ CPD Run: Lov=Lvv = $r ------"
    python 04-single-shot-mpcc.py \
        "$species" \
        "$basis" \
        true \
        --rank "$r" \
        --results "$results_dir" \
        --scan "$scan_mode"
done

echo "=========================================="
echo " ALL RUNS COMPLETED "
echo "=========================================="

