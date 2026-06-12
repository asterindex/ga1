#!/bin/bash
# Run all 6 experiments: 3 methods x 2 datasets (data / data2)
# Parameters: pop=12, gen=15, seed=7, mode=full

set -e

POP=12
GENS=15
SEED=13

run_exp() {
    local METHOD=$1
    local DATASET=$2
    local OUTDIR="${DATASET}_${METHOD}"
    echo ""
    echo "============================================================"
    echo "  METHOD=$METHOD  DATASET=$DATASET  -> $OUTDIR"
    echo "  Started: $(date)"
    echo "============================================================"
    python3.10 main.py \
        --mode full \
        --method "$METHOD" \
        --population "$POP" \
        --generations "$GENS" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --output "$OUTDIR"
    echo "DONE: $OUTDIR at $(date)"
}

echo "Starting 6-experiment run at $(date)"

# data (clean)
run_exp baseline  data
run_exp warmstart data
run_exp pareto    data

# data2 (occluded)
run_exp baseline  data2
run_exp warmstart data2
run_exp pareto    data2

echo ""
echo "============================================================"
echo "ALL 6 EXPERIMENTS DONE at $(date)"
echo "============================================================"
