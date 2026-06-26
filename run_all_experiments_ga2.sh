#!/bin/bash
# Run 4 ga2 experiments on server: baseline, warmstart, pareto, hardware
# Dataset: data | pop=12, gen=15, seed=13, mode=full

set -e

POP=12
GENS=15
SEED=13
DATASET=data
IMAGE="ga2:latest"

run_exp() {
    local METHOD=$1
    local OUTDIR="output_${METHOD}"
    echo ""
    echo "============================================================"
    echo "  METHOD=$METHOD  DATASET=$DATASET  -> $OUTDIR"
    echo "  Started: $(date)"
    echo "============================================================"
    mkdir -p "$OUTDIR"
    docker run --rm --gpus all \
        -v "$(pwd)/$OUTDIR:/app/output" \
        -v "$(pwd)/$DATASET:/app/$DATASET" \
        "$IMAGE" \
        python3 main.py \
            --mode full \
            --method "$METHOD" \
            --population "$POP" \
            --generations "$GENS" \
            --seed "$SEED" \
            --dataset "$DATASET" \
            --output /app/output
    echo "DONE: $OUTDIR at $(date)"
}

echo "Starting ga2 4-experiment run at $(date)"

run_exp baseline
run_exp warmstart
run_exp pareto
run_exp hardware

echo ""
echo "============================================================"
echo "ALL 4 EXPERIMENTS DONE at $(date)"
echo "============================================================"
