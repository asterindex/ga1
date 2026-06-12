#!/bin/bash
# Three-Method NAS Comparison (v3)
# Runs all three experiments sequentially on the same dataset.
#
# Experiments:
#   1. Baseline  - plain GA, no warm start
#   2. Warm Start - GA with Lamarckian warm start
#   3. Pareto    - GA with NSGA-II multi-objective selection
#
# Parameters (identical for all three):
#   population=12, generations=15, seed=7
#
# Usage: bash run_all_experiments_v3.sh [--rebuild]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

POP=12
GENS=15
SEED=7
IMAGE="genetic-nas:latest"
REBUILD=false

for arg in "$@"; do
    case $arg in
        --rebuild) REBUILD=true ;;
    esac
done

# -----------------------------------------------------------------------
echo "============================================================"
echo " Three-Method NAS Comparison (v3)"
echo " Population: $POP | Generations: $GENS | Seed: $SEED"
echo " Image: $IMAGE"
echo "============================================================"

# Stop any leftover containers
docker rm -f nas-baseline nas-warmstart nas-pareto 2>/dev/null || true

# Build image
if $REBUILD || ! docker image inspect $IMAGE &>/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Building Docker image..."
    docker build -t $IMAGE .
else
    echo "[$(date '+%H:%M:%S')] Using existing image $IMAGE"
fi

# -----------------------------------------------------------------------
run_experiment() {
    local METHOD=$1
    local OUT_DIR="output_${METHOD}"

    echo ""
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] Starting: $METHOD"
    echo "============================================================"

    mkdir -p "$OUT_DIR"

    docker run --rm --gpus all \
        -v "$SCRIPT_DIR/data:/app/data:ro" \
        -v "$SCRIPT_DIR/$OUT_DIR:/app/output" \
        --name "nas-${METHOD}" \
        $IMAGE \
        python3.10 main.py \
            --mode full \
            --method "$METHOD" \
            --generations $GENS \
            --population $POP \
            --seed $SEED \
        2>&1 | tee "$OUT_DIR/run.log"

    echo "[$(date '+%H:%M:%S')] $METHOD DONE"

    # results stay in output_${METHOD}/model_history/all_models.json
    # analyze_experiments.py reads from there directly
    echo "  Results in: $OUT_DIR/model_history/all_models.json"
}

# -----------------------------------------------------------------------
run_experiment "baseline"
run_experiment "warmstart"
run_experiment "pareto"

# -----------------------------------------------------------------------
echo ""
echo "============================================================"
echo " ALL EXPERIMENTS DONE"
echo " Results in per-method directories:"
echo "   output_baseline/model_history/all_models.json"
echo "   output_warmstart/model_history/all_models.json"
echo "   output_pareto/model_history/all_models.json"
echo ""
echo " Run analysis:"
echo "   python analyze_experiments.py"
echo "============================================================"
