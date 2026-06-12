#!/bin/bash
# Automatic experiments runner for server

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments_${TIMESTAMP}"

mkdir -p $RESULTS_DIR

echo "🧬 GENETIC NAS - AUTOMATED EXPERIMENTS"
echo "======================================"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Experiment 1: Fast baseline
echo "📊 Experiment 1/4: Fast Baseline (Phase 1 only)"
docker-compose run --rm genetic-nas \
  python3 main.py --mode fast --phase-1-generations 5 --phase-2-generations 0
cp -r output "$RESULTS_DIR/exp1_fast_baseline_p1_5"
rm -rf output/*

# Experiment 2: Fast two-phase  
echo "📊 Experiment 2/4: Fast Two-Phase"
docker-compose run --rm genetic-nas \
  python3 main.py --mode fast --phase-1-generations 5 --phase-2-generations 50
cp -r output "$RESULTS_DIR/exp2_fast_twophase_p1_5_p2_50"
rm -rf output/*

# Experiment 3: Medium two-phase
echo "📊 Experiment 3/4: Medium Two-Phase"
docker-compose run --rm genetic-nas \
  python3 main.py --mode fast --phase-1-generations 10 --phase-2-generations 100
cp -r output "$RESULTS_DIR/exp3_medium_twophase_p1_10_p2_100"
rm -rf output/*

# Experiment 4: Full experiment
echo "📊 Experiment 4/4: Full Two-Phase (for paper)"
docker-compose run --rm genetic-nas \
  python3 main.py --mode full --phase-1-generations 15 --phase-2-generations 200
cp -r output "$RESULTS_DIR/exp4_full_twophase_p1_15_p2_200"

echo ""
echo "✅ All experiments complete!"
echo "📁 Results: $RESULTS_DIR"
echo ""
echo "📊 Summary:"
find $RESULTS_DIR -name "best_model.json" -exec echo "  {}" \; -exec python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'    Fitness: {d[\"fitness\"]:.4f}')" {} \;
