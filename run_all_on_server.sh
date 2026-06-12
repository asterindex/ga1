#!/bin/bash
# Run all experiments on server automatically

set -e

SERVER_USER="anatoly_kot"
SERVER_HOST="193.200.64.60"

echo "🧬 Running all experiments on datacenter"
echo ""

ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
cd ~/genetic_nas

# Create experiments directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_DIR="experiments_$TIMESTAMP"
mkdir -p $EXPERIMENTS_DIR

echo "📁 Results will be saved to: $EXPERIMENTS_DIR"
echo ""

# Experiment 1: Fast mode, short Phase 1, medium Phase 2
echo "=========================================="
echo "🧪 EXPERIMENT 1: Fast + Short"
echo "=========================================="
docker run --rm --gpus all \
  -v $(pwd)/$EXPERIMENTS_DIR/exp1:/app/output \
  genetic-nas:latest \
  python3 main.py --mode fast --phase-1-generations 3 --phase-2-generations 30

echo "✅ Experiment 1 complete"
echo ""

# Experiment 2: Fast mode, longer Phase 2
echo "=========================================="
echo "🧪 EXPERIMENT 2: Fast + Medium"
echo "=========================================="
docker run --rm --gpus all \
  -v $(pwd)/$EXPERIMENTS_DIR/exp2:/app/output \
  genetic-nas:latest \
  python3 main.py --mode fast --phase-1-generations 5 --phase-2-generations 50

echo "✅ Experiment 2 complete"
echo ""

# Experiment 3: Full mode, standard
echo "=========================================="
echo "🧪 EXPERIMENT 3: Full + Standard"
echo "=========================================="
docker run --rm --gpus all \
  -v $(pwd)/$EXPERIMENTS_DIR/exp3:/app/output \
  genetic-nas:latest \
  python3 main.py --mode full --phase-1-generations 10 --phase-2-generations 100

echo "✅ Experiment 3 complete"
echo ""

# Experiment 4: Full mode, extensive Phase 2
echo "=========================================="
echo "🧪 EXPERIMENT 4: Full + Extensive"
echo "=========================================="
docker run --rm --gpus all \
  -v $(pwd)/$EXPERIMENTS_DIR/exp4:/app/output \
  genetic-nas:latest \
  python3 main.py --mode full --phase-1-generations 15 --phase-2-generations 200

echo "✅ Experiment 4 complete"
echo ""

# Create summary
echo "=========================================="
echo "📊 CREATING SUMMARY"
echo "=========================================="

cat > $EXPERIMENTS_DIR/SUMMARY.md << 'EOF'
# Experiments Summary

## Configuration
- Timestamp: $TIMESTAMP
- Mode: All experiments
- Total: 4 experiments

## Experiments

### Experiment 1: Fast + Short
- Mode: fast
- Phase 1: 3 generations
- Phase 2: 30 generations
- Expected time: ~5 min

### Experiment 2: Fast + Medium
- Mode: fast
- Phase 1: 5 generations
- Phase 2: 50 generations
- Expected time: ~10 min

### Experiment 3: Full + Standard
- Mode: full
- Phase 1: 10 generations
- Phase 2: 100 generations
- Expected time: ~30 min

### Experiment 4: Full + Extensive
- Mode: full
- Phase 1: 15 generations
- Phase 2: 200 generations
- Expected time: ~60 min

## Results Location
Each experiment has its own directory:
- exp1/ - Experiment 1 results
- exp2/ - Experiment 2 results
- exp3/ - Experiment 3 results
- exp4/ - Experiment 4 results

## Key Files
- coevolution/phase1_history.json - Phase 1 evolution
- coevolution/phase2_history.json - Phase 2 evolution
- coevolution/all_critics.json - Critic history
- best_model.json - Best architecture
- evolution.png - Evolution visualization
EOF

echo ""
echo "=========================================="
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Results saved to: $EXPERIMENTS_DIR"
echo ""
echo "Download results:"
echo "  scp -r $SERVER_USER@$SERVER_HOST:~/genetic_nas/$EXPERIMENTS_DIR ./results_$(date +%Y%m%d)"

ENDSSH

echo ""
echo "🎉 All experiments finished on server!"
