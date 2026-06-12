# Quick Reference Guide

## Command Reference

### Basic Usage

```bash
# Display help
python3 main.py
python3 main.py --help

# Fast mode (30-40 minutes)
python3 main.py --mode fast

# Full mode (8-12 hours)
python3 main.py --mode full

# With Core ML (Mac only)
python3 main.py --mode full --coreml

# Resume from checkpoint
python3 main.py --mode full --resume
```

### Analysis Commands

```bash
# Analyze results
python3 analyze_models.py

# Validate models
python3 validate_models.py

# Validate specific generation
python3 validate_models.py --generation 5

# Validate limited number of models
python3 validate_models.py --max-models 10
```

### Monitoring

```bash
# View last 50 lines of log
tail -50 output/evolution_*.log

# Follow log in real-time
tail -f output/evolution_*.log

# Check model history
cat output/model_history/all_models.json | python3 -m json.tool | head -50
```

## Mode Comparison

| Mode | Population | Epochs | Generations | Mac M1 | GPU | Use Case |
|------|-----------|--------|-------------|--------|-----|----------|
| Fast | 4         | 5      | 5           | 30-40m | 20-30m | Quick testing |
| Full | 8         | 100*   | 15          | 8-12h  | 6-8h | Full experiments |

*With EarlyStopping (patience=3)

## Output Files

After running an experiment:

```
output/
├── evolution_full_20260115_140554.log  # Timestamped log
├── evolution.png                        # Visualization
├── best_model.json                      # Best architecture
└── model_history/
    └── all_models.json                  # All 120 models (full mode)
```

## Common Workflows

### 1. Quick Test

```bash
# Run fast mode
python3 main.py --mode fast

# Check results
cat output/best_model.json
```

### 2. Full Experiment

```bash
# Run full mode (leave overnight)
python3 main.py --mode full --coreml

# Analyze results
python3 analyze_models.py

# Validate models
python3 validate_models.py --max-models 20
```

### 3. Resume After Interruption

```bash
# Resume from checkpoint
python3 main.py --mode full --resume

# Check which generation it resumed from
grep "Resuming" output/evolution_*.log
```

### 4. Post-Experiment Analysis

```bash
# Generate analysis report
python3 analyze_models.py

# View report
cat output/EXPERIMENT_REPORT.md

# Validate reproducibility
python3 validate_models.py
```

## Configuration Parameters

Key settings in `src/config.py`:

```python
# Population
POPULATION_SIZE_FAST = 4
POPULATION_SIZE_FULL = 8

# Generations
NUM_GENERATIONS_FAST = 5
NUM_GENERATIONS_FULL = 15

# Training
TRAINING_EPOCHS_FAST = 5
TRAINING_EPOCHS_FULL = 100  # With EarlyStopping

# Genetic operators
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.4
TOURNAMENT_SIZE = 2
```

## Platform-Specific Notes

### Mac (Apple Silicon)

```bash
# With Core ML (recommended)
python3 main.py --mode fast --coreml

# Without Core ML (also works)
python3 main.py --mode fast

# Prevent sleep during long runs
caffeinate -i python3 main.py --mode full --coreml
```

### Linux/GPU

```bash
# Check GPU availability
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run experiment (no --coreml flag)
python3 main.py --mode full
```

## Troubleshooting

### Out of Memory

```bash
# Reduce population size
# Edit src/config.py:
POPULATION_SIZE_FULL = 4  # Instead of 8
```

### Slow Training

```bash
# Use Core ML on Mac
python3 main.py --mode fast --coreml

# Reduce epochs for fast testing
# Edit src/config.py:
TRAINING_EPOCHS_FAST = 3  # Instead of 5
```

### Checkpoint Issues

```bash
# Clean checkpoints and restart
rm -rf output/
python3 main.py --mode fast
```

## File Management

### Clean Output Directory

```bash
# Remove all outputs (automatic on new run without --resume)
rm -rf output/

# Keep model history, remove logs
rm output/*.log output/*.png
```

### Backup Results

```bash
# Create timestamped backup
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/run_$timestamp
cp -r output/* backups/run_$timestamp/
```

## Analysis Examples

### Find Best Model

```bash
# View best model details
python3 -c "
import json
with open('output/model_history/all_models.json') as f:
    data = json.load(f)
    models = []
    for gen in data['generations']:
        models.extend(gen['models'])
    best = max(models, key=lambda x: x['fitness'] or 0)
    print(f\"Best: {best['model_id']} - {best['fitness']:.4f}\")
    print(f\"Optimizer: {best['hyperparameters']['optimizer']}\")
    print(f\"LR: {best['hyperparameters']['learning_rate']}\")
"
```

### Count Models by Optimizer

```bash
# Optimizer distribution
python3 -c "
import json
from collections import Counter
with open('output/model_history/all_models.json') as f:
    data = json.load(f)
    models = []
    for gen in data['generations']:
        models.extend(gen['models'])
    opts = [m['hyperparameters']['optimizer'] for m in models]
    print(Counter(opts))
"
```

## Time Estimates

### Mac M1/M2/M3 (with Core ML)

- Fast mode: 30-40 minutes
- Full mode: 8-12 hours
- Per model (fast): ~6-8 minutes
- Per model (full): ~20-40 minutes (varies with EarlyStopping)

### GPU (NVIDIA T4/V100)

- Fast mode: 20-30 minutes
- Full mode: 6-8 hours
- Per model (fast): ~4-6 minutes
- Per model (full): ~15-30 minutes

### CPU Only

- Fast mode: 2-3 hours
- Full mode: Not recommended (>48 hours)

## Best Practices

1. **Start with Fast mode** to verify setup
2. **Use Full mode** for actual experiments
3. **Enable Core ML** on Mac for better performance
4. **Monitor logs** during long runs
5. **Backup results** after completion
6. **Validate models** to verify reproducibility
7. **Analyze results** using provided tools

## Getting Help

```bash
# Display main help
python3 main.py --help

# Display analysis help
python3 analyze_models.py --help

# Display validation help
python3 validate_models.py --help
```

## Additional Resources

- [README.md](README.md) - Full documentation
- [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) - Detailed analysis guide
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) - Model validation guide
