# ga2 — Hardware-Aware Genetic NAS

Evolutionary Neural Architecture Search for Military Vehicles classification with **hardware-aware optimization** for onboard UAV compute platforms (Jetson / Coral proxy via TFLite on Linux GPU server).

## Overview

The genetic algorithm evolves CNN architecture and hyperparameters. ga2 adds **Hardware-Aware NSGA-II** (`--method hardware`) with 4 objectives:

| Objective | Direction | Source |
|-----------|-----------|--------|
| Validation accuracy | maximize | TensorFlow training |
| Inference latency | minimize | TFLite benchmark |
| Model size | minimize | TFLite file size |
| Peak RAM | minimize | tracemalloc during inference |

Comparison methods: `baseline`, `warmstart`, `pareto` (3 objectives: accuracy, params, training time).

**Testing:** server-only via Docker on Linux GPU (`193.200.64.103`, `~/ga2`).

## Key Features

- **Genetic Algorithm**: Population-based evolutionary search
- **Hardware-Aware NAS**: TFLite export + inference benchmark per model
- **NSGA-II**: 3-objective (`pareto`) or 4-objective (`hardware`) Pareto selection
- **GPU Ready**: Docker + CUDA on server
- **Full Tracking**: All models + hardware metrics saved to JSON

## Quick Start

### Local (Mac/Linux)

```bash
# Install dependencies
pip install -r requirements.txt

# Fast test
python3 main.py --mode fast --method baseline

# Hardware-aware NAS (4 objectives + TFLite benchmark)
python3 main.py --mode full --method hardware --dataset data

# Full experiment suite (4 methods)
./run_all_experiments_ga2.sh
```

### Server (Docker + GPU)

See **[SERVER_QUICKSTART.md](SERVER_QUICKSTART.md)** for detailed instructions.

```bash
# Build
docker build -t ga2:latest .

# Run
docker run --gpus all -v $(pwd)/output:/app/output ga2:latest
```

## Run in Docker (Data Center / Linux GPU)

This repo includes Docker support for running experiments on a Linux server (NVIDIA GPU).

### Dependencies split

- **Local Mac** (Apple Silicon): keep using `requirements.txt` (includes `tensorflow-macos`, `tensorflow-metal`, `coremltools`).
- **Linux / Data Center** (Docker): uses `requirements.base.txt` (regular `tensorflow` + scientific stack).

If you want a clean local env (optional):

```bash
pip install -r requirements.base.txt -r requirements.mac.txt
```

### Run (GPU)

From the project root on the server:

```bash
docker compose --profile gpu up --build --abort-on-container-exit trainer-gpu
```

### Run (CPU)

```bash
docker compose up --build --abort-on-container-exit trainer
```

### Change mode (fast/full)

Edit `docker-compose.yml` and change the command, for example:

- fast:
  - `["python", "main.py", "--mode", "fast"]`
- full:
  - `["python", "main.py", "--mode", "full"]`

### Outputs

`output/` is mounted as a volume, so logs/results persist outside the container.

## Modes Comparison

| Mode | Population | Epochs | Generations | Samples | Time (Mac M1) | Time (GPU) |
|------|-----------|--------|-------------|---------|---------------|------------|
| Fast | 4 models  | 5      | 3           | 10k     | ~10-15 min    | ~5-10 min  |
| Full | 8 models  | up to 100* | 15      | 50k     | ~8-12 hours   | ~6-8 hours |

*With EarlyStopping (patience=3, monitor='val_accuracy')

## Architecture

### Genetic Encoding

Each chromosome encodes:
- **Network structure**: Number and types of layers (Conv2D, Dense, Dropout, BatchNorm, MaxPooling)
- **Layer parameters**: Filter counts, kernel sizes, activation functions
- **Hyperparameters**: Learning rate, batch size, optimizer type

### Genetic Operators

- **Selection**: Tournament selection (tournament size = 2)
- **Crossover**: Uniform crossover (probability = 0.7)
- **Mutation**: Structural (add/remove layers) and parametric (probability = 0.4)
- **Elitism**: Best individual preserved across generations

### Fitness Function

Validation accuracy on Military Vehicles test set (496 images). The system records the maximum validation accuracy achieved during training, not the final epoch accuracy, to account for overfitting.

## Output Structure

```
ga2/
├── output/
│   ├── evolution_full_YYYYMMDD_HHMMSS.log  # Timestamped log file
│   ├── evolution.png                        # Evolution visualization
│   ├── best_model.json                      # Best discovered architecture
│   └── model_history/
│       └── all_models.json                  # Complete history of all models
│
├── main.py              # Main entry point
├── analyze_models.py    # Post-hoc analysis tool
├── validate_models.py   # Model validation tool
│
└── src/                 # Source code
    ├── chromosome.py
    ├── config.py
    ├── dataset_loader.py
    ├── evolution.py
    ├── fitness.py
    ├── coreml_fitness.py
    ├── history_loader.py
    ├── logger.py
    ├── model_history.py
    ├── operators.py
    ├── population.py
    └── utils.py
```

## Analysis Tools

### Analyze Results

```bash
python3 analyze_models.py
```

Generates:
- Hyperparameter importance analysis (optimizer, learning rate, batch size)
- Architecture depth analysis
- Layer type frequency in top models
- Evolution trajectory visualization
- Detailed report: `output/EXPERIMENT_REPORT.md`

### Validate Models

```bash
# Validate all models
python3 validate_models.py

# Validate first 10 models
python3 validate_models.py --max-models 10

# Validate specific generation
python3 validate_models.py --generation 5
```

Rebuilds models from JSON and verifies accuracy reproducibility.

## Dataset

**Military and Civilian Vehicles Classification**: 7,198 images in 6 classes
- Training set: 6,702 images (64×64 resized)
- Validation set: 496 images
- Classes: military tank, military aircraft, military helicopter, military truck, civilian car, civilian aircraft

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## System Requirements

### For Mac (Apple Silicon)

- macOS 12.0+
- Python 3.8+
- 8GB+ RAM
- Apple Silicon (M1/M2/M3) for Core ML support

### For Linux/GPU

- Python 3.8+
- TensorFlow 2.x with GPU support
- CUDA-compatible GPU (recommended)

## Configuration

Key parameters in `src/config.py`:

```python
# Population sizes
POPULATION_SIZE_FAST = 4
POPULATION_SIZE_FULL = 8

# Generations
NUM_GENERATIONS_FAST = 3
NUM_GENERATIONS_FULL = 15

# Training epochs
TRAINING_EPOCHS_FAST = 5
TRAINING_EPOCHS_FULL = 100  # With EarlyStopping

# Genetic operators
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.4
```

## Documentation

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference and common workflows
- [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) - Post-experiment analysis guide
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) - Model validation guide

## Research Applications

This project is designed for research in Neural Architecture Search. Key features for academic use:

- **Reproducibility**: All models saved with complete architectural details
- **Interpretability**: Post-hoc analysis tools for understanding evolved architectures
- **Validation**: Built-in model validation for verifying results
- **Extensibility**: Modular design for easy modification and extension

## Performance Notes

### Mac (Apple Silicon with Core ML)
- Fast mode: ~10-15 minutes (10k samples)
- Full mode: ~8-12 hours (50k samples)
- Requires `--coreml` flag

### GPU (CUDA)
- Fast mode: ~5-10 minutes (10k samples)
- Full mode: ~6-8 hours (50k samples)
- Automatic GPU detection

### CPU Only
- Not recommended for full experiments
- Fast mode only: ~30-45 minutes (10k samples)

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss proposed modifications.

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genetic_nas_2026,
  title={Genetic NAS: Neural Architecture Search with Genetic Algorithms},
  author={Your Name},
  year={2026},
  url={https://github.com/asterindex/genetic_nas}
}
```

## Acknowledgments

- Military and Civilian Vehicles Classification dataset from Mendeley Data
- TensorFlow team for the deep learning framework
- Apple for Core ML optimization support
