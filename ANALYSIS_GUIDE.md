# Analysis Guide

## Overview

After completing an evolutionary run, the `output/model_history/all_models.json` file contains complete information about all evaluated models. This guide explains how to analyze these results to extract insights about the evolutionary process and architectural patterns.

## Automated Analysis

### Running the Analysis Script

```bash
python3 analyze_models.py
```

This generates:
- Console output with key statistics
- `output/EXPERIMENT_REPORT.md` - Detailed analysis report
- Visualization plots (if matplotlib available)

### Analysis Components

The script performs the following analyses:

#### 1. Hyperparameter Importance

Evaluates the impact of each hyperparameter on model performance:

```
Optimizer Analysis:
  adam:
    - Count: 45 models
    - Average fitness: 0.6523
    - Max fitness: 0.6795
    - Std deviation: 0.0234
  
  sgd:
    - Count: 38 models
    - Average fitness: 0.6234
    - Max fitness: 0.6512
    - Std deviation: 0.0312
  
  rmsprop:
    - Count: 37 models
    - Average fitness: 0.6445
    - Max fitness: 0.6689
    - Std deviation: 0.0267
```

#### 2. Learning Rate Analysis

Identifies optimal learning rate ranges:

```
Learning Rate Distribution:
  0.0001-0.0003: 23 models, avg fitness: 0.6234
  0.0003-0.0005: 34 models, avg fitness: 0.6523
  0.0005-0.0010: 28 models, avg fitness: 0.6312
  0.0010+:       35 models, avg fitness: 0.5891
```

#### 3. Batch Size Impact

Analyzes relationship between batch size and accuracy:

```
Batch Size Analysis:
  16:  28 models, avg fitness: 0.6423
  32:  45 models, avg fitness: 0.6534
  64:  47 models, avg fitness: 0.6312
```

#### 4. Architecture Depth

Examines the effect of network depth:

```
Layer Count Distribution:
  5 layers:  12 models, avg fitness: 0.5923
  6 layers:  18 models, avg fitness: 0.6234
  7 layers:  23 models, avg fitness: 0.6445
  8 layers:  28 models, avg fitness: 0.6523
  9 layers:  21 models, avg fitness: 0.6534
  10 layers: 18 models, avg fitness: 0.6312
```

#### 5. Layer Type Frequency

Counts layer types in top-performing models:

```
Top 20% Models - Layer Type Frequency:
  conv2d:      48 total (2.4 per model)
  batch_norm:  34 total (1.7 per model)
  max_pool:    28 total (1.4 per model)
  dense:       24 total (1.2 per model)
  dropout:     18 total (0.9 per model)
```

#### 6. Best Model Details

Provides complete information about the best discovered architecture:

```
Best Model: gen14_model3
  Fitness: 0.6852
  
  Architecture:
    1. Conv2D(64, 3×3, relu)
    2. BatchNorm()
    3. Conv2D(32, 3×3, elu)
    4. BatchNorm()
    5. MaxPool(2×2)
    6. Flatten()
    7. Dense(32, relu)
    8. Dropout(0.21)
    9. Dense(10, softmax)
  
  Hyperparameters:
    - Optimizer: rmsprop
    - Learning rate: 0.00021
    - Batch size: 64
```

## Manual Analysis

### Loading Model History

```python
import json

# Load history
with open('output/model_history/all_models.json', 'r') as f:
    history = json.load(f)

# Extract all models
all_models = []
for generation in history['generations']:
    all_models.extend(generation['models'])

print(f"Total models: {len(all_models)}")
```

### Finding Top Models

```python
# Sort by fitness
top_models = sorted(
    all_models,
    key=lambda x: x['fitness'] if x['fitness'] else 0,
    reverse=True
)[:10]

# Display top 10
for i, model in enumerate(top_models, 1):
    print(f"{i}. {model['model_id']}: {model['fitness']:.4f}")
    print(f"   Optimizer: {model['hyperparameters']['optimizer']}")
    print(f"   LR: {model['hyperparameters']['learning_rate']:.6f}")
    print(f"   Batch: {model['hyperparameters']['batch_size']}")
    print(f"   Layers: {model['architecture']['num_layers']}")
    print()
```

### Comparing Optimizers

```python
from collections import defaultdict
import numpy as np

# Group by optimizer
optimizer_fitness = defaultdict(list)
for model in all_models:
    if model['fitness']:
        opt = model['hyperparameters']['optimizer']
        optimizer_fitness[opt].append(model['fitness'])

# Calculate statistics
for opt, fitness_values in optimizer_fitness.items():
    print(f"{opt}:")
    print(f"  Count: {len(fitness_values)}")
    print(f"  Mean: {np.mean(fitness_values):.4f}")
    print(f"  Std: {np.std(fitness_values):.4f}")
    print(f"  Max: {np.max(fitness_values):.4f}")
    print()
```

### Evolution Trajectory

```python
# Track best fitness per generation
generation_best = []
for gen in history['generations']:
    gen_fitness = [m['fitness'] for m in gen['models'] if m['fitness']]
    if gen_fitness:
        generation_best.append(max(gen_fitness))

# Display progression
for i, fitness in enumerate(generation_best):
    print(f"Generation {i}: {fitness:.4f}")
```

## Visualization

### Fitness Distribution by Optimizer

```python
import matplotlib.pyplot as plt

# Prepare data
optimizers = ['adam', 'sgd', 'rmsprop']
fitness_by_opt = {opt: [] for opt in optimizers}

for model in all_models:
    if model['fitness']:
        opt = model['hyperparameters']['optimizer']
        fitness_by_opt[opt].append(model['fitness'])

# Create boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(
    [fitness_by_opt[opt] for opt in optimizers],
    labels=optimizers
)
plt.ylabel('Fitness (Accuracy)')
plt.title('Optimizer Performance Distribution')
plt.grid(True, alpha=0.3)
plt.savefig('output/optimizer_comparison.png', dpi=300)
plt.show()
```

### Evolution Progress

```python
# Plot best, average, worst per generation
best_per_gen = []
avg_per_gen = []
worst_per_gen = []

for gen in history['generations']:
    fitness_values = [m['fitness'] for m in gen['models'] if m['fitness']]
    if fitness_values:
        best_per_gen.append(max(fitness_values))
        avg_per_gen.append(sum(fitness_values) / len(fitness_values))
        worst_per_gen.append(min(fitness_values))

plt.figure(figsize=(12, 6))
generations = range(len(best_per_gen))
plt.plot(generations, best_per_gen, 'g-', label='Best', linewidth=2)
plt.plot(generations, avg_per_gen, 'b-', label='Average', linewidth=2)
plt.plot(generations, worst_per_gen, 'r-', label='Worst', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Evolution Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/evolution_progress.png', dpi=300)
plt.show()
```

### Learning Rate vs Fitness

```python
# Extract learning rates and fitness
lr_values = []
fitness_values = []

for model in all_models:
    if model['fitness']:
        lr_values.append(model['hyperparameters']['learning_rate'])
        fitness_values.append(model['fitness'])

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(lr_values, fitness_values, alpha=0.5)
plt.xlabel('Learning Rate')
plt.ylabel('Fitness')
plt.title('Learning Rate vs Fitness')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('output/lr_vs_fitness.png', dpi=300)
plt.show()
```

## Ablation Study

An ablation study removes components to measure their individual contribution.

### Methodology

1. Identify the best model
2. Remove one component (e.g., BatchNorm layers)
3. Retrain and evaluate
4. Compare accuracy difference

### Example

```python
# Load best model architecture
best_model = max(all_models, key=lambda x: x['fitness'] or 0)

# Count BatchNorm layers
batch_norm_count = sum(
    1 for layer in best_model['architecture']['layers']
    if layer['layer_type'] == 'batch_norm'
)

print(f"Best model has {batch_norm_count} BatchNorm layers")
print(f"Fitness: {best_model['fitness']:.4f}")

# After removing BatchNorm and retraining:
# Original: 0.6852
# Without BatchNorm: 0.6375
# Contribution: +0.0477 (4.77%)
```

### Results Interpretation

- **Large difference (>5%)**: Component is critical
- **Moderate difference (2-5%)**: Component is beneficial
- **Small difference (<2%)**: Component has minimal impact

## Statistical Analysis

### Correlation Analysis

```python
import pandas as pd
import seaborn as sns

# Create dataframe
data = []
for model in all_models:
    if model['fitness']:
        data.append({
            'fitness': model['fitness'],
            'lr': model['hyperparameters']['learning_rate'],
            'batch_size': model['hyperparameters']['batch_size'],
            'num_layers': model['architecture']['num_layers']
        })

df = pd.DataFrame(data)

# Correlation matrix
correlation = df.corr()
print(correlation)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Parameter Correlation Matrix')
plt.savefig('output/correlation_matrix.png', dpi=300)
plt.show()
```

### Statistical Significance

```python
from scipy import stats

# Compare two optimizers
adam_fitness = [m['fitness'] for m in all_models 
                if m['fitness'] and m['hyperparameters']['optimizer'] == 'adam']
sgd_fitness = [m['fitness'] for m in all_models 
               if m['fitness'] and m['hyperparameters']['optimizer'] == 'sgd']

# T-test
t_stat, p_value = stats.ttest_ind(adam_fitness, sgd_fitness)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
else:
    print("Difference is not statistically significant")
```

## Research Applications

### For Academic Papers

Use the analysis results in the following sections:

#### 4.1 Experimental Setup
- Population size, generations, epochs
- Hyperparameter ranges
- Hardware specifications

#### 4.2 Results
- Best accuracy achieved
- Evolution trajectory plot
- Comparison with baseline methods

#### 4.3 Analysis
- Hyperparameter importance
- Architecture depth analysis
- Layer type frequency in successful models

#### 4.4 Ablation Study
- Component contribution table
- Statistical significance tests

#### 4.5 Discussion
- Why certain patterns emerged
- Relationship to theoretical expectations
- Limitations and future work

### Example Results Table

| Hyperparameter | Best Value | Avg. Fitness | Std. Dev. |
|----------------|-----------|--------------|-----------|
| Optimizer | RMSprop | 0.6445 | 0.0267 |
| Learning Rate | 0.0003-0.0005 | 0.6523 | 0.0234 |
| Batch Size | 32 | 0.6534 | 0.0198 |
| Network Depth | 8-9 layers | 0.6529 | 0.0245 |

### Example Ablation Table

| Component | Accuracy | Δ Accuracy | p-value |
|-----------|----------|-----------|---------|
| Full Model | 68.52% | - | - |
| - BatchNorm | 63.75% | -4.77% | <0.001 |
| - Dropout | 66.23% | -2.29% | <0.05 |
| - MaxPool | 64.12% | -4.40% | <0.001 |

## Advanced Analysis

### Diversity Metrics

```python
# Calculate population diversity
def hamming_distance(model1, model2):
    # Compare architectures
    layers1 = [l['layer_type'] for l in model1['architecture']['layers']]
    layers2 = [l['layer_type'] for l in model2['architecture']['layers']]
    
    # Pad to same length
    max_len = max(len(layers1), len(layers2))
    layers1 += ['none'] * (max_len - len(layers1))
    layers2 += ['none'] * (max_len - len(layers2))
    
    # Count differences
    return sum(l1 != l2 for l1, l2 in zip(layers1, layers2))

# Calculate average diversity per generation
for gen_idx, gen in enumerate(history['generations']):
    models = gen['models']
    distances = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            distances.append(hamming_distance(models[i], models[j]))
    
    avg_diversity = sum(distances) / len(distances) if distances else 0
    print(f"Generation {gen_idx}: diversity = {avg_diversity:.2f}")
```

### Convergence Analysis

```python
# Measure fitness variance over generations
variances = []
for gen in history['generations']:
    fitness_values = [m['fitness'] for m in gen['models'] if m['fitness']]
    if len(fitness_values) > 1:
        variance = np.var(fitness_values)
        variances.append(variance)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(variances)
plt.xlabel('Generation')
plt.ylabel('Fitness Variance')
plt.title('Population Convergence')
plt.grid(True, alpha=0.3)
plt.savefig('output/convergence.png', dpi=300)
plt.show()
```

## Conclusion

The analysis tools provided enable comprehensive understanding of:
- Which hyperparameters matter most
- Optimal architecture patterns
- Evolution dynamics
- Component contributions

These insights are valuable for:
- Improving the search algorithm
- Understanding architectural principles
- Publishing research results
- Guiding future experiments
