"""
Utility functions for visualization and saving results
"""

import json
import matplotlib.pyplot as plt
from typing import List
from chromosome import Chromosome
from population import Population


def save_chromosome(chromosome: Chromosome, filename: str):
    """Save chromosome to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chromosome.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Chromosome saved to {filename}")


def plot_evolution_history(population: Population, filename: str = 'evolution.png'):
    """Visualize evolution history"""
    plt.figure(figsize=(12, 6))
    
    generations = range(len(population.best_fitness_history))
    
    plt.plot(generations, population.best_fitness_history, 'b-', label='Best accuracy', linewidth=2)
    plt.plot(generations, population.avg_fitness_history, 'r--', label='Average accuracy', linewidth=2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Population Evolution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")
    plt.close()


def print_population_diversity(population: Population):
    """Print information about population diversity"""
    optimizers = {}
    layer_counts = {}
    
    for ind in population.individuals:
        # Count optimizers
        opt = ind.optimizer
        optimizers[opt] = optimizers.get(opt, 0) + 1
        
        # Count number of layers
        num_layers = len(ind.layers)
        layer_counts[num_layers] = layer_counts.get(num_layers, 0) + 1
    
    print("\n=== Population Diversity ===")
    print("\nOptimizers:")
    for opt, count in sorted(optimizers.items()):
        print(f"  {opt}: {count}")
    
    print("\nNumber of layers:")
    for num, count in sorted(layer_counts.items()):
        print(f"  {num} layers: {count}")
    print("=" * 35 + "\n")
