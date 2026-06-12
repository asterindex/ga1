"""
Load and restore population from model_history/all_models.json
"""

import json
import os
from typing import Optional, Dict, Any
from population import Population
from chromosome import Chromosome, Layer


def load_from_model_history(history_file: str = "output/model_history/all_models.json") -> Optional[Dict[str, Any]]:
    """
    Load checkpoint from model_history/all_models.json
    
    Args:
        history_file: Path to model history JSON file
        
    Returns:
        Dict with: population, generation, best_fitness_history, avg_fitness_history
        Or None if not found
    """
    if not os.path.exists(history_file):
        print(f"\n❌ Model history file not found: {history_file}")
        return None
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history['generations']:
            print(f"\n❌ No generations found in history file")
            return None
        
        # Get last generation
        last_gen = history['generations'][-1]
        generation = last_gen['generation']
        models = last_gen['models']
        mode = last_gen['mode']
        
        print(f"\n{'='*70}")
        print(f"📂 Loading from model history")
        print(f"{'='*70}")
        print(f"Generation: {generation}")
        print(f"Models: {len(models)}")
        print(f"Mode: {mode}")
        print(f"{'='*70}\n")
        
        # Convert models to chromosomes
        individuals = []
        for model_data in models:
            chromosome = _model_data_to_chromosome(model_data)
            individuals.append(chromosome)
        
        # Calculate fitness histories from all generations
        best_fitness_history = []
        avg_fitness_history = []
        
        for gen in history['generations']:
            fitnesses = [m['fitness'] for m in gen['models'] if m['fitness'] is not None]
            if fitnesses:
                best_fitness_history.append(max(fitnesses))
                avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        
        # Create population
        input_shape = (64, 64, 3)  # Military Vehicles
        output_dim = 6  # Military Vehicles: 6 класів
        
        population = Population(len(models), input_shape, output_dim)
        population.generation = generation
        population.individuals = individuals
        population.best_fitness_history = best_fitness_history
        population.avg_fitness_history = avg_fitness_history
        
        return {
            'population': population,
            'generation': generation,
            'mode': mode,
            'best_fitness': max(best_fitness_history) if best_fitness_history else 0
        }
    
    except Exception as e:
        print(f"❌ Error loading from model history: {e}")
        import traceback
        traceback.print_exc()
        return None


def _model_data_to_chromosome(model_data: Dict[str, Any]) -> Chromosome:
    """Convert model_data from history to Chromosome"""
    chromosome = Chromosome()

    # Restore layers
    chromosome.layers = []
    for layer_data in model_data['architecture']['layers']:
        layer = Layer(
            layer_type=layer_data['type'],
            neurons=layer_data.get('neurons'),
            activation=layer_data.get('activation'),
            rate=layer_data.get('rate') or layer_data.get('dropout_rate'),
            filters=layer_data.get('filters'),
            kernel_size=layer_data.get('kernel_size'),
            pool_size=layer_data.get('pool_size'),
            l2_reg=layer_data.get('l2_reg', 0.0),
        )
        chromosome.layers.append(layer)

    # Restore hyperparameters
    hp = model_data['hyperparameters']
    chromosome.optimizer = hp['optimizer']
    chromosome.learning_rate = hp['learning_rate']
    chromosome.batch_size = hp['batch_size']
    chromosome.lr_scheduler = hp.get('lr_scheduler', 'none')

    # Restore fitness
    chromosome.fitness = model_data['fitness']
    chromosome.trained = model_data['trained']

    return chromosome
