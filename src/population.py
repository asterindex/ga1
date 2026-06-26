"""
Population management for chromosomes
"""

import random
from typing import List
import config
from chromosome import Chromosome
import operators

class Population:
    """Class for population management"""
    
    def __init__(self, size: int, input_shape, output_dim: int):
        self.size = size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.individuals: List[Chromosome] = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_random(self):
        """Initialize population with random chromosomes"""
        self.individuals = [
            Chromosome.random(self.input_shape, self.output_dim) 
            for _ in range(self.size)
        ]
        print(f"Ініціалізовано популяцію з {self.size} особин")
    
    def get_best(self) -> Chromosome:
        """Return best individual"""
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_worst(self) -> Chromosome:
        """Return worst individual"""
        return min(self.individuals, key=lambda x: x.fitness)
    
    def get_average_fitness(self) -> float:
        """Return average fitness of population"""
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)
    
    def sort_by_fitness(self):
        """Sort population by fitness (best to worst)"""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
    
    def update_statistics(self):
        """Update population statistics"""
        best_fitness = self.get_best().fitness
        avg_fitness = self.get_average_fitness()
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
    
    def evolve(self) -> 'Population':
        """
        Create new generation through selection, crossover and mutation
        """
        new_population = Population(self.size, self.input_shape, self.output_dim)
        new_population.generation = self.generation + 1
        
        # Elitism: keep best individual
        self.sort_by_fitness()
        elite = self.individuals[0].copy()
        new_individuals = [elite]
        
        # Generate rest of population
        while len(new_individuals) < self.size:
            # Selection
            parent1 = operators.tournament_selection(self.individuals)
            parent2 = operators.tournament_selection(self.individuals)
            
            # Crossover
            if random.random() < config.CROSSOVER_RATE:
                child1, child2 = operators.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation (повертає мутовану хромосому або оригінал якщо невалідна)
            if random.random() < config.MUTATION_RATE:
                child1 = operators.mutate(child1)
            if random.random() < config.MUTATION_RATE:
                child2 = operators.mutate(child2)
            
            new_individuals.append(child1)
            if len(new_individuals) < self.size:
                new_individuals.append(child2)
        
        new_population.individuals = new_individuals[:self.size]
        
        # Copy history
        new_population.best_fitness_history = self.best_fitness_history.copy()
        new_population.avg_fitness_history = self.avg_fitness_history.copy()
        
        return new_population
    
    def pareto_evolve(self, mode: str = config.OBJECTIVE_MODE_STANDARD) -> 'Population':
        """
        Create new generation using NSGA-II crowded-comparison operator.
        Parents are chosen via Pareto rank + crowding distance instead of
        pure tournament selection.
        """
        import pareto as pareto_mod

        new_population = Population(self.size, self.input_shape, self.output_dim)
        new_population.generation = self.generation + 1

        # Assign ranks/distances for current population (used in logging)
        pareto_mod.assign_ranks(self.individuals, mode=mode)

        # Elitism: keep the individual with highest accuracy from front 0
        front0 = [ind for ind in self.individuals if getattr(ind, 'pareto_rank', 999) == 0]
        if front0:
            elite = max(front0, key=lambda x: x.fitness).copy()
        else:
            elite = self.get_best().copy()
        new_individuals = [elite]

        # Fill rest with offspring selected by Pareto crowded-comparison
        pool = pareto_mod.pareto_select(self.individuals, self.size, mode=mode)

        while len(new_individuals) < self.size:
            parent1 = random.choice(pool).copy()
            parent2 = random.choice(pool).copy()

            if random.random() < config.CROSSOVER_RATE:
                child1, child2 = operators.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < config.MUTATION_RATE:
                child1 = operators.mutate(child1)
            if random.random() < config.MUTATION_RATE:
                child2 = operators.mutate(child2)

            new_individuals.append(child1)
            if len(new_individuals) < self.size:
                new_individuals.append(child2)

        new_population.individuals = new_individuals[:self.size]

        new_population.best_fitness_history = self.best_fitness_history.copy()
        new_population.avg_fitness_history = self.avg_fitness_history.copy()

        return new_population

    def print_summary(self):
        """Print population summary"""
        best = self.get_best()
        worst = self.get_worst()
        avg = self.get_average_fitness()
        
        print(f"\n{'='*70}")
        print(f"Покоління {self.generation}")
        print(f"{'='*70}")
        print(f"Найкраща точність:  {best.fitness:.4f}")
        print(f"Середня точність:   {avg:.4f}")
        print(f"Найгірша точність:  {worst.fitness:.4f}")
        print(f"\nНайкраща модель:")
        print(f"  Learning rate: {best.learning_rate:.5f}")
        print(f"  Batch size: {best.batch_size}")
        print(f"  Оптимізатор: {best.optimizer}")
        print(f"  Кількість шарів: {len(best.layers)}")
        print(f"  Структура: {' -> '.join([str(layer) for layer in best.layers])}")
        print(f"{'='*70}\n")
