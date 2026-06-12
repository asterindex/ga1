"""
Main evolution loop for genetic algorithm
"""

import numpy as np
import time
import os
from datetime import datetime
from typing import Tuple, Optional
import config
from population import Population
from chromosome import Chromosome
from model_history import ModelHistoryTracker
from history_loader import load_from_model_history
from logger import get_logger


class EvolutionEngine:
    """Evolution engine"""

    def __init__(self, mode: str = config.MODE_FULL, verbose: bool = True,
                 output_dir: str = 'output', method: str = 'baseline'):
        self.mode = mode
        self.verbose = verbose
        self.output_dir = output_dir
        self.method = method
        self.epochs = (config.TRAINING_EPOCHS_FULL if mode == config.MODE_FULL
                       else config.TRAINING_EPOCHS_FAST)

        # Create subdirectories in output
        os.makedirs(output_dir, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"evolution_{mode.lower()}_{timestamp}.log"

        self.history_tracker = ModelHistoryTracker(
            history_dir=os.path.join(output_dir, 'model_history'),
            method=method,
            dataset=config.DATASET_PATH,
        )
        self.logger = get_logger(verbose=verbose, log_file=os.path.join(output_dir, log_filename))
        
        # Select fitness module
        import fitness as fitness_module
        self.logger.info("🔧 Використовується TensorFlow")
        self.fitness = fitness_module
        
        self.logger.header("🧬 Ініціалізація генетичного алгоритму")
        self.logger.info(f"Режим: {mode.upper()}")
        self.logger.info(f"Епох тренування: {self.epochs}")
        self.logger.info(f"Розмір популяції: {config.POPULATION_SIZE}")
        self.logger.info(f"Кількість поколінь: {config.NUM_GENERATIONS}")
        self.logger.info(f"Ймовірність кросоверу: {config.CROSSOVER_RATE}")
        self.logger.info(f"Ймовірність мутації: {config.MUTATION_RATE}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data"""
        self.logger.info("📊 Завантаження датасету...")
        
        # Визначаємо розмір вибірки залежно від режиму
        subset_size = config.DATASET_SUBSET_FAST if self.mode == config.MODE_FAST else config.DATASET_SUBSET_FULL
        
        X_train, y_train, X_val, y_val = self.fitness.load_dataset(subset_size=subset_size)
        self.logger.success(f"Датасет завантажено")
        self.logger.debug(f"  ├─ Тренувальний набір: {X_train.shape}")
        self.logger.debug(f"  └─ Валідаційний набір: {X_val.shape}")
        return X_train, y_train, X_val, y_val
    
    
    def evaluate_population(self, population: Population,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           current_generation: int,
                           total_generations: int):
        """Evaluate entire population and return trained models"""
        self.logger.subheader(f"🔬 Оцінка популяції (покоління {population.generation})")
        
        untrained_count = sum(1 for ind in population.individuals if not ind.trained)
        self.logger.info(f"Моделей для тренування: {untrained_count}/{len(population.individuals)}")
        
        # Dictionary to store trained models for saving weights
        trained_models = {}
        
        for i, individual in enumerate(population.individuals):
            if not individual.trained:
                self.logger.info(f"🏋️  Тренування моделі {i+1}/{len(population.individuals)}")
                
                # Log model parameters
                self.logger.model_info(i+1, len(population.individuals), {
                    'optimizer': individual.optimizer,
                    'learning_rate': individual.learning_rate,
                    'batch_size': individual.batch_size,
                    'num_layers': len(individual.layers)
                })
                
                # Train model and get trained model back
                accuracy, trained_model = self.fitness.evaluate_fitness(
                    individual,
                    X_train, y_train,
                    X_val, y_val,
                    epochs=self.epochs,
                    verbose=1,
                    return_model=True  # Request trained model
                )
                
                individual.fitness = accuracy
                individual.trained = True
                
                # Store trained model for weight saving
                if trained_model is not None:
                    trained_models[i] = trained_model
                
                # Log result with emoji depending on accuracy
                accuracy_str = f"{accuracy:.4f}".replace('.', ',')
                if accuracy > 0.95:
                    self.logger.success(f"  └─ Точність: {accuracy_str} 🏆")
                elif accuracy > 0.90:
                    self.logger.success(f"  └─ Точність: {accuracy_str} ⭐")
                elif accuracy > 0.85:
                    self.logger.info(f"  └─ Точність: {accuracy_str} ✓")
                else:
                    self.logger.warning(f"  └─ Точність: {accuracy_str}")
        
        return trained_models
    
    def run(self, resume: bool = False) -> Tuple[Population, Chromosome]:
        """
        Run full evolution cycle
        
        Args:
            resume: Resume from checkpoint (if False - new evolution)
            
        Returns: (final population, best individual)
        """
        # Load data
        X_train, y_train, X_val, y_val = self.load_data()
        
        # Get dimensions
        input_shape, output_dim = self.fitness.get_dataset_info()
        
        # Restore from model history if --resume
        start_generation = 0
        if resume:
            self.logger.info("🔍 Пошук історії моделей...")
            history_data = load_from_model_history()
            
            if history_data:
                population = history_data['population']
                start_generation = history_data['generation'] + 1
                self.logger.checkpoint_loaded(history_data['generation'], history_data['best_fitness'])
                self.logger.info(f"▶️  Продовження з покоління {start_generation}")
            else:
                self.logger.warning("❌ Історію не знайдено, починаємо з нуля")
                population = Population(config.POPULATION_SIZE, input_shape, output_dim)
                self.logger.info("🎲 Ініціалізація випадкової популяції...")
                population.initialize_random()
                trained_models = self.evaluate_population(population, X_train, y_train, X_val, y_val, 0, config.NUM_GENERATIONS)
                population.update_statistics()
                self.logger.generation_summary(0, population.get_best().fitness, 
                                              population.get_average_fitness(), 
                                              population.get_worst().fitness)
                self.logger.best_model_details(population.get_best())
                
                # Save models with weights
                saved_paths = self.history_tracker.add_generation(0, population.individuals, self.mode, trained_models)
                for idx, path in saved_paths.items():
                    population.individuals[idx].parent_weights_file = path
                history_file = self.history_tracker.save_history()
                self.logger.debug(f"💾 Історію моделей оновлено: {history_file}")
        else:
            # Initialize new population
            population = Population(config.POPULATION_SIZE, input_shape, output_dim)
            self.logger.info("🎲 Ініціалізація випадкової популяції...")
            population.initialize_random()
            self.logger.success(f"Створено {len(population.individuals)} випадкових моделей")
            
            # Evaluate initial population
            trained_models = self.evaluate_population(population, X_train, y_train, X_val, y_val, 0, config.NUM_GENERATIONS)
            population.update_statistics()
            self.logger.generation_summary(0, population.get_best().fitness, 
                                          population.get_average_fitness(), 
                                          population.get_worst().fitness)
            self.logger.best_model_details(population.get_best())
            
            # Save all models from generation 0 to history with weights
            saved_paths = self.history_tracker.add_generation(0, population.individuals, self.mode, trained_models)
            for idx, path in saved_paths.items():
                population.individuals[idx].parent_weights_file = path
            
            # Save history to disk after generation 0 (incremental save)
            history_file = self.history_tracker.save_history()
            self.logger.debug(f"💾 Історію моделей оновлено: {history_file}")
        
        # Main evolution loop
        for generation in range(start_generation, config.NUM_GENERATIONS):
            self.logger.header(f"🧬 Покоління {generation + 1}/{config.NUM_GENERATIONS}")
            self.logger.elapsed_time()
            
            # Create new generation
            self.logger.info("🔄 Створення нового покоління...")
            self.logger.genetic_operation('elitism', 'Зберігаємо найкращу особину')
            
            prev_best = population.get_best().fitness
            population = population.evolve()
            
            self.logger.genetic_operation('selection', f'Турнірна селекція (розмір турніру: {config.TOURNAMENT_SIZE})')
            self.logger.genetic_operation('crossover', f'Ймовірність: {config.CROSSOVER_RATE}')
            self.logger.genetic_operation('mutation', f'Ймовірність: {config.MUTATION_RATE}')
            self.logger.success(f"Нове покоління створено ({len(population.individuals)} особин)")
            
            # Evaluate new generation
            trained_models = self.evaluate_population(population, X_train, y_train, X_val, y_val, generation + 1, config.NUM_GENERATIONS)
            
            # Update statistics
            population.update_statistics()
            curr_best = population.get_best().fitness
            
            # Check if best model improved
            if curr_best > prev_best:
                improvement = curr_best - prev_best
                self.logger.success(f"📈 Покращення! Δ = +{improvement:.4f}")
            elif curr_best < prev_best:
                decline = prev_best - curr_best
                self.logger.warning(f"📉 Погіршення. Δ = -{decline:.4f}")
            else:
                self.logger.info("➡️  Без змін")
            
            self.logger.generation_summary(generation + 1, curr_best, 
                                          population.get_average_fitness(), 
                                          population.get_worst().fitness)
            self.logger.best_model_details(population.get_best())
            
            # Save all models from this generation to history with weights
            saved_paths = self.history_tracker.add_generation(generation + 1, population.individuals, self.mode, trained_models)
            
            # Оновлюємо parent_weights_file в хромосомах для наступного покоління
            for idx, path in saved_paths.items():
                population.individuals[idx].parent_weights_file = path
            
            # Save history to disk after each generation (incremental save)
            history_file = self.history_tracker.save_history()
            self.logger.debug(f"💾 Історію моделей оновлено: {history_file}")
        
        # Final results
        best_individual = population.get_best()
        
        self.logger.final_summary(best_individual.fitness, config.NUM_GENERATIONS)
        self.logger.info("🏆 Найкраща знайдена модель:")
        self.logger.info(f"  ├─ Точність: {best_individual.fitness:.4f}")
        self.logger.info(f"  ├─ Learning rate: {best_individual.learning_rate:.5f}")
        self.logger.info(f"  ├─ Batch size: {best_individual.batch_size}")
        self.logger.info(f"  ├─ Оптимізатор: {best_individual.optimizer}")
        self.logger.info(f"  └─ Кількість шарів: {len(best_individual.layers)}")
        self.logger.debug("\n📋 Структура:")
        for i, layer in enumerate(best_individual.layers):
            self.logger.debug(f"  {i+1}. {layer}")
        
        # Save complete model history
        history_file = self.history_tracker.save_history()
        stats = self.history_tracker.get_statistics()
        self.logger.success(f"💾 Повну історію моделей збережено: {history_file}")
        self.logger.info(f"  ├─ Всього моделей: {stats['total_models']}")
        self.logger.info(f"  ├─ Найкраща точність: {stats['best_fitness']:.4f}")
        self.logger.info(f"  ├─ Середня точність: {stats['avg_fitness']:.4f}")
        self.logger.info(f"  └─ Найгірша точність: {stats['worst_fitness']:.4f}")
        
        return population, best_individual
