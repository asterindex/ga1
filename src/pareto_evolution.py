"""
ParetoEvolutionEngine - GA with NSGA-II multi-objective selection.

Objectives:
  1. Maximise val_accuracy
  2. Minimise num_params
  3. Minimise training_time

Differences from EvolutionEngine:
  - population.pareto_evolve() is used instead of population.evolve()
  - pareto.assign_ranks() is called after each evaluation
  - History entries include pareto_rank, crowding_distance, num_params, training_time
"""

import numpy as np
import time
import os
from datetime import datetime
from typing import Tuple

import config
from population import Population
from chromosome import Chromosome
from model_history import ModelHistoryTracker
from history_loader import load_from_model_history
from logger import get_logger
import pareto as pareto_mod


class ParetoEvolutionEngine:
    """Evolution engine using NSGA-II multi-objective selection."""

    def __init__(self, mode: str = config.MODE_FULL, verbose: bool = True,
                 output_dir: str = 'output'):
        self.mode = mode
        self.verbose = verbose
        self.output_dir = output_dir
        self.epochs = (config.TRAINING_EPOCHS_FULL if mode == config.MODE_FULL
                       else config.TRAINING_EPOCHS_FAST)

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"evolution_pareto_{mode.lower()}_{timestamp}.log"

        self.history_tracker = ModelHistoryTracker(
            history_dir=os.path.join(output_dir, 'model_history'),
            method='pareto',
            dataset=config.DATASET_PATH,
        )
        self.logger = get_logger(
            verbose=verbose,
            log_file=os.path.join(output_dir, log_filename))

        import fitness as fitness_module
        self.fitness = fitness_module

        self.logger.header("Pareto GA (NSGA-II) - Ініціалізація")
        self.logger.info(f"Режим: {mode.upper()}")
        self.logger.info(f"Епох тренування: {self.epochs}")
        self.logger.info(f"Розмір популяції: {config.POPULATION_SIZE}")
        self.logger.info(f"Кількість поколінь: {config.NUM_GENERATIONS}")
        self.logger.info("Цілі: accuracy (max) | num_params (min) | training_time (min)")

    # ------------------------------------------------------------------
    def load_data(self):
        self.logger.info("Завантаження датасету...")
        subset_size = (config.DATASET_SUBSET_FAST if self.mode == config.MODE_FAST
                       else config.DATASET_SUBSET_FULL)
        X_train, y_train, X_val, y_val = self.fitness.load_dataset(
            subset_size=subset_size)
        self.logger.success(
            f"Датасет завантажено: train={X_train.shape}, val={X_val.shape}")
        return X_train, y_train, X_val, y_val

    # ------------------------------------------------------------------
    def evaluate_population(self, population: Population,
                            X_train, y_train, X_val, y_val,
                            current_generation: int,
                            total_generations: int) -> dict:
        self.logger.subheader(
            f"Оцінка популяції (покоління {population.generation})")

        untrained = [i for i, ind in enumerate(population.individuals)
                     if not ind.trained]
        self.logger.info(f"Моделей для тренування: {len(untrained)}"
                         f"/{len(population.individuals)}")

        trained_models = {}

        for i, individual in enumerate(population.individuals):
            if not individual.trained:
                self.logger.info(f"Тренування моделі {i + 1}"
                                 f"/{len(population.individuals)}")

                accuracy, trained_model = self.fitness.evaluate_fitness(
                    individual,
                    X_train, y_train,
                    X_val, y_val,
                    epochs=self.epochs,
                    verbose=1,
                    return_model=True
                )

                individual.fitness = accuracy
                individual.trained = True

                if trained_model is not None:
                    trained_models[i] = trained_model

                self.logger.info(
                    f"  acc={accuracy:.4f}  "
                    f"params={individual.num_params:,}  "
                    f"time={individual.training_time:.1f}s")

        # Assign Pareto ranks after full evaluation
        pareto_mod.assign_ranks(population.individuals)

        front0 = [ind for ind in population.individuals
                  if getattr(ind, 'pareto_rank', 999) == 0]
        self.logger.info(
            f"Pareto front 0: {len(front0)} особин  "
            f"| best acc={max(ind.fitness for ind in front0):.4f}")

        return trained_models

    # ------------------------------------------------------------------
    def run(self, resume: bool = False) -> Tuple[Population, Chromosome]:
        X_train, y_train, X_val, y_val = self.load_data()
        input_shape, output_dim = self.fitness.get_dataset_info()

        start_generation = 0

        if resume:
            self.logger.info("Пошук збережених даних...")
            history_data = load_from_model_history()
            if history_data:
                population = history_data['population']
                start_generation = history_data['generation'] + 1
                self.logger.info(
                    f"Відновлено з покоління {start_generation}")
            else:
                self.logger.warning("Збережених даних не знайдено, починаємо з нуля")
                resume = False

        if not resume:
            population = Population(
                config.POPULATION_SIZE, input_shape, output_dim)
            population.initialize_random()
            self.logger.success(
                f"Ініціалізовано {len(population.individuals)} хромосом")

            trained_models = self.evaluate_population(
                population, X_train, y_train, X_val, y_val,
                0, config.NUM_GENERATIONS)
            population.update_statistics()

            saved_paths = self.history_tracker.add_generation(
                0, population.individuals, self.mode, trained_models)
            for idx, path in saved_paths.items():
                population.individuals[idx].parent_weights_file = path
            self.history_tracker.save_history()

            self._log_generation_summary(population, 0)

        # Main loop
        for generation in range(start_generation, config.NUM_GENERATIONS):
            self.logger.header(
                f"Покоління {generation + 1}/{config.NUM_GENERATIONS}")

            population = population.pareto_evolve()

            trained_models = self.evaluate_population(
                population, X_train, y_train, X_val, y_val,
                generation + 1, config.NUM_GENERATIONS)

            population.update_statistics()

            saved_paths = self.history_tracker.add_generation(
                generation + 1, population.individuals,
                self.mode, trained_models)
            for idx, path in saved_paths.items():
                population.individuals[idx].parent_weights_file = path
            self.history_tracker.save_history()

            self._log_generation_summary(population, generation + 1)

        best = max(population.individuals, key=lambda x: x.fitness)
        self.logger.final_summary(best.fitness, config.NUM_GENERATIONS)
        self.logger.info(
            f"Найкраща модель: acc={best.fitness:.4f}  "
            f"params={best.num_params:,}  "
            f"time={best.training_time:.1f}s  "
            f"pareto_rank={getattr(best, 'pareto_rank', '?')}")

        stats = self.history_tracker.get_statistics()
        self.logger.success(
            f"Всього моделей: {stats['total_models']}  "
            f"| best={stats['best_fitness']:.4f}")

        return population, best

    # ------------------------------------------------------------------
    def _log_generation_summary(self, population: Population,
                                 generation: int) -> None:
        best = population.get_best()
        avg = population.get_average_fitness()
        worst = population.get_worst()
        front0_size = sum(
            1 for ind in population.individuals
            if getattr(ind, 'pareto_rank', 999) == 0)

        self.logger.generation_summary(generation, best.fitness, avg, worst.fitness)
        self.logger.info(
            f"  Pareto front 0 size: {front0_size}  "
            f"| best params: {best.num_params:,}")
