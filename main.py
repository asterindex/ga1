"""
Main file for launching the genetic algorithm NAS.

Methods:
  baseline  - plain GA, no warm start
  warmstart - GA with Lamarckian warm start
  pareto    - GA with NSGA-II multi-objective selection
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
import utils


def main():
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm for Neural Architecture Search on Military Vehicles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full --method baseline   # plain GA
  python main.py --mode full --method warmstart  # GA + Lamarckian warm start
  python main.py --mode full --method pareto     # GA + NSGA-II Pareto front
  python main.py --mode fast                     # quick test (4 models, 3 gen)
        """
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['full', 'fast'],
                        help='full = 50 epochs, fast = 5 epochs')
    parser.add_argument('--method', type=str, default='baseline',
                        choices=['baseline', 'warmstart', 'pareto'],
                        help='Search method (default: baseline)')
    parser.add_argument('--generations', type=int, default=None,
                        help='Number of generations (overrides mode default)')
    parser.add_argument('--population', type=int, default=None,
                        help='Population size (overrides mode default)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Detailed output (default: on)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')

    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset directory (default: data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: output_METHOD)')

    # Keep legacy flag for backwards compatibility
    parser.add_argument('--no-warm-start', action='store_true',
                        help=argparse.SUPPRESS)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Legacy: --no-warm-start maps to --method baseline
    if args.no_warm_start and args.method == 'baseline':
        pass  # already baseline

    # GPU diagnostics
    print("=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow: {tf.__version__}  |  GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    except Exception as e:
        print(f"GPU check failed: {e}")
    print("=" * 60)

    # Apply mode defaults
    if args.mode == 'fast':
        config.POPULATION_SIZE = 4
        config.NUM_GENERATIONS = config.NUM_GENERATIONS_FAST
        config.TRAINING_EPOCHS_FAST = 3
    else:
        config.POPULATION_SIZE = config.POPULATION_SIZE_FULL
        config.NUM_GENERATIONS = config.NUM_GENERATIONS_FULL

    if args.generations is not None:
        config.NUM_GENERATIONS = args.generations
    if args.population is not None:
        config.POPULATION_SIZE = args.population
    if args.dataset is not None:
        config.DATASET_PATH = args.dataset

    # Random seed
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(args.seed)
        except Exception:
            pass
        print(f"Random seed: {args.seed}")

    # Configure warm start based on method
    if args.method == 'warmstart':
        config.WARM_START_ENABLED = True
        print(f"Method: WARM START (epoch reduction: {config.WARM_START_EPOCH_REDUCTION})")
    else:
        config.WARM_START_ENABLED = False
        if args.method == 'pareto':
            print("Method: PARETO (NSGA-II, 3 objectives)")
        else:
            print("Method: BASELINE (plain GA)")

    verbose = args.verbose and not args.quiet
    if args.output:
        output_dir = args.output
    else:
        output_dir = f'output_{args.method}'

    if not args.resume:
        import shutil
        if os.path.exists(output_dir):
            print("Cleaning output directory...")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Failed to delete {item_path}: {e}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning NAS: method={args.method}  "
          f"pop={config.POPULATION_SIZE}  gen={config.NUM_GENERATIONS}")

    # Select and run engine
    if args.method == 'pareto':
        from pareto_evolution import ParetoEvolutionEngine
        engine = ParetoEvolutionEngine(
            mode=args.mode, verbose=verbose, output_dir=output_dir)
    else:
        from evolution import EvolutionEngine
        engine = EvolutionEngine(
            mode=args.mode, verbose=verbose, output_dir=output_dir,
            method=args.method)

    population, best_chromosome = engine.run(resume=args.resume)

    utils.print_population_diversity(population)

    best_model_path = os.path.join(output_dir, 'best_model.json')
    utils.save_chromosome(best_chromosome, best_model_path)

    evolution_plot_path = os.path.join(output_dir, 'evolution.png')
    utils.plot_evolution_history(population, evolution_plot_path)

    print("\nDone!")
    print(f"Results in: {output_dir}/")
    print(f"  best_model.json | evolution.png | model_history/")


if __name__ == '__main__':
    main()
