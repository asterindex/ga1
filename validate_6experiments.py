"""
Validate best models from all 6 experiments.

For each experiment directory, loads the best model's architecture + weights,
evaluates on the test split of the corresponding dataset, and prints a summary.
"""

import json
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensorflow import keras
from chromosome import Chromosome, Layer
import config as cfg
from dataset_loader import load_dataset

# -----------------------------------------------------------------------
# Experiment config: (output_dir, dataset_path)
# -----------------------------------------------------------------------
EXPERIMENTS = [
    ("data_baseline",   "data"),
    ("data_warmstart",  "data"),
    ("data_pareto",     "data"),
    ("data2_baseline",  "data2"),
    ("data2_warmstart", "data2"),
    ("data2_pareto",    "data2"),
]


def build_model_from_data(model_data: dict, input_shape: tuple) -> keras.Model:
    """Reconstruct a Keras model from JSON architecture description."""
    chromosome = Chromosome()
    chromosome.optimizer  = model_data['hyperparameters']['optimizer']
    chromosome.learning_rate = model_data['hyperparameters']['learning_rate']
    chromosome.batch_size = model_data['hyperparameters']['batch_size']
    chromosome.fitness    = model_data['fitness']

    chromosome.layers = []
    for ld in model_data['architecture']['layers']:
        layer = Layer(ld['type'])
        if 'neurons'      in ld: layer.neurons      = ld['neurons']
        if 'activation'   in ld: layer.activation   = ld['activation']
        if 'filters'      in ld: layer.filters       = ld['filters']
        if 'kernel_size'  in ld: layer.kernel_size   = ld['kernel_size']
        if 'pool_size'    in ld: layer.pool_size     = ld['pool_size']
        if 'rate'         in ld: layer.rate          = ld['rate']
        elif 'dropout_rate' in ld: layer.rate        = ld['dropout_rate']
        if 'l2_reg'       in ld: layer.l2_reg        = ld['l2_reg']
        chromosome.layers.append(layer)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    for layer in chromosome.layers:
        l2 = (keras.regularizers.l2(layer.l2_reg)
              if getattr(layer, 'l2_reg', 0.0) > 0 else None)

        lt = layer.layer_type
        if lt == 'conv2d':
            model.add(keras.layers.Conv2D(
                layer.filters, layer.kernel_size,
                activation=layer.activation, padding='same',
                kernel_regularizer=l2))
        elif lt == 'depthwise_conv':
            model.add(keras.layers.SeparableConv2D(
                layer.filters, layer.kernel_size,
                activation=layer.activation, padding='same',
                depthwise_regularizer=l2, pointwise_regularizer=l2))
        elif lt == 'batch_norm':
            model.add(keras.layers.BatchNormalization())
        elif lt == 'maxpool':
            model.add(keras.layers.MaxPooling2D(pool_size=layer.pool_size))
        elif lt == 'flatten':
            model.add(keras.layers.Flatten())
        elif lt == 'global_avg_pool':
            model.add(keras.layers.GlobalAveragePooling2D())
        elif lt == 'dense':
            model.add(keras.layers.Dense(
                layer.neurons, activation=layer.activation,
                kernel_regularizer=l2))
        elif lt == 'dropout':
            model.add(keras.layers.Dropout(layer.rate))

    optimizer_map = {
        'adam':    keras.optimizers.Adam(learning_rate=chromosome.learning_rate),
        'sgd':     keras.optimizers.SGD(learning_rate=chromosome.learning_rate),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=chromosome.learning_rate),
    }
    model.compile(
        optimizer=optimizer_map.get(chromosome.optimizer, 'adam'),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model, chromosome


def load_best_models(history_path: Path, top_n: int = 3):
    """Return top-N models by fitness from all_models.json."""
    with open(history_path) as f:
        data = json.load(f)
    all_models = [m for g in data['generations'] for m in g['models']]
    all_models.sort(key=lambda m: m.get('fitness', 0), reverse=True)
    return all_models[:top_n], data['metadata']


def validate_experiment(exp_dir: str, dataset_path: str, top_n: int = 3):
    """Validate top-N models for one experiment. Returns list of result dicts."""
    history_path = Path(exp_dir) / 'model_history' / 'all_models.json'
    weights_dir  = Path(exp_dir) / 'model_history' / 'weights'

    if not history_path.exists():
        print(f"  ⚠️  History not found: {history_path}")
        return []

    top_models, meta = load_best_models(history_path, top_n=top_n)
    if not top_models:
        print(f"  ⚠️  No models in {history_path}")
        return []

    # Load dataset once (override config path)
    print(f"  Loading dataset: {dataset_path} ...", end=' ', flush=True)
    original_path = cfg.DATASET_PATH
    cfg.DATASET_PATH = dataset_path
    X_train, y_train, X_test, y_test = load_dataset(subset_size=None)
    cfg.DATASET_PATH = original_path
    print(f"test={len(X_test)}")
    input_shape = X_test.shape[1:]

    results = []
    for model_data in top_models:
        model_id = model_data['model_id']
        orig_fitness = model_data.get('fitness', 0.0)

        # Try loading saved .h5 first, then rebuild
        weights_file = model_data.get('weights_file', '')
        full_weights_path = weights_dir / weights_file if weights_file else None

        try:
            if full_weights_path and full_weights_path.exists():
                model = keras.models.load_model(str(full_weights_path), compile=True)
                load_method = 'h5'
            else:
                model, _ = build_model_from_data(model_data, input_shape)
                load_method = 'rebuild (no weights)'

            loss, acc = model.evaluate(X_test, y_test, verbose=0, batch_size=128)
            diff = acc - orig_fitness
            print(f"    {model_id}: orig={orig_fitness:.4f}  test={acc:.4f}  "
                  f"diff={diff:+.4f}  [{load_method}]")

            results.append({
                'model_id':        model_id,
                'exp_dir':         exp_dir,
                'dataset':         dataset_path,
                'original_fitness': float(orig_fitness),
                'test_accuracy':   float(acc),
                'test_loss':       float(loss),
                'difference':      float(diff),
                'load_method':     load_method,
                'status':          'success',
            })

            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"    {model_id}: ❌ {e}")
            results.append({
                'model_id': model_id,
                'exp_dir':  exp_dir,
                'dataset':  dataset_path,
                'original_fitness': float(orig_fitness),
                'test_accuracy':    None,
                'status':           'error',
                'error':            str(e),
            })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-n', type=int, default=3,
                        help='Validate top-N best models per experiment (default 3)')
    parser.add_argument('--output', default='validation_6exp_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATION: 6 EXPERIMENTS")
    print(f"Top-{args.top_n} models per experiment")
    print("=" * 70)

    all_results = {}
    summary_rows = []

    for exp_dir, dataset_path in EXPERIMENTS:
        print(f"\n[{exp_dir}]")
        res = validate_experiment(exp_dir, dataset_path, top_n=args.top_n)
        all_results[exp_dir] = res

        best = max((r for r in res if r['status'] == 'success'),
                   key=lambda r: r['test_accuracy'], default=None)
        if best:
            summary_rows.append({
                'experiment':       exp_dir,
                'best_test_acc':    best['test_accuracy'],
                'orig_fitness':     best['original_fitness'],
                'diff':             best['difference'],
            })

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<22} {'Orig fitness':>14} {'Test acc':>10} {'Diff':>8}")
    print("-" * 60)
    for row in summary_rows:
        print(f"{row['experiment']:<22} {row['orig_fitness']:>14.4f} "
              f"{row['best_test_acc']:>10.4f} {row['diff']:>+8.4f}")

    # Save
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'top_n': args.top_n,
        },
        'results': all_results,
        'summary': summary_rows,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
