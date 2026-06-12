"""
analyze_6experiments.py
=======================
Аналіз 6 експериментів: 3 методи x 2 датасети (data / data2)
"""

import json
import os
from pathlib import Path

RESULTS_DIR = Path("results_6exp")

EXPERIMENTS = {
    "data_baseline":  {"method": "Baseline",  "dataset": "data (clean)"},
    "data_warmstart": {"method": "Warm Start", "dataset": "data (clean)"},
    "data_pareto":    {"method": "Pareto",     "dataset": "data (clean)"},
    "data2_baseline": {"method": "Baseline",   "dataset": "data2 (occluded)"},
    "data2_warmstart":{"method": "Warm Start", "dataset": "data2 (occluded)"},
    "data2_pareto":   {"method": "Pareto",     "dataset": "data2 (occluded)"},
}


def load_experiment(name):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_accuracy(data):
    best = 0.0
    for gen in data["generations"]:
        for m in gen["models"]:
            if m.get("fitness") and m["fitness"] > best:
                best = m["fitness"]
    return best


def convergence(data):
    """Best accuracy per generation."""
    curve = []
    for gen in data["generations"]:
        fitnesses = [m["fitness"] for m in gen["models"] if m.get("fitness")]
        curve.append(max(fitnesses) if fitnesses else 0.0)
    return curve


def best_architecture(data):
    """Return model_data of the best model."""
    best_m, best_f = None, 0.0
    for gen in data["generations"]:
        for m in gen["models"]:
            if m.get("fitness") and m["fitness"] > best_f:
                best_f = m["fitness"]
                best_m = m
    return best_m


def layer_type_counts(model_data):
    counts = {}
    for layer in model_data["architecture"]["layers"]:
        t = layer["type"]
        counts[t] = counts.get(t, 0) + 1
    return counts


def summarize_arch(model_data):
    layers = model_data["architecture"]["layers"]
    types = [l["type"] for l in layers]
    hp = model_data["hyperparameters"]
    params = model_data.get("num_params", "?")
    lines = []
    lines.append(f"  Layers ({len(layers)}): {' -> '.join(types)}")
    lines.append(f"  Optimizer: {hp['optimizer']}, LR: {hp['learning_rate']:.5f}, "
                 f"Batch: {hp['batch_size']}, Scheduler: {hp.get('lr_scheduler','?')}")
    lines.append(f"  Params: {params:,}" if isinstance(params, int) else f"  Params: {params}")
    # L2 info
    l2_layers = [(l["type"], l.get("l2_reg",0)) for l in layers if l.get("l2_reg",0) > 0]
    if l2_layers:
        lines.append(f"  L2 reg: {l2_layers}")
    # depthwise / GAP
    special = [l["type"] for l in layers if l["type"] in ("depthwise_conv","global_avg_pool")]
    if special:
        lines.append(f"  Special layers: {special}")
    return "\n".join(lines)


print("=" * 70)
print("  6-EXPERIMENT ANALYSIS: 3 methods x 2 datasets")
print("=" * 70)

results = {}
for name, meta in EXPERIMENTS.items():
    try:
        data = load_experiment(name)
        acc = best_accuracy(data)
        meta_data = data.get("metadata", {})
        results[name] = {
            "method": meta["method"],
            "dataset": meta["dataset"],
            "best_acc": acc,
            "generations": meta_data.get("total_generations", "?"),
            "total_models": meta_data.get("total_models", "?"),
            "convergence": convergence(data),
            "best_model": best_architecture(data),
        }
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")

# Summary table
print("\n{'EXPERIMENT':<20} {'DATASET':<18} {'BEST ACC':>10} {'GENS':>5} {'MODELS':>7}")
print("-" * 70)
for name, r in results.items():
    print(f"{r['method']:<20} {r['dataset']:<18} {r['best_acc']:>10.4f} "
          f"{r['generations']:>5} {r['total_models']:>7}")

# Convergence curves
print("\n\n=== CONVERGENCE (best acc per generation) ===")
for name, r in results.items():
    curve_str = "  ".join(f"{v:.3f}" for v in r["convergence"])
    print(f"\n{r['method']:12} / {r['dataset']}:")
    print(f"  {curve_str}")

# Best architectures
print("\n\n=== BEST ARCHITECTURES ===")
for name, r in results.items():
    bm = r["best_model"]
    if bm is None:
        continue
    print(f"\n{'='*60}")
    print(f"  {r['method']} / {r['dataset']}  ->  acc={r['best_acc']:.4f}")
    print(summarize_arch(bm))

# data vs data2 comparison
print("\n\n=== data vs data2 COMPARISON ===")
for method in ("Baseline", "Warm Start", "Pareto"):
    clean = next(r for r in results.values() if r["method"]==method and "clean" in r["dataset"])
    occ   = next(r for r in results.values() if r["method"]==method and "occluded" in r["dataset"])
    diff = occ["best_acc"] - clean["best_acc"]
    sign = "+" if diff >= 0 else ""
    print(f"  {method:<12}: clean={clean['best_acc']:.4f}  occluded={occ['best_acc']:.4f}  "
          f"diff={sign}{diff:.4f}")

print("\nDone.")
