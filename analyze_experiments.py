"""
analyze_experiments.py
======================
Compares results of three NAS experiments:
  1. Baseline  GA  (output/baseline_results.json)
  2. Warm Start GA (output/warmstart_results.json)
  3. Pareto GA     (output/pareto_results.json)

Generates four figures in output/figures/:
  fig_comparison_bar.png       - best accuracy per method
  fig_convergence.png          - convergence curves (accuracy vs evaluations)
  fig_pareto_scatter.png       - accuracy vs num_params (Pareto front highlighted)
  fig_accuracy_boxplot.png     - distribution of all evaluated accuracies

Also prints a summary table to stdout.
"""

import json
import os
import sys
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

METHOD_LABELS = {
    "baseline":  "Базовий GA",
    "warmstart": "GA + Warm Start",
    "pareto":    "GA + Pareto (NSGA-II)",
}
METHOD_COLORS = {
    "baseline":  "#4878cf",
    "warmstart": "#e8a020",
    "pareto":    "#6acc65",
}

FIGSIZE = (9, 5)
DPI = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(method: str) -> list:
    # Try per-method directory first (server layout)
    candidates = [
        OUTPUT_DIR / f"output_{method}" / "model_history" / "all_models.json",
        OUTPUT_DIR / f"{method}_results.json",
        BASE_DIR / f"output_{method}" / "model_history" / "all_models.json",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        print(f"WARNING: no results found for '{method}' (tried {[str(c) for c in candidates]})")
        return []

    with open(path) as f:
        data = json.load(f)
    # Flatten: list of generations, each with "models" list
    models = []
    if isinstance(data, list):
        for gen in data:
            models.extend(gen.get("models", []))
    elif isinstance(data, dict):
        for gen in data.get("generations", []):
            models.extend(gen.get("models", []))
    return models


def build_convergence(models: list) -> tuple:
    """
    Returns (eval_indices, best_so_far) where eval_indices[i] is the
    cumulative number of model evaluations at step i, and best_so_far[i]
    is the best accuracy seen up to that step.
    """
    sorted_models = sorted(models, key=lambda m: m.get("generation", 0) * 1000 + m.get("index_in_gen", 0))
    best = -1.0
    xs, ys = [], []
    for i, m in enumerate(sorted_models):
        acc = m.get("fitness", 0.0)
        if acc > best:
            best = acc
        xs.append(i + 1)
        ys.append(best)
    return xs, ys


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 - Best accuracy bar chart
# ---------------------------------------------------------------------------

def plot_bar(results: dict):
    methods = [m for m in ["baseline", "warmstart", "pareto"] if results[m]]
    best_accs = [max(m.get("fitness", 0) for m in results[method]) for method in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [a * 100 for a in best_accs], color=colors,
                  width=0.5, edgecolor='white', linewidth=0.8)

    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{acc * 100:.2f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Найкраща точність на валідації, %")
    ax.set_ylim(min(a * 100 for a in best_accs) - 2,
                max(a * 100 for a in best_accs) + 2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "fig_comparison_bar.png")


# ---------------------------------------------------------------------------
# Figure 2 - Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(results: dict):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method in ["baseline", "warmstart", "pareto"]:
        if not results[method]:
            continue
        xs, ys = build_convergence(results[method])
        ax.plot(xs, [y * 100 for y in ys],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linewidth=2)

    ax.set_xlabel("Кількість навчених моделей")
    ax.set_ylabel("Найкраща точність, %")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "fig_convergence.png")


# ---------------------------------------------------------------------------
# Figure 3 - Pareto scatter (accuracy vs num_params)
# ---------------------------------------------------------------------------

def plot_pareto_scatter(results: dict):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method in ["baseline", "warmstart", "pareto"]:
        if not results[method]:
            continue
        models = results[method]
        xs = [m.get("num_params", 0) / 1000 for m in models]   # thousands
        ys = [m.get("fitness", 0) * 100 for m in models]
        ax.scatter(xs, ys,
                   label=METHOD_LABELS[method],
                   color=METHOD_COLORS[method],
                   alpha=0.45, s=20)

    # Highlight Pareto-optimal models (accuracy vs params) across all methods
    all_models = []
    for method in results:
        for m in results[method]:
            all_models.append((m.get("fitness", 0), m.get("num_params", 0)))

    pareto_pts = _pareto_front_2d(all_models)
    if pareto_pts:
        px = [p[1] / 1000 for p in pareto_pts]
        py = [p[0] * 100 for p in pareto_pts]
        order = sorted(range(len(px)), key=lambda i: px[i])
        ax.plot([px[i] for i in order], [py[i] for i in order],
                'k--', linewidth=1, label="Фронт Парето (acc vs params)")

    ax.set_xlabel("Кількість параметрів моделі, тис.")
    ax.set_ylabel("Точність на валідації, %")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "fig_pareto_scatter.png")


def _pareto_front_2d(points):
    """Return non-dominated points (acc, params) where acc-max, params-min."""
    dominated = set()
    n = len(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ai, pi = points[i]
            aj, pj = points[j]
            if aj >= ai and pj <= pi and (aj > ai or pj < pi):
                dominated.add(i)
                break
    return [points[i] for i in range(n) if i not in dominated]


# ---------------------------------------------------------------------------
# Figure 4 - Box plot of accuracy distribution
# ---------------------------------------------------------------------------

def plot_boxplot(results: dict):
    methods = [m for m in ["baseline", "warmstart", "pareto"] if results[m]]
    data = [[m.get("fitness", 0) * 100 for m in results[method]]
            for method in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_ylabel("Точність на валідації, %")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, "fig_accuracy_boxplot.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    print("\n" + "=" * 76)
    print(f"{'Метод':<24} {'Найкраща':>9} {'Середня':>9} "
          f"{'Найгірша':>9} {'Моделей':>8} {'Найк. params':>14}")
    print("-" * 76)

    for method in ["baseline", "warmstart", "pareto"]:
        models = results[method]
        if not models:
            print(f"{METHOD_LABELS[method]:<24}  (немає даних)")
            continue
        accs = [m.get("fitness", 0) for m in models]
        best_model = max(models, key=lambda m: m.get("fitness", 0))
        best_params = best_model.get("num_params", 0)
        print(
            f"{METHOD_LABELS[method]:<24} "
            f"{max(accs)*100:>8.2f}% "
            f"{statistics.mean(accs)*100:>8.2f}% "
            f"{min(accs)*100:>8.2f}% "
            f"{len(models):>8} "
            f"{best_params:>14,}"
        )
    print("=" * 76)

    # Time comparison (if available)
    print("\nСередній час тренування однієї моделі:")
    for method in ["baseline", "warmstart", "pareto"]:
        times = [m.get("training_time", 0) for m in results[method]
                 if m.get("training_time", 0) > 0]
        if times:
            print(f"  {METHOD_LABELS[method]}: {statistics.mean(times):.1f}s "
                  f"(total: {sum(times)/60:.1f} хв)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading results...")
    results = {method: load_results(method)
               for method in ["baseline", "warmstart", "pareto"]}

    available = [m for m in results if results[m]]
    if not available:
        print("ERROR: No result files found in output/. "
              "Run run_all_experiments_v3.sh first.")
        sys.exit(1)

    print(f"Methods available: {', '.join(available)}")
    for m in available:
        print(f"  {m}: {len(results[m])} models")

    print("\nGenerating figures...")
    plot_bar(results)
    plot_convergence(results)
    plot_pareto_scatter(results)
    plot_boxplot(results)

    print_summary(results)
    print(f"\nFigures saved to: {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
