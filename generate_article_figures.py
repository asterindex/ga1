"""Generate article figures from 6-experiment results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

OUT = Path("article/figures")
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = Path("results_6exp")

EXPS = {
    "data_baseline":   ("Baseline",   "data",  "#4878CF"),
    "data_warmstart":  ("Warm Start", "data",  "#6ACC65"),
    "data_pareto":     ("Pareto",     "data",  "#D65F5F"),
    "data2_baseline":  ("Baseline",   "data2", "#4878CF"),
    "data2_warmstart": ("Warm Start", "data2", "#6ACC65"),
    "data2_pareto":    ("Pareto",     "data2", "#D65F5F"),
}

def load(name):
    with open(RESULTS / f"{name}.json") as f:
        d = json.load(f)
    gens = d["generations"]
    all_models = [m for g in gens for m in g["models"]]
    return gens, all_models

def best_per_gen(gens):
    best = []
    current_best = 0
    for g in gens:
        gen_best = max((m["fitness"] for m in g["models"]), default=0)
        current_best = max(current_best, gen_best)
        best.append(current_best)
    return best

# ── Figure 1: Bar chart of best accuracy ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

methods = ["Baseline", "Warm Start", "Pareto"]
data_accs  = []
data2_accs = []

for prefix in ["data_baseline", "data_warmstart", "data_pareto"]:
    _, models = load(prefix)
    data_accs.append(max(m["fitness"] for m in models) * 100)
for prefix in ["data2_baseline", "data2_warmstart", "data2_pareto"]:
    _, models = load(prefix)
    data2_accs.append(max(m["fitness"] for m in models) * 100)

x = np.arange(len(methods))
w = 0.35
bars1 = ax.bar(x - w/2, data_accs,  w, label="data (чистий)",      color="#4C72B0", alpha=0.88)
bars2 = ax.bar(x + w/2, data2_accs, w, label="data2 (з оклюзіями)", color="#DD8452", alpha=0.88)

ax.set_ylabel("Точність на тестовій вибірці (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(96.5, 100.2)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "fig1_results_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig1 done")

# ── Figure 2: Convergence curves ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

styles = {
    "Baseline":   ("-",  "o", "#4C72B0"),
    "Warm Start": ("--", "s", "#DD8452"),
    "Pareto":     (":",  "^", "#55A868"),
}

for ax, dataset, title in zip(axes, ["data", "data2"], ["data (чистий)", "data2 (з оклюзіями)"]):
    for method, (ls, mk, col) in styles.items():
        key = f"{dataset}_{method.lower().replace(' ', '')}"
        gens, _ = load(key)
        curve = best_per_gen(gens)
        gen_ids = list(range(len(curve)))
        ax.plot(gen_ids, [v * 100 for v in curve],
                linestyle=ls, marker=mk, color=col, label=method,
                linewidth=1.8, markersize=5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Покоління", fontsize=10)
    ax.set_ylabel("Найкраща accuracy (%)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig2_convergence.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig2 done")

# ── Figure 3: Augmentation example ──────────────────────────────────────────
import os, random
random.seed(42)

img_dir_clean  = Path("data/Images")
img_dir_occl   = Path("data2/Images")

# Pick a file present in both
files = [f.name for f in img_dir_clean.glob("*.jpg") if (img_dir_occl / f.name).exists()]
fname = random.choice(files)

img_clean = Image.open(img_dir_clean / fname).convert("RGB")
img_occl  = Image.open(img_dir_occl  / fname).convert("RGB")

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
axes[0].imshow(img_clean)
axes[0].set_title("a) Оригінальне зображення", fontsize=10)
axes[0].axis("off")
axes[1].imshow(img_occl)
axes[1].set_title("b) З об'єктно-прив'язаним Random Erasing", fontsize=10)
axes[1].axis("off")
fig.tight_layout(pad=0.5)
fig.savefig(OUT / "fig3_augmentation_example.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"fig3 done (sample: {fname})")

print("All figures saved to", OUT)
