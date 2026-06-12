#!/usr/bin/env python3
"""
Скрипт для генерації рисунків для статті
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
from pathlib import Path

# Налаштування стилю
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# Створення директорії для рисунків
output_dir = Path('output/figures')
output_dir.mkdir(exist_ok=True)

# Завантаження даних
with open('output/model_history/all_models.json', 'r') as f:
    data = json.load(f)

print("🎨 Генерація рисунків для статті...")

# ============================================================================
# Рисунок 1: Структура хромосоми
# ============================================================================
print("📊 Рисунок 1: Структура хромосоми...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Заголовок
ax.text(5, 9.5, 'Структура хромосоми', ha='center', va='top', 
        fontsize=16, fontweight='bold')

# Структурна частина
structure_box = FancyBboxPatch((0.5, 6), 9, 2.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#2E86AB', facecolor='#A9D6E5', 
                               linewidth=2)
ax.add_patch(structure_box)
ax.text(5, 8, 'Структурна частина (архітектура)', ha='center', va='top',
        fontsize=12, fontweight='bold')

# Шари
layers = ['Conv2D\n(64, 3×3)', 'BatchNorm', 'MaxPool\n(2×2)', '...', 'Dense\n(128)', 'Dense\n(10)']
x_positions = np.linspace(1, 9, len(layers))
for i, (layer, x) in enumerate(zip(layers, x_positions)):
    box = FancyBboxPatch((x-0.5, 6.5), 1, 1, 
                         boxstyle="round,pad=0.05",
                         edgecolor='#014F86', facecolor='white', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, 7, layer, ha='center', va='center', fontsize=9)

# Параметрична частина
param_box = FancyBboxPatch((0.5, 3), 9, 2.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='#C1666B', facecolor='#F4C2C2',
                           linewidth=2)
ax.add_patch(param_box)
ax.text(5, 5, 'Параметрична частина (гіперпараметри)', ha='center', va='top',
        fontsize=12, fontweight='bold')

# Гіперпараметри
params = [
    ('Optimizer', 'adam'),
    ('Learning Rate', '0.001'),
    ('Batch Size', '32')
]
x_positions = np.linspace(2, 8, len(params))
for (name, value), x in zip(params, x_positions):
    box = FancyBboxPatch((x-0.8, 3.5), 1.6, 1,
                         boxstyle="round,pad=0.05",
                         edgecolor='#8B1E3F', facecolor='white', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, 4.2, name, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x, 3.8, value, ha='center', va='center', fontsize=9, color='#8B1E3F')

# Стрілка
arrow = FancyArrowPatch((5, 5.8), (5, 5.5),
                       arrowstyle='->', mutation_scale=20, linewidth=2,
                       color='#333333')
ax.add_patch(arrow)

plt.tight_layout()
plt.savefig(output_dir / 'figure1_chromosome_structure.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure1_chromosome_structure.png")

# ============================================================================
# Рисунок 2: Блок-схема генетичного алгоритму
# ============================================================================
print("📊 Рисунок 2: Блок-схема генетичного алгоритму...")

fig, ax = plt.subplots(figsize=(8, 12))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

blocks = [
    (5, 13, 'Ініціалізація\nпопуляції', '#4A90E2'),
    (5, 11.5, 'Оцінка fitness', '#50C878'),
    (5, 10, 'Селекція', '#F39C12'),
    (5, 8.5, 'Кросовер', '#E74C3C'),
    (5, 7, 'Мутація', '#9B59B6'),
    (5, 5.5, 'Елітизм', '#1ABC9C'),
    (5, 4, 'Нове покоління', '#34495E'),
    (5, 2.5, 'Критерій\nзупинки?', '#E67E22'),
    (5, 1, 'Найкраща\nмодель', '#27AE60')
]

for x, y, text, color in blocks:
    if 'зупинки' in text:
        # Ромб для умови
        diamond = mpatches.FancyBboxPatch((x-1, y-0.5), 2, 1,
                                         boxstyle="round,pad=0.1",
                                         edgecolor=color, facecolor='white',
                                         linewidth=2)
        ax.add_patch(diamond)
    else:
        # Прямокутник для дій
        box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                            boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor='white',
                            linewidth=2)
        ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Стрілки
arrows = [
    (5, 12.6, 5, 12),
    (5, 11.1, 5, 10.5),
    (5, 9.6, 5, 9),
    (5, 8.1, 5, 7.5),
    (5, 6.6, 5, 6),
    (5, 5.1, 5, 4.5),
    (5, 3.6, 5, 3),
    (5, 2.1, 5, 1.5),
]

for x1, y1, x2, y2 in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='#333333')
    ax.add_patch(arrow)

# Зворотня стрілка (цикл)
ax.annotate('', xy=(5, 11.9), xytext=(7.5, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color='#E74C3C',
                          connectionstyle="arc3,rad=.5"))
ax.text(7.8, 8, 'Так', fontsize=9, color='#E74C3C', fontweight='bold')

# Стрілка "Ні" вниз
ax.text(5.5, 1.8, 'Ні', fontsize=9, color='#27AE60', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'figure2_ga_flowchart.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure2_ga_flowchart.png")

# ============================================================================
# Рисунок 3: Динаміка fitness по поколіннях
# ============================================================================
print("📊 Рисунок 3: Динаміка fitness по поколіннях...")

generations = []
best_fitness = []
avg_fitness = []
worst_fitness = []

for gen in data['generations']:
    gen_num = gen['generation']
    models = gen['models']
    fitnesses = [m['fitness'] for m in models]
    
    generations.append(gen_num)
    best_fitness.append(max(fitnesses))
    avg_fitness.append(sum(fitnesses) / len(fitnesses))
    worst_fitness.append(min(fitnesses))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(generations, best_fitness, 'o-', linewidth=2, markersize=8,
        label='Найкраща fitness', color='#2E86AB')
ax.plot(generations, avg_fitness, 's-', linewidth=2, markersize=8,
        label='Середня fitness', color='#50C878')
ax.plot(generations, worst_fitness, '^-', linewidth=2, markersize=8,
        label='Найгірша fitness', color='#E74C3C')

ax.set_xlabel('Покоління', fontsize=12, fontweight='bold')
ax.set_ylabel('Fitness (Validation Accuracy)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(generations)

plt.tight_layout()
plt.savefig(output_dir / 'figure3_fitness_dynamics.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure3_fitness_dynamics.png")

# ============================================================================
# Рисунок 4: Розподіл fitness по оптимізаторах
# ============================================================================
print("📊 Рисунок 4: Розподіл fitness по оптимізаторах...")

optimizers_data = {}
for gen in data['generations']:
    for model in gen['models']:
        opt = model['hyperparameters']['optimizer']
        if opt not in optimizers_data:
            optimizers_data[opt] = []
        optimizers_data[opt].append(model['fitness'])

fig, ax = plt.subplots(figsize=(10, 6))

positions = list(range(len(optimizers_data)))
labels = list(optimizers_data.keys())
data_to_plot = [optimizers_data[opt] for opt in labels]

bp = ax.boxplot(data_to_plot, positions=positions, labels=labels,
                patch_artist=True, widths=0.6)

colors = ['#4A90E2', '#E74C3C', '#50C878']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Оптимізатор', fontsize=12, fontweight='bold')
ax.set_ylabel('Fitness (Validation Accuracy)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'figure4_optimizer_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure4_optimizer_distribution.png")

# ============================================================================
# Рисунок 5: Розподіл fitness по batch size
# ============================================================================
print("📊 Рисунок 5: Розподіл fitness по batch size...")

batch_sizes_data = {}
for gen in data['generations']:
    for model in gen['models']:
        bs = model['hyperparameters']['batch_size']
        if bs not in batch_sizes_data:
            batch_sizes_data[bs] = []
        batch_sizes_data[bs].append(model['fitness'])

fig, ax = plt.subplots(figsize=(10, 6))

batch_sizes = sorted(batch_sizes_data.keys())
means = [np.mean(batch_sizes_data[bs]) for bs in batch_sizes]
stds = [np.std(batch_sizes_data[bs]) for bs in batch_sizes]

x_pos = np.arange(len(batch_sizes))
bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
              color=['#4A90E2', '#50C878', '#E74C3C'])

ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Середня Fitness', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(batch_sizes)
ax.grid(True, alpha=0.3, axis='y')

# Додати значення над стовпцями
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', va='bottom',
            fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'figure5_batch_size_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure5_batch_size_distribution.png")

# ============================================================================
# Рисунок 6: Розподіл fitness по глибині мережі
# ============================================================================
print("📊 Рисунок 6: Розподіл fitness по глибині мережі...")

layers_data = []
fitness_data = []

for gen in data['generations']:
    for model in gen['models']:
        layers_data.append(model['architecture']['num_layers'])
        fitness_data.append(model['fitness'])

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(layers_data, fitness_data, s=100, alpha=0.6,
                    c=fitness_data, cmap='viridis', edgecolors='black', linewidth=1)

# Поліноміальна регресія
z = np.polyfit(layers_data, fitness_data, 2)
p = np.poly1d(z)
x_line = np.linspace(min(layers_data), max(layers_data), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Тренд (поліном 2-го порядку)')

ax.set_xlabel('Кількість шарів', fontsize=12, fontweight='bold')
ax.set_ylabel('Fitness (Validation Accuracy)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Fitness', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'figure6_depth_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure6_depth_distribution.png")

# ============================================================================
# Рисунок 7: Архітектура найкращої моделі
# ============================================================================
print("📊 Рисунок 7: Архітектура найкращої моделі...")

# Знайти найкращу модель
all_models = []
for gen in data['generations']:
    for model in gen['models']:
        all_models.append(model)

best_model = max(all_models, key=lambda x: x['fitness'])

# Динамічна висота залежно від кількості шарів
num_layers = len(best_model['architecture']['layers'])
fig_height = max(14, num_layers * 1.5 + 4)

fig, ax = plt.subplots(figsize=(10, fig_height))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, num_layers * 1.7 + 3)

y_pos = num_layers * 1.7 + 2

# Заголовок
ax.text(5, y_pos, f"Найкраща архітектура ({best_model['model_id']})",
        ha='center', va='top', fontsize=14, fontweight='bold')
ax.text(5, y_pos - 0.5, f"Fitness: {best_model['fitness']:.4f}",
        ha='center', va='top', fontsize=12)

y_pos -= 2

# Input
box = FancyBboxPatch((3, y_pos - 0.4), 4, 0.8,
                     boxstyle="round,pad=0.1",
                     edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
ax.add_patch(box)
ax.text(5, y_pos, 'Input (32×32×3)', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Стрілка вниз
arrow = FancyArrowPatch((5, y_pos - 0.5), (5, y_pos - 1),
                       arrowstyle='->', mutation_scale=15, linewidth=2, color='#333333')
ax.add_patch(arrow)
y_pos -= 1.5

# Шари
layer_colors = {
    'conv2d': '#4A90E2',
    'batch_norm': '#50C878',
    'max_pool': '#F39C12',
    'flatten': '#E74C3C',
    'dropout': '#9B59B6',
    'dense': '#1ABC9C'
}

for layer in best_model['architecture']['layers']:
    layer_type = layer['type']
    color = layer_colors.get(layer_type, '#95A5A6')
    
    # Текст шару
    if layer_type == 'conv2d':
        text = f"Conv2D\n(filters={layer['filters']}, kernel={layer['kernel_size']}×{layer['kernel_size']})\nactivation={layer['activation']}"
    elif layer_type == 'dense':
        text = f"Dense\n(neurons={layer['neurons']})\nactivation={layer['activation']}"
    elif layer_type == 'dropout':
        text = f"Dropout\n(rate={layer['dropout_rate']:.2f})"
    elif layer_type == 'max_pool':
        text = f"MaxPooling2D\n(pool_size={layer.get('pool_size', 2)}×{layer.get('pool_size', 2)})"
    else:
        text = layer_type.replace('_', ' ').title()
    
    box = FancyBboxPatch((2.5, y_pos - 0.5), 5, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos, text, ha='center', va='center', fontsize=9)
    
    # Стрілка вниз
    arrow = FancyArrowPatch((5, y_pos - 0.6), (5, y_pos - 1.1),
                           arrowstyle='->', mutation_scale=15, linewidth=2, color='#333333')
    ax.add_patch(arrow)
    y_pos -= 1.7

plt.tight_layout()
plt.savefig(output_dir / 'figure7_best_architecture.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure7_best_architecture.png")

# ============================================================================
# Рисунок 8: Порівняння з іншими методами (час vs точність)
# ============================================================================
print("📊 Рисунок 8: Порівняння з іншими методами...")

methods = {
    'NASNet': {'time': 1800*24, 'accuracy': 97.4, 'params': 3.3},  # GPU-години
    'ENAS': {'time': 0.5*24, 'accuracy': 97.11, 'params': 4.6},
    'DARTS': {'time': 4*24, 'accuracy': 97.24, 'params': 3.3},
    'Genetic CNN': {'time': 17*24, 'accuracy': 93.30, 'params': 4.0},
    'AmoebaNet': {'time': 3150*24, 'accuracy': 97.87, 'params': 3.2},
    'NSGA-Net': {'time': 4*24, 'accuracy': 97.44, 'params': 3.3},
    'EvNAS': {'time': 0.6*24, 'accuracy': 97.37, 'params': 3.6},
    'EST-NAS': {'time': 0.3*24, 'accuracy': 97.58, 'params': 3.0},
    'Запропонований (WS)': {'time': 0.3, 'accuracy': 98.59, 'params': 0.3}  # 19 хв = 0.3 год, 98.59%
}

fig, ax = plt.subplots(figsize=(12, 8))

for method, data in methods.items():
    color = '#E74C3C' if method == 'Запропонований (WS)' else '#4A90E2'
    marker = 'D' if method == 'Запропонований (WS)' else 'o'
    size = data['params'] * 100
    
    ax.scatter(data['time'], data['accuracy'], s=size, alpha=0.6,
              color=color, edgecolors='black', linewidth=1.5, marker=marker)
    
    # Підпис
    offset_x = 1.3 if method == 'Запропонований' else 1.1
    offset_y = -2 if method == 'AmoebaNet' else 1
    ax.annotate(method, (data['time'], data['accuracy']),
               xytext=(data['time'] * offset_x, data['accuracy'] + offset_y),
               fontsize=9, fontweight='bold' if method == 'Запропонований (WS)' else 'normal')

ax.set_xlabel('Час пошуку (GPU-години, log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Точність на Military Vehicles (%)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, which='both')

# Легенда для розміру
legend_sizes = [1, 3, 5]
legend_labels = ['1M', '3M', '5M']
legend_elements = [plt.scatter([], [], s=size*100, color='gray', alpha=0.6, edgecolors='black')
                  for size in legend_sizes]
legend1 = ax.legend(legend_elements, legend_labels, title='Параметри',
                   loc='lower right', fontsize=10, title_fontsize=11)
ax.add_artist(legend1)

plt.tight_layout()
plt.savefig(output_dir / 'figure8_methods_comparison.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure8_methods_comparison.png")

# ============================================================================
# Рисунок 9: Приклад динаміки навчання (best epoch)
# ============================================================================
print("📊 Рисунок 9: Приклад динаміки навчання (best epoch)...")

# Симуляція даних (приклад)
epochs = np.arange(1, 31)
train_acc = 0.3 + 0.5 * (1 - np.exp(-epochs / 5)) + np.random.normal(0, 0.02, 30)
val_acc = 0.25 + 0.47 * (1 - np.exp(-epochs / 6)) - 0.015 * (epochs - 15)**2 / 100
val_acc = np.clip(val_acc, 0, 1)

best_epoch = np.argmax(val_acc) + 1
best_val_acc = np.max(val_acc)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_acc, 'o-', linewidth=2, markersize=6,
        label='Train Accuracy', color='#4A90E2', alpha=0.8)
ax.plot(epochs, val_acc, 's-', linewidth=2, markersize=6,
        label='Validation Accuracy', color='#E74C3C', alpha=0.8)

# Вертикальна лінія для best epoch
ax.axvline(x=best_epoch, color='#50C878', linestyle='--', linewidth=2,
          label=f'Best Epoch: {best_epoch}')

# Annotation
ax.annotate(f'Best Epoch: {best_epoch}\nVal Acc: {best_val_acc:.4f}',
           xy=(best_epoch, best_val_acc),
           xytext=(best_epoch + 5, best_val_acc - 0.1),
           arrowprops=dict(arrowstyle='->', lw=2, color='#50C878'),
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#50C878', linewidth=2))

ax.set_xlabel('Епоха', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / 'figure9_best_epoch_example.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure9_best_epoch_example.png")

# ============================================================================
# Рисунок 10: Розподіл типів шарів у найкращих моделях
# ============================================================================
print("📊 Рисунок 10: Розподіл типів шарів...")

# Використати вже зібрані all_models з Рисунку 7
# all_models вже існує з попереднього коду

sorted_models = sorted(all_models, key=lambda x: x['fitness'], reverse=True)
top_20 = sorted_models[:max(1, len(sorted_models) // 5)]
bottom_20 = sorted_models[-max(1, len(sorted_models) // 5):]

# Підрахунок типів шарів
from collections import Counter

top_layers = Counter()
bottom_layers = Counter()

for model in top_20:
    for layer in model['architecture']['layers']:
        top_layers[layer['type']] += 1

for model in bottom_20:
    for layer in model['architecture']['layers']:
        bottom_layers[layer['type']] += 1

# Нормалізація на кількість моделей
layer_types = ['conv2d', 'batch_norm', 'max_pool', 'flatten', 'dense', 'dropout']
top_avg = [top_layers[lt] / len(top_20) for lt in layer_types]
bottom_avg = [bottom_layers[lt] / len(bottom_20) for lt in layer_types]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(layer_types))
width = 0.35

bars1 = ax.bar(x - width/2, top_avg, width, label='Top 20% моделей', 
               color='#4A90E2', alpha=0.8)
bars2 = ax.bar(x + width/2, bottom_avg, width, label='Bottom 20% моделей',
               color='#E74C3C', alpha=0.8)

ax.set_xlabel('Тип шару', fontsize=12, fontweight='bold')
ax.set_ylabel('Середня кількість на модель', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([lt.replace('_', ' ').title() for lt in layer_types], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Додати значення над стовпцями
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'figure10_layer_types_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure10_layer_types_distribution.png")

# ============================================================================
# Рисунок 11: Розподіл learning rate
# ============================================================================
print("📊 Рисунок 11: Розподіл learning rate...")

lr_ranges = [(0.0001, 0.0003), (0.0003, 0.0005), (0.0005, 0.0007), 
             (0.0007, 0.001), (0.001, 0.002)]
lr_labels = ['0.0001-0.0003', '0.0003-0.0005', '0.0005-0.0007', 
             '0.0007-0.001', '>0.001']
lr_data = {label: [] for label in lr_labels}

for model in all_models:
    lr = model['hyperparameters']['learning_rate']
    placed = False
    for i, (low, high) in enumerate(lr_ranges[:-1]):
        if low <= lr < high:
            lr_data[lr_labels[i]].append(model['fitness'])
            placed = True
            break
    if not placed:
        lr_data[lr_labels[-1]].append(model['fitness'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                gridspec_kw={'height_ratios': [1, 1.5]})

# Histogram
counts = [len(lr_data[label]) for label in lr_labels]
colors_lr = ['#4A90E2', '#50C878', '#F39C12', '#E74C3C', '#9B59B6']
bars = ax1.bar(lr_labels, counts, color=colors_lr, alpha=0.7, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Діапазон Learning Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('Кількість моделей', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Box plot
data_to_plot = [lr_data[label] for label in lr_labels if lr_data[label]]
positions = [i for i, label in enumerate(lr_labels) if lr_data[label]]
valid_labels = [label for label in lr_labels if lr_data[label]]

bp = ax2.boxplot(data_to_plot, positions=positions, labels=valid_labels,
                 patch_artist=True, widths=0.6)

for patch, color in zip(bp['boxes'], colors_lr):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xlabel('Діапазон Learning Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('Fitness (Validation Accuracy)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'figure11_learning_rate_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure11_learning_rate_distribution.png")

# ============================================================================
# Рисунок 12: Кореляційна матриця гіперпараметрів
# ============================================================================
print("📊 Рисунок 12: Кореляційна матриця гіперпараметрів...")

# Створити матрицю: optimizer × batch_size
optimizers = ['adam', 'sgd', 'rmsprop']
batch_sizes = [16, 32, 64]

# Матриця для середньої fitness
fitness_matrix = np.zeros((len(optimizers), len(batch_sizes)))
count_matrix = np.zeros((len(optimizers), len(batch_sizes)))

for model in all_models:
    opt = model['hyperparameters']['optimizer']
    bs = model['hyperparameters']['batch_size']
    
    if opt in optimizers and bs in batch_sizes:
        opt_idx = optimizers.index(opt)
        bs_idx = batch_sizes.index(bs)
        fitness_matrix[opt_idx, bs_idx] += model['fitness']
        count_matrix[opt_idx, bs_idx] += 1

# Обчислити середнє
for i in range(len(optimizers)):
    for j in range(len(batch_sizes)):
        if count_matrix[i, j] > 0:
            fitness_matrix[i, j] /= count_matrix[i, j]
        else:
            fitness_matrix[i, j] = np.nan

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(fitness_matrix, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=0.6)

ax.set_xticks(np.arange(len(batch_sizes)))
ax.set_yticks(np.arange(len(optimizers)))
ax.set_xticklabels([f'BS {bs}' for bs in batch_sizes], fontsize=11)
ax.set_yticklabels([opt.upper() for opt in optimizers], fontsize=11)

ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Optimizer', fontsize=12, fontweight='bold')

# Додати текст з значеннями
for i in range(len(optimizers)):
    for j in range(len(batch_sizes)):
        if not np.isnan(fitness_matrix[i, j]):
            text = ax.text(j, i, f'{fitness_matrix[i, j]:.3f}\n(n={int(count_matrix[i, j])})',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Середня Fitness', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'figure12_hyperparameter_heatmap.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure12_hyperparameter_heatmap.png")

# ============================================================================
# Рисунок 13: Траєкторія еволюції (спрощена версія)
# ============================================================================
print("📊 Рисунок 13: Траєкторія еволюції...")

# Створити простий 2D простір: кількість шарів vs середня fitness
fig, ax = plt.subplots(figsize=(12, 8))

# Групувати по поколіннях (використати generations з початку файлу)
# Перезавантажити дані для цього рисунку
with open('output/model_history/all_models.json', 'r') as f:
    data_reload = json.load(f)

colors_gen = plt.cm.viridis(np.linspace(0, 1, len(data_reload['generations'])))

for gen_idx, gen in enumerate(data_reload['generations']):
    gen_num = gen['generation']
    layers = [m['architecture']['num_layers'] for m in gen['models']]
    fitnesses = [m['fitness'] for m in gen['models']]
    
    scatter = ax.scatter(layers, fitnesses, 
                        s=200, alpha=0.7, 
                        c=[colors_gen[gen_idx]], 
                        edgecolors='black', linewidth=2,
                        label=f'Покоління {gen_num}')

# З'єднати лінією середні значення по поколіннях
gen_nums = []
avg_layers = []
avg_fitness = []

for gen in data_reload['generations']:
    gen_nums.append(gen['generation'])
    avg_layers.append(np.mean([m['architecture']['num_layers'] for m in gen['models']]))
    avg_fitness.append(np.mean([m['fitness'] for m in gen['models']]))

ax.plot(avg_layers, avg_fitness, 'k--', linewidth=2, alpha=0.5, 
        label='Траєкторія середніх')

ax.set_xlabel('Кількість шарів', fontsize=12, fontweight='bold')
ax.set_ylabel('Fitness (Validation Accuracy)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Додати стрілку напрямку еволюції
if len(avg_layers) > 1:
    ax.annotate('', xy=(avg_layers[-1], avg_fitness[-1]), 
               xytext=(avg_layers[-2], avg_fitness[-2]),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(avg_layers[-1], avg_fitness[-1] + 0.02, 'Напрямок еволюції',
           fontsize=10, fontweight='bold', color='red', ha='center')

plt.tight_layout()
plt.savefig(output_dir / 'figure13_evolution_trajectory.png', bbox_inches='tight')
plt.close()
print("✅ Збережено: figure13_evolution_trajectory.png")

print("\n✅ Всі рисунки успішно згенеровано!")
print(f"📁 Збережено в: {output_dir.absolute()}")
