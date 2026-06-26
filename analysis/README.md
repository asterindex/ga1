# 📊 Analysis Scripts

Ця папка містить скрипти для аналізу результатів генетичного алгоритму.

## 🚀 Швидкий старт

Після завершення еволюції запусти основний аналіз:

```bash
cd /path/to/ga2
python3 analysis/analyze_models.py
```

**Вимога:** Має існувати файл `output/model_history/all_models.json`

---

## 📁 Структура

```
analysis/
├── README.md              # Цей файл
├── analyze_models.py      # 🔥 Основний скрипт аналізу генів
├── (TODO) correlation_matrix.py
├── (TODO) evolution_trajectory.py
└── (TODO) visualizations.py
```

---

## 🧬 analyze_models.py

### Що аналізує:

1. **Optimizer Importance** - порівняння Adam/SGD/RMSprop
2. **Learning Rate** - оптимальні діапазони LR
3. **Batch Size** - вплив розміру батчу
4. **Architecture Depth** - оптимальна кількість шарів
5. **Layer Types** - які типи шарів найкращі
6. **Best Model Details** - детальний розбір топової моделі

### Використання:

```bash
# Базовий запуск
python3 analysis/analyze_models.py

# З Python
from analysis.analyze_models import load_model_history, analyze_gene_importance

history = load_model_history('output/model_history/all_models.json')
analyze_gene_importance(history)
```

### Приклад виводу:

```
🧬 GENE IMPORTANCE ANALYSIS
======================================================================

📊 Total models analyzed: 40
   ├─ Top 20%: 8 models
   └─ Bottom 20%: 8 models

----------------------------------------------------------------------
1️⃣  OPTIMIZER IMPORTANCE
----------------------------------------------------------------------

ADAM:
   ├─ Count: 15
   ├─ Avg fitness: 0.5234
   ├─ Max fitness: 0.6795
   └─ Std dev: 0.0456

RMSPROP:
   ├─ Count: 13
   ├─ Avg fitness: 0.4876
   ├─ Max fitness: 0.6123
   └─ Std dev: 0.0512

...
```

---

## 💡 Додавання власних скриптів

Створи новий файл в `analysis/`:

```python
# analysis/my_custom_analysis.py

import json
import sys
import os

# Додати батьківську директорію в PATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.analyze_models import load_model_history


def my_analysis(history):
    """Твій власний аналіз"""
    print("🔬 My Custom Analysis")
    # ... твій код ...


if __name__ == "__main__":
    history = load_model_history()
    if history:
        my_analysis(history)
```

Запуск:

```bash
python3 analysis/my_custom_analysis.py
```

---

## 📚 Корисні ресурси

- [ANALYSIS_GUIDE.md](../ANALYSIS_GUIDE.md) - Повний гайд по аналізу
- [model_history JSON structure](../ANALYSIS_GUIDE.md#структура-даних)

---

**Успіхів з аналізом! 📊🧬**
