# 🚀 ЗАПУСК НА СЕРВЕРІ

## Налаштування сервера

### 1. Перевірка GPU
```bash
# Перевірити NVIDIA GPU
nvidia-smi

# Перевірити Docker GPU підтримку
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 2. Build Docker image
```bash
# На сервері
cd /path/to/ga2
docker build -t ga2:latest .
```

---

## Запуск експериментів

### Швидкий тест (5 хв)
```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  ga2:latest \
  python3 main.py --mode fast --generations 3
```

### Середній експеримент (30 хв)
```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  ga2:latest \
  python3 main.py --mode fast --generations 5
```

### Повний експеримент для статті (2-4 год)
```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  ga2:latest \
  python3 main.py --mode full --generations 15
```

### Запуск в фоні (detached)
```bash
docker run -d --name nas-experiment \
  --gpus all \
  -v $(pwd)/output:/app/output \
  ga2:latest \
  python3 main.py --mode full --generations 20

# Моніторинг
docker logs -f nas-experiment

# Зупинити
docker stop nas-experiment
```

---

## Docker Compose (рекомендовано)

### Запуск
```bash
# Default
docker-compose up

# Custom parameters
docker-compose run --rm ga2 \
  python3 main.py --mode full --generations 15

# В фоні
docker-compose up -d

# Логи
docker-compose logs -f
```

### Зупинка
```bash
docker-compose down
```

---

## Оптимізація для GPU

### TensorFlow налаштування
```bash
# Автоматично включено в Dockerfile:
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Динамічна пам'ять
export TF_CPP_MIN_LOG_LEVEL=2          # Менше логів
```

### Моніторинг GPU
```bash
# Під час роботи
watch -n 1 nvidia-smi

# Або
docker exec nas-experiment nvidia-smi
```

---

## Результати

### Структура output/
```
output/
├── best_model.json           # Найкраща архітектура
├── evolution.png             # Графік еволюції
├── generation_N.json         # Дані кожного покоління
└── history.json              # Повна історія еволюції
```

### Копіювання результатів з сервера
```bash
# З сервера на локальну машину
scp -r user@server:/path/to/ga2/output ./output_server

# Або через Docker volume
docker cp nas-experiment:/app/output ./output_server
```

---

## Рекомендовані експерименти

### Експеримент 1: Quick test
```bash
docker-compose run --rm ga2 \
  python3 main.py --mode fast --generations 5
```
**Час:** ~5 хвилин  
**Результат:** Швидка перевірка роботи системи

### Експеримент 2: Full run
```bash
docker-compose run --rm ga2 \
  python3 main.py --mode full --generations 15
```
**Час:** ~1-2 години  
**Результат:** Повний експеримент для статті

### Експеримент 3: Extended search
```bash
docker-compose run --rm ga2 \
  python3 main.py --mode full --generations 20
```
**Час:** ~2-3 години  
**Результат:** Екстенсивний пошук архітектур

---

## Troubleshooting

### GPU не виявляється
```bash
# Перевірити nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of memory
```bash
# Зменшити batch_size в config.py
# Або обмежити GPU пам'ять:
docker run --gpus all --memory=16g ...
```

### Повільно тренується
```bash
# Перевірити GPU використання
nvidia-smi

# Перевірити TensorFlow бачить GPU
docker run --rm --gpus all ga2:latest \
  python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## Продуктивність

### Очікувані покращення на GPU:
- **CPU (M1 Mac)**: 1 model × 5 epochs ≈ 30 сек
- **GPU (CUDA)**: 1 model × 5 epochs ≈ **3-5 сек** (6-10× швидше)

### Приклад timeline:
```
Full mode (15 generations × 8 models × 10 epochs):
  CPU: ~2 години
  GPU: ~10-15 хвилин ⚡
```

---

## Автоматизація

### Скрипт для серії експериментів
```bash
#!/bin/bash
# run_experiments.sh

experiments=(
  "fast 5 50"
  "fast 10 100" 
  "full 10 200"
  "full 15 300"
)

for exp in "${experiments[@]}"; do
  read -r mode p1 p2 <<< "$exp"
  echo "🚀 Running: mode=$mode, phase1=$p1, phase2=$p2"
  
  docker-compose run --rm ga2 \
    python3 main.py --mode $mode --phase-1-generations $p1 --phase-2-generations $p2
  
  # Backup results
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp -r output "results_${mode}_p1${p1}_p2${p2}_${timestamp}"
done
```

### Запуск
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```
