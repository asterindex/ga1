#!/bin/bash
# Порівняльний експеримент v2: більша популяція, ширший простір пошуку
# Baseline (GA без warm start) vs GA з ламаркіанським warm start
# Популяція: 12, Поколінь: 15, Seed: 7

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

POP=12
GENS=15
SEED=7
IMAGE="ga2:latest"

echo "============================================================"
echo " Порівняльний експеримент v2"
echo " Популяція: $POP | Поколінь: $GENS | Seed: $SEED"
echo " MAX_FILTERS=128, MAX_LAYERS=6, KERNEL=[3,5]"
echo "============================================================"

# Зупинити контейнери якщо є
docker rm -f nas-baseline nas-warmstart 2>/dev/null || true

# Побудувати образ
echo "[$(date)] Будуємо Docker образ..."
docker build -t $IMAGE . --quiet

# --- BASELINE (без warm start) ---
echo ""
echo "[$(date)] Запускаємо BASELINE (без warm start)..."
mkdir -p output_baseline_v2

docker run --rm --gpus all \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/output_baseline_v2:/app/output" \
    --name nas-baseline \
    $IMAGE \
    python main.py --mode full \
        --generations $GENS \
        --population $POP \
        --seed $SEED \
        --no-warm-start \
    2>&1 | tee output_baseline_v2/run.log

echo "[$(date)] BASELINE завершено."

# Зберегти результати
cp output_baseline_v2/model_history/all_models.json output/baseline_v2_results.json 2>/dev/null || true

# --- WARM START ---
echo ""
echo "[$(date)] Запускаємо WARM START..."
mkdir -p output_warmstart_v2

docker run --rm --gpus all \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/output_warmstart_v2:/app/output" \
    --name nas-warmstart \
    $IMAGE \
    python main.py --mode full \
        --generations $GENS \
        --population $POP \
        --seed $SEED \
    2>&1 | tee output_warmstart_v2/run.log

echo "[$(date)] WARM START завершено."

# Зберегти результати
cp output_warmstart_v2/model_history/all_models.json output/warmstart_v2_results.json 2>/dev/null || true

echo ""
echo "============================================================"
echo " ГОТОВО! Результати:"
echo "   output/baseline_v2_results.json"
echo "   output/warmstart_v2_results.json"
echo "============================================================"
