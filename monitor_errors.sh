#!/bin/bash

# Моніторинг помилок експерименту на сервері

SERVER="anatoly_kot@193.200.64.103"
CONTAINER="ga2"

echo "🔍 Моніторинг помилок експерименту..."
echo "======================================"
echo "Сервер: $SERVER"
echo "Контейнер: $CONTAINER"
echo "Інтервал: 60 секунд"
echo "Натисніть Ctrl+C для зупинки"
echo "======================================"
echo ""

while true; do
    TIMESTAMP=$(date "+%H:%M:%S")
    
    # Перевірка чи контейнер працює
    STATUS=$(ssh $SERVER "docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null)
    
    if [ -z "$STATUS" ]; then
        echo "[$TIMESTAMP] ❌ КОНТЕЙНЕР НЕ ПРАЦЮЄ!"
        sleep 60
        continue
    fi
    
    # Підрахунок помилок за останню хвилину
    ERROR_COUNT=$(ssh $SERVER "docker logs --since 60s $CONTAINER 2>&1 | grep -iE '(error|exception|failed|помилка)' | wc -l" 2>/dev/null | tr -d ' ')
    
    # GPU статус
    GPU_INFO=$(ssh $SERVER "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader" 2>/dev/null)
    
    # Останній прогрес
    LAST_LOG=$(ssh $SERVER "docker logs --tail 3 $CONTAINER 2>&1 | grep -E '(Модель|покоління|Epoch|accuracy)' | tail -1" 2>/dev/null)
    
    # Вивід
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "[$TIMESTAMP] ⚠️  ПОМИЛОК: $ERROR_COUNT | GPU: $GPU_INFO"
        echo "           Останні помилки:"
        ssh $SERVER "docker logs --since 60s $CONTAINER 2>&1 | grep -iE '(error|exception|failed|помилка)' | tail -3" 2>/dev/null | sed 's/^/           /'
    else
        echo "[$TIMESTAMP] ✅ Без помилок | GPU: $GPU_INFO"
    fi
    
    if [ ! -z "$LAST_LOG" ]; then
        echo "           $LAST_LOG"
    fi
    
    echo ""
    sleep 60
done
