# 🚀 ШВИДКИЙ СТАРТ НА СЕРВЕРІ

## 1. Підготовка сервера

```bash
# Перевірка GPU
nvidia-smi

# Перевірка Docker + GPU
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## 2. Деплой проекту

### Варіант A: Автоматично
```bash
# На локальній машині (змінити SERVER_USER та SERVER_HOST в скрипті)
./deploy_to_server.sh
```

### Варіант B: Вручну
```bash
# 1. Скопіювати на сервер
scp -r genetic_nas/ user@server:/home/user/

# 2. На сервері
cd ~/genetic_nas
docker build -t genetic-nas:latest .
```

## 3. Запуск експериментів

### Тест (2 хв)
```bash
docker-compose run --rm genetic-nas \
  python3 main.py --mode fast --generations 3
```

### Для статті (1-2 год на GPU)
```bash
# 15 generations
docker-compose run -d --name nas-main genetic-nas \
  python3 main.py --mode full --generations 15

# Моніторинг
docker logs -f nas-main
```

### Серія експериментів (автоматично)
```bash
./run_all_experiments.sh
```

## 4. Моніторинг

```bash
# Логи
docker logs -f genetic-nas

# GPU usage
watch -n 1 nvidia-smi

# Або обидва
tmux new-session \; \
  split-window -h \; \
  send-keys 'docker logs -f genetic-nas' C-m \; \
  select-pane -t 0 \; \
  send-keys 'watch -n 1 nvidia-smi' C-m
```

## 5. Результати

```bash
# Перевірка
ls -lh output/

# Скачати на локальну машину
scp -r user@server:/home/user/genetic_nas/output ./output_server

# Або через Docker
docker cp nas-main:/app/output ./output_server
```

## 📊 Очікувані часи на GPU

| Config | Generations | GPU Time |
|--------|-------------|----------|
| Test | 3 | ~2 хв |
| Fast | 5 | ~5 хв |
| Medium | 10 | ~15 хв |
| Full | 15 | ~30-60 хв |
| Large | 20 | ~1-2 год |

**На CPU (без GPU) час × 10-20!**
