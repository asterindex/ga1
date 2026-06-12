# КОМАНДИ ДЛЯ СЕРВЕРА

## Підключення
```bash
ssh anatoly_kot@193.200.64.60 -i priv.key.pem
```

## Копіювання проекту на сервер

### З локальної машини:
```bash
# Якщо priv.key.pem в поточній директорії
scp -i priv.key.pem -r ./ anatoly_kot@193.200.64.60:/home/anatoly_kot/genetic_nas/

# Або rsync (рекомендовано - швидше)
rsync -avz -e "ssh -i priv.key.pem" \
  --exclude 'output' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'venv' \
  ./ anatoly_kot@193.200.64.60:/home/anatoly_kot/genetic_nas/
```

## На сервері

### 1. Перевірка GPU
```bash
nvidia-smi

# Має показати GPU + CUDA version
```

### 2. Build Docker image
```bash
cd ~/genetic_nas
docker build -t genetic-nas:latest .

# Перевірка GPU в Docker
docker run --rm --gpus all genetic-nas:latest \
  python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 3. Швидкий тест (1 хв)
```bash
docker-compose run --rm genetic-nas \
  python3 main.py --mode fast --phase-1-generations 1 --phase-2-generations 5
```

### 4. Експеримент для статті (30-60 хв)
```bash
# Запуск в фоні
docker-compose run -d --name nas-experiment genetic-nas \
  python3 main.py --mode full --phase-1-generations 15 --phase-2-generations 200

# Моніторинг
docker logs -f nas-experiment

# GPU usage
watch -n 2 nvidia-smi
```

### 5. Декілька експериментів паралельно
```bash
# Experiment 1
docker run -d --name exp1 --gpus all \
  -v $(pwd)/output_exp1:/app/output \
  genetic-nas:latest \
  python3 main.py --mode fast --phase-1-generations 5 --phase-2-generations 50

# Experiment 2  
docker run -d --name exp2 --gpus all \
  -v $(pwd)/output_exp2:/app/output \
  genetic-nas:latest \
  python3 main.py --mode full --phase-1-generations 10 --phase-2-generations 100

# Моніторинг обидвох
docker logs -f exp1 &
docker logs -f exp2
```

### 6. Автоматична серія експериментів
```bash
./run_all_experiments.sh

# Це запустить 4 експерименти і збереже результати в experiments_TIMESTAMP/
```

## Скачати результати

### З сервера на локальну машину:
```bash
# На локальній машині
scp -i priv.key.pem -r anatoly_kot@193.200.64.60:/home/anatoly_kot/genetic_nas/output ./output_server

# Або rsync
rsync -avz -e "ssh -i priv.key.pem" \
  anatoly_kot@193.200.64.60:/home/anatoly_kot/genetic_nas/output/ \
  ./output_server/
```

## Troubleshooting

### Docker не бачить GPU
```bash
# Встановити nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Перевірка
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory
```bash
# Перевірити доступну пам'ять GPU
nvidia-smi

# Якщо мало - зменшити batch_size або population
# В src/config.py змінити:
POPULATION_SIZE_FULL = 6  # Замість 8
```

### Зупинити всі експерименти
```bash
docker ps | grep genetic-nas | awk '{print $1}' | xargs docker stop
```
