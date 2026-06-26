# КОМАНДИ ДЛЯ СЕРВЕРА

## Підключення
```bash
ssh anatoly_kot@193.200.64.103 -i priv.key.pem
```

## Копіювання проекту на сервер

### З локальної машини:
```bash
# Якщо priv.key.pem в поточній директорії
scp -i priv.key.pem -r ./ anatoly_kot@193.200.64.103:/home/anatoly_kot/ga2/

# Або rsync (рекомендовано - швидше)
rsync -avz -e "ssh -i priv.key.pem" \
  --exclude 'output' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'venv' \
  ./ anatoly_kot@193.200.64.103:/home/anatoly_kot/ga2/
```

## На сервері

### 1. Перевірка GPU
```bash
nvidia-smi

# Має показати GPU + CUDA version
```

### 2. Build Docker image
```bash
cd ~/ga2
docker build -t ga2:latest .

# Перевірка GPU в Docker
docker run --rm --gpus all ga2:latest \
  python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 3. Швидкий тест (1 хв)
```bash
docker-compose run --rm ga2 \
  python3 main.py --mode fast --phase-1-generations 1 --phase-2-generations 5
```

### 4. Експеримент для статті (30-60 хв)
```bash
# Запуск в фоні
docker-compose run -d --name nas-experiment ga2 \
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
  ga2:latest \
  python3 main.py --mode fast --phase-1-generations 5 --phase-2-generations 50

# Experiment 2  
docker run -d --name exp2 --gpus all \
  -v $(pwd)/output_exp2:/app/output \
  ga2:latest \
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
scp -i priv.key.pem -r anatoly_kot@193.200.64.103:/home/anatoly_kot/ga2/output ./output_server

# Або rsync
rsync -avz -e "ssh -i priv.key.pem" \
  anatoly_kot@193.200.64.103:/home/anatoly_kot/ga2/output/ \
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
docker ps | grep ga2 | awk '{print $1}' | xargs docker stop
```
