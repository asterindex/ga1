#!/bin/bash
# Run 3 parallel experiments on server

SERVER_USER="anatoly_kot"
SERVER_HOST="193.200.64.103"

echo "🚀 Запуск 3 паралельних експериментів на сервері"
echo ""

ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
cd ~/ga2

# Stop current experiment
docker stop ga2 2>/dev/null || true
docker rm ga2 2>/dev/null || true

# Create directories for outputs
mkdir -p output_exp1 output_exp2 output_exp3

# Run 3 experiments in parallel
echo "Starting experiment 1..."
docker run -d --name nas-exp1 --gpus all \
  -v ~/ga2/output_exp1:/app/output \
  ga2:latest \
  python3 main.py --mode full

sleep 2

echo "Starting experiment 2..."
docker run -d --name nas-exp2 --gpus all \
  -v ~/ga2/output_exp2:/app/output \
  ga2:latest \
  python3 main.py --mode full

sleep 2

echo "Starting experiment 3..."
docker run -d --name nas-exp3 --gpus all \
  -v ~/ga2/output_exp3:/app/output \
  ga2:latest \
  python3 main.py --mode full

echo ""
echo "✅ 3 experiments started!"
echo ""
docker ps | grep nas-exp

ENDSSH

echo ""
echo "📊 Моніторинг:"
echo "  ssh $SERVER_USER@$SERVER_HOST 'docker logs -f nas-exp1'"
echo "  ssh $SERVER_USER@$SERVER_HOST 'nvidia-smi'"
echo ""
echo "📥 Скачати результати:"
echo "  scp -r $SERVER_USER@$SERVER_HOST:~/ga2/output_exp* ./"
