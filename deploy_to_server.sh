#!/bin/bash
# Deploy to server and run experiments

set -e

# Configuration
SERVER_USER="anatoly_kot"
SERVER_HOST="193.200.64.60"
SSH_KEY="priv.key.pem"
SERVER_PATH="/home/$SERVER_USER/genetic_nas"

echo "🚀 Deploying to server: $SERVER_USER@$SERVER_HOST"

# 1. Copy project to server
echo "📦 Copying files..."
rsync -avz -e "ssh -i $SSH_KEY" \
  --exclude 'output' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'venv' \
  ./ $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

# 2. Build Docker image on server
echo "🔨 Building Docker image..."
ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker build -t genetic-nas:latest ."

# 3. Run experiment
echo "🧬 Starting experiment..."
ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST "cd $SERVER_PATH && docker-compose up -d"

echo "✅ Deployment complete!"
echo ""
echo "📊 Monitoring commands:"
echo "  ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST 'docker logs -f genetic-nas'"
echo "  ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST 'docker exec genetic-nas nvidia-smi'"
echo ""
echo "📥 Download results:"
echo "  scp -i $SSH_KEY -r $SERVER_USER@$SERVER_HOST:$SERVER_PATH/output ./output_server"
