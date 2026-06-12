#!/bin/bash
# Automatic deploy and run experiments on server

set -e

# Configuration
SERVER_USER="anatoly_kot"
SERVER_HOST="193.200.64.60"
SSH_KEY="priv.key.pem"
SERVER_PATH="/home/$SERVER_USER/genetic_nas"

# Check for HF_TOKEN (optional but recommended)
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN not set"
    echo "💡 For Hugging Face datasets, create token at: https://huggingface.co/settings/tokens"
    echo "💡 Then run: export HF_TOKEN=your_token_here"
    echo ""
    echo "Continuing without token (may fail if dataset requires auth)..."
    echo ""
    HF_TOKEN_ARG=""
else
    echo "✅ HF_TOKEN found, will pass to container"
    HF_TOKEN_ARG="-e HF_TOKEN=$HF_TOKEN"
fi

echo "🚀 Deploying to datacenter: $SERVER_USER@$SERVER_HOST"
echo ""

# 1. Copy project to server
echo "📦 Step 1/4: Copying files to server..."
rsync -avz \
  --exclude 'output*' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '*.png' \
  ./ $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

echo "✅ Files copied"
echo ""

# 2. Build Docker image on server
echo "🔨 Step 2/4: Building Docker image on server..."
ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
cd ~/genetic_nas
echo "Building Docker image..."
docker build -t genetic-nas:latest . 2>&1 | tail -n 30
echo "Docker image built successfully!"
ENDSSH

echo "✅ Docker image built"
echo ""

# 3. GPU check
echo "🎮 Step 3/4: Checking GPU..."
ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
echo "GPU Status:"
nvidia-smi | head -n 15
ENDSSH

echo ""

# 4. Run experiments
echo "🧬 Step 4/4: Starting experiments..."
# Transfer HF_TOKEN if present
if [ -n "$HF_TOKEN" ]; then
    ssh $SERVER_USER@$SERVER_HOST "echo 'export HF_TOKEN=$HF_TOKEN' > ~/.hf_token_env"
fi

ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
cd ~/genetic_nas

# Load HF_TOKEN if exists
if [ -f ~/.hf_token_env ]; then
    source ~/.hf_token_env
    HF_TOKEN_ARG="-e HF_TOKEN=$HF_TOKEN"
else
    HF_TOKEN_ARG=""
fi

# Clean previous runs
docker stop genetic-nas 2>/dev/null || true
docker rm genetic-nas 2>/dev/null || true

# Test run first (fast check)
echo "Running TensorFlow GPU test..."
docker run --rm --gpus all genetic-nas:latest \
  python3 -c "import tensorflow as tf; print('✅ TensorFlow GPU:', len(tf.config.list_physical_devices('GPU')), 'devices')"

# Start main experiment (using docker run instead of docker-compose)
echo ""
echo "Starting main experiment (detached mode)..."
docker run -d \
  --name genetic-nas \
  --gpus all \
  $HF_TOKEN_ARG \
  -v ~/genetic_nas/output:/app/output \
  genetic-nas:latest \
  python3 main.py --mode fast --generations 10

echo ""
echo "Experiment started! Checking initial output..."
sleep 5
docker logs genetic-nas | tail -n 30
ENDSSH

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "📊 Monitor experiment:"
echo "  ssh $SERVER_USER@$SERVER_HOST 'docker logs -f genetic-nas'"
echo ""
echo "🔍 Check GPU usage:"
echo "  ssh $SERVER_USER@$SERVER_HOST 'watch -n 2 nvidia-smi'"
echo ""
echo "📥 Download results (after completion):"
echo "  scp -r $SERVER_USER@$SERVER_HOST:$SERVER_PATH/output ./output_server"
echo ""
echo "🛑 Stop experiment:"
echo "  ssh $SERVER_USER@$SERVER_HOST 'docker stop genetic-nas'"
