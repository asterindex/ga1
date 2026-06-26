#!/bin/bash
# Stop all ga2 containers and experiments on server

set -e

SERVER_USER="anatoly_kot"
SERVER_HOST="193.200.64.103"
SSH_KEY="${SSH_KEY:-priv.key.pem}"

SSH_OPTS=(-o ConnectTimeout=15)
if [[ -f "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

echo "Stopping ga2 on $SERVER_USER@$SERVER_HOST ..."

ssh "${SSH_OPTS[@]}" "$SERVER_USER@$SERVER_HOST" << 'ENDSSH'
set -e
echo "Running containers:"
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' | grep -E 'ga2|genetic-nas' || docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' | head -5

# Stop ga2-named containers
docker stop ga2 2>/dev/null || true
docker rm -f ga2 2>/dev/null || true

# Stop any container from ga2 image
for id in $(docker ps -q --filter ancestor=ga2:latest); do
  docker stop "$id" 2>/dev/null || true
done

# Stop detached docker-compose if any
cd ~/ga2 2>/dev/null && docker compose down 2>/dev/null || docker-compose down 2>/dev/null || true

echo "Remaining containers:"
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' | head -10
echo "Done."
ENDSSH
