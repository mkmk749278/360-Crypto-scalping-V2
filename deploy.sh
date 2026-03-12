#!/usr/bin/env bash
# 360-Crypto-scalping-V2 — Docker deployment script
set -euo pipefail

echo "🚀 360-Crypto-scalping-V2 — Docker Deployment"
echo "==============================================="

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DO_CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) DO_CLEAN=true ;;
        *) echo "❌ Unknown argument: $arg"; echo "Usage: $0 [--clean]"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Termux is not supported for Docker deployment
# ---------------------------------------------------------------------------
if command -v termux-setup-storage &>/dev/null; then
    echo "❌ Termux detected. Docker deployment is not supported on Termux."
    echo "   Please deploy on a Linux VPS with Docker installed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || {
    echo "❌ Docker not installed. Install it with:"
    echo "   curl -fsSL https://get.docker.com | sh"
    exit 1
}
docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 || {
    echo "❌ Docker Compose not installed."
    exit 1
}

# ---------------------------------------------------------------------------
# --clean: Docker-level cleanup before building
# ---------------------------------------------------------------------------
if [ "$DO_CLEAN" = true ]; then
    echo ""
    echo "🧹 --clean requested: performing Docker-level cleanup before build..."

    if [ -f docker-compose.yml ]; then
        echo "  Stopping existing services..."
        docker compose down 2>/dev/null || true
    fi

    # Remove any orphaned named containers
    if docker ps -a --format '{{.Names}}' | grep -q "^360scalp-v2-engine$"; then
        echo "  Removing container: 360scalp-v2-engine"
        docker stop 360scalp-v2-engine 2>/dev/null || true
        docker rm   360scalp-v2-engine 2>/dev/null || true
    fi

    # Remove project-specific images
    IMAGE_IDS=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' \
        | grep -iE "(^|/)(360scalp|360-crypto-scalping)" | awk '{print $2}' | sort -u || true)
    if [ -n "$IMAGE_IDS" ]; then
        echo "  Removing 360scalp-related images..."
        echo "$IMAGE_IDS" | xargs -r docker rmi -f 2>/dev/null || true
    fi

    echo "  Pruning unused Docker resources..."
    docker system prune -af 2>/dev/null || true
    docker builder prune -af 2>/dev/null || true

    echo "✅ Docker-level cleanup complete."
    echo ""
fi

# ---------------------------------------------------------------------------
# Check .env exists
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your credentials before continuing."
    echo "   nano .env"
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate TELEGRAM_BOT_TOKEN is not still a placeholder
# ---------------------------------------------------------------------------
if grep -q "your_bot_token_here" .env 2>/dev/null; then
    echo "⚠️  TELEGRAM_BOT_TOKEN is still a placeholder. Please edit .env first."
    echo "   nano .env"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create logs directory (for bind-mount fallback / local testing)
# ---------------------------------------------------------------------------
mkdir -p logs

# ---------------------------------------------------------------------------
# Build and start
# ---------------------------------------------------------------------------
echo "🔨 Building Docker image..."
docker compose build --no-cache

echo "🚀 Starting engine..."
docker compose up -d

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Status:"
docker compose ps
echo ""
echo "📋 Useful commands:"
echo "  docker compose logs -f engine      # Follow live logs"
echo "  docker compose restart engine      # Restart the engine"
echo "  docker compose down                # Stop the engine"
echo "  docker compose up -d --build       # Rebuild and restart"
