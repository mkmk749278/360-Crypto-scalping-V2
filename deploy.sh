#!/usr/bin/env bash
# 360-Crypto-Eye-Scalping – deployment script for VPS / Termux
set -euo pipefail

echo "=== 360-Crypto-Eye-Scalping Deployment ==="

# Detect environment
if command -v termux-setup-storage &>/dev/null; then
    ENV="termux"
    echo "Detected: Termux"
    pkg update -y && pkg install -y python git
else
    ENV="vps"
    echo "Detected: VPS / Linux"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment …"
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing dependencies …"
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Create .env if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  .env created from .env.example — please edit it with your credentials:"
    echo "    nano .env"
    echo ""
fi

# Validate .env
if grep -q "your_bot_token_here" .env 2>/dev/null; then
    echo "⚠️  WARNING: TELEGRAM_BOT_TOKEN is not configured in .env"
fi

# Systemd service (VPS only)
if [ "$ENV" = "vps" ] && command -v systemctl &>/dev/null; then
    SERVICE_FILE="/etc/systemd/system/crypto-signal-engine.service"
    if [ ! -f "$SERVICE_FILE" ]; then
        echo "Creating systemd service …"
        sudo tee "$SERVICE_FILE" > /dev/null << UNIT
[Unit]
Description=360 Crypto Eye Scalping Engine
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python -m src.main
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT
        sudo systemctl daemon-reload
        echo "Service created. Start with: sudo systemctl start crypto-signal-engine"
    fi
fi

echo ""
echo "=== Deployment Complete ==="
echo "  Start: python -m src.main"
if [ "$ENV" = "vps" ]; then
    echo "  Or:    sudo systemctl start crypto-signal-engine"
fi
echo "  Logs:  logs/"
echo ""
