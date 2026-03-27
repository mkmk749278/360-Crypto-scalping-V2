#!/usr/bin/env bash
# 360-Crypto-Eye-Scalping — Complete Nuke-and-Deploy Script
# Usage: sudo bash clean_deploy.sh
# Tested on Ubuntu 20.04, 22.04, 24.04
set -euo pipefail

# ─── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $*${NC}"; }
err()  { echo -e "${RED}❌  $*${NC}"; }
info() { echo -e "${CYAN}ℹ️   $*${NC}"; }
hdr()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; \
         echo -e "${BOLD}${CYAN}  $*${NC}"; \
         echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; }

# ─── Must run as root ──────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    err "This script must be run as root (sudo bash clean_deploy.sh)"
    exit 1
fi

# Resolve the non-root home directory (works whether called with sudo or as root)
REAL_USER="${SUDO_USER:-root}"
if [[ "$REAL_USER" == "root" ]]; then
    DEPLOY_HOME="/root"
else
    DEPLOY_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
fi
PROJECT_DIR="$DEPLOY_HOME/360-Crypto-scalping-V2"
REPO_URL="https://github.com/mkmk749278/360-Crypto-scalping-V2.git"
SERVICE_NAME="360-crypto-engine"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — STOP EVERYTHING
# ──────────────────────────────────────────────────────────────────────────────
hdr "PHASE 1 — STOP EVERYTHING"

# Stop known systemd services (ignore errors — service may not exist)
CRYPTO_SERVICES=(
    "crypto-signal-engine"
    "360-crypto"
    "crypto-engine"
    "360-crypto-engine"
)
for svc in "${CRYPTO_SERVICES[@]}"; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        info "Stopping service: $svc"
        systemctl stop "$svc" || true
    fi
    if systemctl is-enabled --quiet "$svc" 2>/dev/null; then
        info "Disabling service: $svc"
        systemctl disable "$svc" || true
    fi
done

# Catch any other systemd service whose name contains "crypto"
while IFS= read -r svc; do
    [[ -z "$svc" ]] && continue
    info "Stopping extra crypto service: $svc"
    systemctl stop "$svc" || true
    systemctl disable "$svc" || true
done < <(systemctl list-units --type=service --all --no-legend 2>/dev/null \
         | awk '{print $1}' | grep -i crypto || true)

# Kill Python processes matching crypto/signal-engine patterns
info "Killing matching Python processes …"
for pat in "src.main" "main.py" "360-Crypto" "360_Crypto" "crypto-signal-engine"; do
    pkill -f "$pat" 2>/dev/null && warn "Killed processes matching: $pat" || true
done

# Kill screen sessions with crypto-related names
if command -v screen &>/dev/null; then
    screen -ls 2>/dev/null | grep -iE 'crypto|scalp|360' | awk -F'.' '{print $1}' \
        | tr -d '\t ' | while read -r sid; do
            [[ -z "$sid" ]] && continue
            info "Killing screen session: $sid"
            screen -X -S "$sid" quit 2>/dev/null || true
        done
fi

# Kill tmux sessions with crypto-related names
if command -v tmux &>/dev/null; then
    tmux list-sessions 2>/dev/null | grep -iE 'crypto|scalp|360' | cut -d: -f1 \
        | while read -r sess; do
            [[ -z "$sess" ]] && continue
            info "Killing tmux session: $sess"
            tmux kill-session -t "$sess" 2>/dev/null || true
        done
fi

info "Waiting 3 seconds for processes to die …"
sleep 3

# Force-kill anything still alive
for pat in "src.main" "main.py"; do
    pkill -9 -f "$pat" 2>/dev/null || true
done

ok "Phase 1 complete — all crypto processes stopped"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — CLEAN EVERYTHING
# ──────────────────────────────────────────────────────────────────────────────
hdr "PHASE 2 — CLEAN EVERYTHING"

# Remove old project directories
OLD_DIRS=(
    "$DEPLOY_HOME/360-Crypto-scalping-V2"
    "$DEPLOY_HOME/360-Crypto-Eye-Scalping"
    "$DEPLOY_HOME/crypto-signal-engine"
)
for d in "${OLD_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        info "Removing directory: $d"
        rm -rf "$d"
    fi
done

# Glob-based removal for wildcard patterns in home dir
for pattern in "360-crypto*" "*360*crypto*" "*crypto*scalping*"; do
    # Use find to avoid glob expansion errors when no matches exist
    while IFS= read -r d; do
        [[ -z "$d" ]] && continue
        info "Removing: $d"
        rm -rf "$d"
    done < <(find "$DEPLOY_HOME" -maxdepth 1 -type d -iname "$pattern" 2>/dev/null || true)
done

# Remove /opt entries
find /opt -maxdepth 1 -type d -iname "*360-crypto*" 2>/dev/null | while read -r d; do
    info "Removing /opt directory: $d"
    rm -rf "$d"
done

# Remove old systemd service files
for svc in "crypto-signal-engine" "360-crypto" "crypto-engine" "360-crypto-engine"; do
    f="/etc/systemd/system/${svc}.service"
    if [[ -f "$f" ]]; then
        info "Removing service file: $f"
        rm -f "$f"
    fi
done

# Flush Redis database 0
if command -v redis-cli &>/dev/null; then
    info "Flushing Redis database 0 …"
    redis-cli -n 0 FLUSHDB 2>/dev/null && ok "Redis DB 0 flushed" \
        || warn "Redis flush failed (Redis may not be running yet — will start later in Phase 3)"
fi

# Remove old log files
rm -f "$DEPLOY_HOME"/logs/crypto* 2>/dev/null || true
rm -f /var/log/crypto* 2>/dev/null || true

# Reload systemd to pick up removed unit files
systemctl daemon-reload 2>/dev/null || true

ok "Phase 2 complete — old files and processes removed"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — INSTALL PREREQUISITES
# ──────────────────────────────────────────────────────────────────────────────
hdr "PHASE 3 — INSTALL PREREQUISITES"

info "Updating package lists …"
apt-get update -y

info "Upgrading installed packages …"
apt-get upgrade -y

# Try to install python3.11; add deadsnakes PPA if not available
info "Installing Python 3.11 and dependencies …"
if ! apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null; then
    warn "python3.11 not found in default repos — adding deadsnakes PPA …"
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
    apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

info "Installing system tools …"
apt-get install -y python3-pip git redis-server build-essential curl wget

# Enable and start Redis
info "Enabling and starting Redis …"
systemctl enable redis-server
systemctl start redis-server

# Verify Python 3.11+
if python3.11 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
    PY_VER=$(python3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    ok "Python $PY_VER available ✅"
else
    err "Python 3.11+ is required but could not be installed"
    exit 1
fi

# Verify Redis is running
if redis-cli ping 2>/dev/null | grep -q "PONG"; then
    ok "Redis is running ✅"
else
    err "Redis is not responding to PING"
    exit 1
fi

ok "Phase 3 complete — prerequisites installed"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FRESH CLONE & SETUP
# ──────────────────────────────────────────────────────────────────────────────
hdr "PHASE 4 — FRESH CLONE & SETUP"

info "Cloning repository into $PROJECT_DIR …"
git clone "$REPO_URL" "$PROJECT_DIR"
cd "$PROJECT_DIR"

info "Creating Python 3.11 virtual environment …"
python3.11 -m venv venv

info "Activating virtual environment and upgrading pip …"
# shellcheck source=/dev/null
source venv/bin/activate
pip install --upgrade pip

# Purge stale pip caches for a clean install (best-effort)
python3.11 -m pip cache purge 2>/dev/null || true

info "Installing Python dependencies …"
pip install -r requirements.txt

info "Creating logs directory …"
mkdir -p logs

# Create .env from example
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warn ".env created from .env.example"
    warn "⚠️  IMPORTANT: Edit $PROJECT_DIR/.env with your actual credentials before starting the engine"
    warn "   Run: nano $PROJECT_DIR/.env"
else
    info ".env already exists — skipping copy"
fi

ok "Phase 4 complete — repository cloned and dependencies installed"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — CREATE SYSTEMD SERVICE
# ──────────────────────────────────────────────────────────────────────────────
hdr "PHASE 5 — CREATE SYSTEMD SERVICE"

VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

info "Writing systemd unit file: $SERVICE_FILE …"
cat > "$SERVICE_FILE" << UNIT
[Unit]
Description=360 Crypto Eye Scalping Signal Engine
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PYTHON -m src.main
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

info "Reloading systemd daemon …"
systemctl daemon-reload

info "Enabling $SERVICE_NAME to start on boot …"
systemctl enable "$SERVICE_NAME"

ok "Phase 5 complete — systemd service created and enabled"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6 — START & VERIFY
# ──────────────────────────────────────────────────────────────────────────────
hdr "PHASE 6 — START & VERIFY"

info "Starting $SERVICE_NAME …"
systemctl start "$SERVICE_NAME"

info "Waiting 5 seconds for service to initialise …"
sleep 5

SERVICE_STATUS=$(systemctl is-active "$SERVICE_NAME" 2>/dev/null || echo "unknown")

echo ""
info "Last 20 lines of journal output:"
echo "────────────────────────────────────────────"
journalctl -u "$SERVICE_NAME" -n 20 --no-pager 2>/dev/null || true
echo "────────────────────────────────────────────"
echo ""

# ─── Final Summary ─────────────────────────────────────────────────────────────
hdr "DEPLOYMENT SUMMARY"

if [[ "$SERVICE_STATUS" == "active" ]]; then
    ok "Service status: ACTIVE ✅"
else
    warn "Service status: $SERVICE_STATUS"
    warn "The engine may still be starting up or waiting for .env credentials."
fi

echo ""
echo -e "${BOLD}Management commands:${NC}"
echo -e "  ${GREEN}View live logs :${NC}  journalctl -u $SERVICE_NAME -f"
echo -e "  ${GREEN}Restart engine :${NC}  systemctl restart $SERVICE_NAME"
echo -e "  ${GREEN}Stop engine    :${NC}  systemctl stop $SERVICE_NAME"
echo -e "  ${GREEN}Service status :${NC}  systemctl status $SERVICE_NAME"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT — verify your credentials:${NC}"
echo -e "  ${CYAN}nano $PROJECT_DIR/.env${NC}"
echo -e "  Then restart: ${CYAN}systemctl restart $SERVICE_NAME${NC}"
echo ""
ok "Clean deployment complete 🎉"
