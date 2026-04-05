#!/usr/bin/env bash
set -euo pipefail

REMOTE="gleb@213.165.215.12"
APP_DIR="/home/gleb/ttsbench"
SERVICE="ttsbench"

echo "==> Синхронизация файлов..."
ssh "$REMOTE" "mkdir -p $APP_DIR"
# Упаковать, исключая лишнее, и распаковать на сервере
tar czf - \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  -C "$(pwd)" . \
  | ssh "$REMOTE" "tar xzf - -C $APP_DIR"

echo "==> Установка uv и зависимостей..."
ssh "$REMOTE" bash -s <<'SETUP'
set -euo pipefail
cd ~/ttsbench

# установить uv если нет
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# синхронизировать зависимости
uv sync --no-dev

# создать data/ если нет
mkdir -p data
[ -f data/library.json ] || echo '[]' > data/library.json
SETUP

echo "==> Настройка systemd-сервиса..."
ssh "$REMOTE" bash -s <<'SERVICE'
set -euo pipefail
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/ttsbench.service <<EOF
[Unit]
Description=TTSBench
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/gleb/ttsbench
ExecStart=/home/gleb/.local/bin/uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=3
Environment=PATH=/home/gleb/.local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable ttsbench
systemctl --user restart ttsbench
SERVICE

echo "==> Готово! Сервис запущен на http://213.165.215.12:8000"
