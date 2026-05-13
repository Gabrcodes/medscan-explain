#!/usr/bin/env bash
# Bootstrap the Streamlit hosting server (Ubuntu 22.04, t3.medium).
# Run on the server after the repo has been cloned to ~/medscan.
set -euo pipefail

REPO="https://github.com/Gabrcodes/medscan-explain.git"
DOMAIN="${DOMAIN:-ai.gabr.online}"
ACME_EMAIL="${ACME_EMAIL:-bedokak@gmail.com}"

echo "== system packages =="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3.11 python3.11-venv python3-pip python3.11-dev \
  build-essential git nginx certbot python3-certbot-nginx \
  swapfile-tools >/dev/null 2>&1 || true

# fallback if python3.11 not in apt: use system python3
if ! command -v python3.11 >/dev/null; then
  sudo apt-get install -y -qq python3 python3-venv python3-pip python3-dev build-essential git nginx certbot python3-certbot-nginx
  PY=python3
else
  PY=python3.11
fi

echo "== swap (2 GB buffer for ConvNeXt-Base on 4 GB RAM) =="
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== repo =="
cd "$HOME"
[ -d medscan ] || git clone "$REPO" medscan
cd medscan

echo "== python venv + CPU torch =="
[ -d .venv ] || $PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
# CPU-only torch (much smaller than the CUDA build) — index URL keeps it small
pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch torchvision
# everything else from our requirements.txt EXCEPT the GPU torch lines (already installed CPU above)
grep -vE '^(torch|torchvision)' requirements.txt | pip install --quiet -r /dev/stdin
pip install --quiet "datasets<4.0"   # script-dataset support if anyone re-trains here

echo "== systemd unit =="
sudo tee /etc/systemd/system/medscan.service >/dev/null <<EOF
[Unit]
Description=MedScan + Explain — Streamlit demo
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$HOME/medscan
Environment="AWS_REGION=us-east-1"
Environment="HF_HOME=$HOME/medscan/data/hf"
ExecStart=$HOME/medscan/.venv/bin/streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable medscan
sudo systemctl restart medscan

echo "== nginx site =="
sudo tee /etc/nginx/sites-available/medscan >/dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        # Streamlit needs websockets
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/medscan /etc/nginx/sites-enabled/medscan
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

echo "== certbot (Let's Encrypt) =="
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$ACME_EMAIL" --redirect || \
  echo "(certbot failed — DNS may not be propagated yet; re-run: sudo certbot --nginx -d $DOMAIN)"

echo "== done =="
sudo systemctl status medscan --no-pager -l | head -20
echo
echo "→ Streamlit will be live at https://$DOMAIN once DNS + certbot complete."
