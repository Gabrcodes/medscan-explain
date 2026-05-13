#!/usr/bin/env bash
# Bootstrap + train MedScan on a fresh AWS Deep Learning AMI (PyTorch, Ubuntu) GPU box.
# Usage (on the instance, after the repo has been unpacked to ~/medscan):
#   cd ~/medscan && bash scripts/run_on_instance.sh pneumonia efficientnet_b0 10
set -euo pipefail

DATASET="${1:-pneumonia}"
BACKBONE="${2:-efficientnet_b0}"
EPOCHS="${3:-10}"

# The AWS Deep Learning AMI ships PyTorch in a venv, not the system python.
if [ -f /opt/pytorch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /opt/pytorch/bin/activate
fi
PY=python

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== installing project deps (torch/torchvision already on the DLAMI) =="
python3 -m pip install --quiet --upgrade pip
# install everything except torch/torchvision (the DLAMI ships those with the right CUDA build)
python3 -m pip install --quiet \
  "timm>=0.9.16" "datasets>=2.18" "scikit-learn>=1.4" matplotlib seaborn \
  "grad-cam>=1.5.0" tqdm pyyaml "huggingface_hub>=0.23"

echo "== training: $DATASET / $BACKBONE / ${EPOCHS}ep =="
python3 -m src.train --dataset "$DATASET" --backbone "$BACKBONE" --epochs "$EPOCHS"

echo "== evaluating =="
CKPT="checkpoints/${DATASET}_${BACKBONE}_best.pt"
python3 -m src.evaluate --checkpoint "$CKPT"

echo "== done. artifacts: =="
ls -la checkpoints/ outputs/
