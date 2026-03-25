#!/usr/bin/env bash
# ONE-SHOT RunPod setup — run this once after uploading competition_pack to the pod.
# Usage:  bash runpod_oneshot.sh
#
# What it does:
#   1. Creates venv
#   2. Installs PyTorch + all deps
#   3. Logs into HuggingFace (you paste your token)
#   4. Downloads PlantVillage dataset
#   5. Prints GPU info + next steps
set -euo pipefail

echo "============================================"
echo "  RunPod One-Shot Setup"
echo "============================================"

# 1. Venv
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip --quiet

# 2. PyTorch + deps
echo "[2/5] Installing PyTorch (CUDA 12.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
echo "[2/5] Installing other dependencies..."
pip install transformers datasets huggingface_hub accelerate scikit-learn pillow timm open_clip_torch --quiet

# 3. HuggingFace login
echo "[3/5] HuggingFace login (paste your token):"
huggingface-cli login

# 4. GPU check
echo "[4/5] GPU check:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected!')
"

# 5. Download PlantVillage so it's cached before the clock starts
echo "[5/5] Pre-downloading PlantVillage dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('abdallahalidev/plantvillage-dataset', split='train[:10]')
print(f'  Dataset accessible. Preview: {len(ds)} samples loaded.')
" 2>/dev/null || echo "  (Optional) Dataset pre-download failed — will download during training."

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  # Quick baseline (CLIP zero-shot, no training):"
echo "  .venv/bin/python advanced/clip_zero_shot.py --test-dir data/test"
echo ""
echo "  # Train ViT-large (recommended on A100):"
echo "  .venv/bin/python advanced/train_vit_large.py --data-dir data/train --max-images 0"
echo ""
echo "  # Train DINOv2 (for ensembling):"
echo "  .venv/bin/python advanced/train_dinov2.py --data-dir data/train --max-images 0"
echo ""
echo "  # TIP: Use tmux so training survives disconnects:"
echo "  tmux new -s train"
echo "  # ... run training ... then Ctrl+B, D to detach"
echo "  # tmux attach -t train   to come back"
echo ""
