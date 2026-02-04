#!/bin/bash
# Setup script for anatomy-guided V-JEPA pretraining on RunPod
# ============================================================
#
# Prerequisites:
#   - A100 80GB GPU (recommended for full training)
#   - Python 3.10+
#   - CUDA 11.8+
#
# Run this script after starting your RunPod instance:
#   bash setup_runpod.sh

set -e  # Exit on error

echo "=========================================="
echo " Anatomy-Guided V-JEPA Pretraining Setup"
echo "=========================================="

# --- 1. Install dependencies ---
echo ""
echo "[1/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install decord pandas numpy pyyaml tqdm pillow
pip install timm einops

# --- 2. Clone repos if not present ---
echo ""
echo "[2/6] Setting up repositories..."
cd /workspace

if [ ! -d "vjepa_pretrain" ]; then
    echo "Cloning V-JEPA repo..."
    git clone https://github.com/facebookresearch/jepa.git vjepa_pretrain
fi

# --- 3. Copy custom surgical pretraining files ---
echo ""
echo "[3/6] Installing surgical pretraining modules..."

# Copy anatomy-guided mask collator
cp -v /workspace/vjepa/vjepa_pretrain/src/masks/anatomy_guided_multiblock3d.py \
      /workspace/vjepa_pretrain/src/masks/

# Copy surgical video dataset
cp -v /workspace/vjepa/vjepa_pretrain/src/datasets/surgical_video_dataset.py \
      /workspace/vjepa_pretrain/src/datasets/

# Copy training script
mkdir -p /workspace/vjepa_pretrain/app/vjepa_surgical
cp -v /workspace/vjepa/vjepa_pretrain/app/vjepa_surgical/train.py \
      /workspace/vjepa_pretrain/app/vjepa_surgical/
cp -v /workspace/vjepa/vjepa_pretrain/app/vjepa_surgical/__init__.py \
      /workspace/vjepa_pretrain/app/vjepa_surgical/

# Copy config
cp -v /workspace/vjepa/vjepa_pretrain/configs/pretrain/surgical_vitl16.yaml \
      /workspace/vjepa_pretrain/configs/pretrain/

# --- 4. Setup data directories ---
echo ""
echo "[4/6] Setting up data directories..."

mkdir -p /workspace/data/endoscapes
mkdir -p /workspace/data/sages

# Create symlinks if data already exists elsewhere
if [ -d "/workspace/vjepa/data/endoscapes" ]; then
    echo "Linking Endoscapes data..."
    ln -sf /workspace/vjepa/data/endoscapes/* /workspace/data/endoscapes/ 2>/dev/null || true
fi

# --- 5. Download pretrained V-JEPA checkpoint ---
echo ""
echo "[5/6] Setting up pretrained checkpoint..."

# Note: V-JEPA 2 checkpoints are available from HuggingFace
# We'll use the transformers library to download
pip install transformers huggingface_hub

# The checkpoint will be downloaded automatically by HuggingFace transformers
# when we load the model, or we can pre-download:
python -c "
from huggingface_hub import hf_hub_download
import os

# Download V-JEPA 2 ViT-L checkpoint
try:
    path = hf_hub_download(
        repo_id='facebook/vjepa2-vitl-fpc16-256-ssv2',
        filename='pytorch_model.bin',
        local_dir='/workspace/checkpoints/vjepa2_vitl'
    )
    print(f'Downloaded checkpoint to: {path}')
except Exception as e:
    print(f'Note: Could not pre-download checkpoint: {e}')
    print('The model will be downloaded when training starts.')
"

# --- 6. Verify installation ---
echo ""
echo "[6/6] Verifying installation..."

cd /workspace/vjepa_pretrain
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Test imports
from src.masks.anatomy_guided_multiblock3d import AnatomyGuidedMaskCollator
from src.datasets.surgical_video_dataset import SurgicalVideoDataset
print('Custom modules imported successfully!')
"

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  cd /workspace/vjepa_pretrain"
echo "  python app/vjepa_surgical/train.py --fname configs/pretrain/surgical_vitl16.yaml"
echo ""
echo "Or for distributed training:"
echo "  torchrun --nproc_per_node=1 app/vjepa_surgical/train.py --fname configs/pretrain/surgical_vitl16.yaml"
echo ""
