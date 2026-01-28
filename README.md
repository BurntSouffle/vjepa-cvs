# V-JEPA 2 CVS Classification

Video-based Critical View of Safety (CVS) classification using V-JEPA 2 features.

## Overview

This project trains a classifier on top of frozen V-JEPA 2 (ViT-L) features to predict the three CVS criteria (C1, C2, C3) from laparoscopic cholecystectomy videos.

**Datasets supported:**
- Endoscapes-CVS201 (201 videos, ~49k frames)
- SAGES CVS Challenge 2025 (700 videos, ~63k frames)
- Combined (both datasets merged)

**Model variants:**
- Pooling: Mean or Attention (learnable)
- Head: MLP (2-layer) or Simple (single linear layer)

## Installation

```bash
# Create conda environment
conda create -n vjepa2 python=3.11
conda activate vjepa2

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate timm einops pillow tqdm scikit-learn pyyaml pandas
```

## Dataset Setup

### RunPod (default paths)
```
/workspace/
├── endoscapes/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── all_metadata.csv
│   ├── train_vids.txt
│   ├── val_vids.txt
│   └── test_vids.txt
└── sages_cvs_challenge_2025_r1/
    └── sages_cvs_challenge_2025/
        ├── frames/
        └── labels/
```

### Local Windows
Create `config_local.yaml` (already in .gitignore) with your paths.

## Usage

### Training

```bash
# RunPod (using default config.yaml paths)
python train.py --config configs/exp1_sages_baseline.yaml

# Local Windows
python train.py --config config_local.yaml

# Quick test (subset of data)
python train.py --config config.yaml --quick-test
```

### Experiment Configs

| Config | Dataset | Pooling | Head | Purpose |
|--------|---------|---------|------|---------|
| `exp1_sages_baseline.yaml` | Combined | Mean | MLP | More data baseline |
| `exp2_sages_attention.yaml` | Combined | Attention | MLP | Better pooling |
| `exp3_sages_simple.yaml` | Combined | Mean | Simple | Reduce overfitting |
| `exp4_endo_attention_simple.yaml` | Endoscapes | Attention | Simple | Max regularisation |

### Evaluation

```bash
python eval.py --checkpoint results/run_XXXXXX/best_model.pt
```

### Compare Experiments

```bash
python compare_experiments.py --results-dir .
```

## Project Structure

```
vjepa/
├── config.yaml              # Default config (RunPod paths)
├── config_local.yaml        # Local config (Windows paths, gitignored)
├── configs/                 # Experiment configurations
│   ├── exp1_sages_baseline.yaml
│   ├── exp2_sages_attention.yaml
│   ├── exp3_sages_simple.yaml
│   └── exp4_endo_attention_simple.yaml
├── model.py                 # V-JEPA CVS model with pooling/head options
├── dataset.py               # Endoscapes dataset
├── dataset_sages.py         # SAGES dataset
├── dataset_combined.py      # Combined dataset
├── train.py                 # Training script
├── eval.py                  # Evaluation script
├── utils.py                 # Utilities (metrics, logging, etc.)
└── compare_experiments.py   # Experiment comparison
```

## Configuration Options

```yaml
model:
  pooling_type: "mean"      # "mean" or "attention"
  head_type: "mlp"          # "mlp" or "simple"

data:
  dataset_type: "endoscapes"  # "endoscapes", "sages", or "combined"
```

## Results

Target: Beat SwinCVS baseline of 67.45% mAP on Endoscapes.

| Experiment | mAP | C1 | C2 | C3 |
|------------|-----|----|----|------|
| SwinCVS (baseline) | 67.45% | 60.0% | 55.0% | 87.0% |
| V-JEPA (TBD) | -- | -- | -- | -- |

## License

Research use only. See dataset licenses for Endoscapes and SAGES.
