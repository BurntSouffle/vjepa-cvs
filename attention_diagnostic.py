"""
V-JEPA Attention Diagnostic Script
===================================
This script:
1. Loads the model and inspects layer structure
2. Extracts attention weights from the pooling layer
3. Visualizes which spatial/temporal positions get highest attention
4. Saves visualizations

Usage:
    python attention_diagnostic.py --checkpoint results/best_model.pt --config config.yaml
    python attention_diagnostic.py --checkpoint results/best_model.pt --config configs/exp2_sages_attention.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inspect_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load and inspect checkpoint structure.
    Returns dict with layer info and model configuration.
    """
    print("=" * 70)
    print(f"STEP 1: INSPECTING CHECKPOINT")
    print(f"Path: {checkpoint_path}")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    result = {
        "checkpoint_keys": list(checkpoint.keys()),
        "training_info": {},
        "layers": {"backbone": [], "pooler": [], "classifier": []},
        "attention_info": {},
        "token_info": {},
    }

    # Extract training info
    for key in ["epoch", "best_val_map", "best_epoch"]:
        if key in checkpoint:
            result["training_info"][key] = checkpoint[key]

    # Get state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    print("\n1. CHECKPOINT CONTENTS:")
    print("-" * 40)
    for key in checkpoint.keys():
        if key == "model_state_dict":
            print(f"  {key}: {len(checkpoint[key])} layers")
        elif key == "optimizer_state_dict":
            print(f"  {key}: optimizer state")
        elif key == "config":
            print(f"  {key}: training config dict")
        else:
            print(f"  {key}: {checkpoint.get(key)}")

    # Group layers
    for name, param in state_dict.items():
        shape = tuple(param.shape)
        size = param.numel()
        info = {"name": name, "shape": shape, "params": size}

        if name.startswith("backbone."):
            result["layers"]["backbone"].append(info)
        elif name.startswith("pooler."):
            result["layers"]["pooler"].append(info)
        elif name.startswith("classifier."):
            result["layers"]["classifier"].append(info)

    print("\n2. LAYER SUMMARY:")
    print("-" * 40)
    backbone_params = sum(l["params"] for l in result["layers"]["backbone"])
    pooler_params = sum(l["params"] for l in result["layers"]["pooler"])
    classifier_params = sum(l["params"] for l in result["layers"]["classifier"])

    print(f"  Backbone: {len(result['layers']['backbone'])} layers, {backbone_params/1e6:.1f}M params")
    print(f"  Pooler:   {len(result['layers']['pooler'])} layers, {pooler_params:,} params")
    print(f"  Classifier: {len(result['layers']['classifier'])} layers, {classifier_params:,} params")

    # Analyze pooler
    print("\n3. POOLER ANALYSIS:")
    print("-" * 40)
    pooler_names = {l["name"]: l["shape"] for l in result["layers"]["pooler"]}

    if not result["layers"]["pooler"]:
        print("  Type: MeanPooling (no learnable parameters)")
        result["attention_info"]["type"] = "mean"
        result["attention_info"]["num_heads"] = None
    elif "pooler.query" in pooler_names:
        print("  Type: AttentionPooling (learnable)")
        result["attention_info"]["type"] = "attention"

        query_shape = pooler_names["pooler.query"]
        print(f"  Query shape: {query_shape}")
        result["attention_info"]["query_shape"] = query_shape

        if "pooler.attention.in_proj_weight" in pooler_names:
            in_proj_shape = pooler_names["pooler.attention.in_proj_weight"]
            embed_dim = in_proj_shape[1]
            print(f"  Embed dim: {embed_dim}")
            result["attention_info"]["embed_dim"] = embed_dim

            # Determine num_heads
            for num_heads in [1, 2, 4, 8, 16]:
                if embed_dim % num_heads == 0:
                    head_dim = embed_dim // num_heads
                    if head_dim in [64, 128, 256]:
                        result["attention_info"]["num_heads"] = num_heads
                        result["attention_info"]["head_dim"] = head_dim
                        print(f"  Num heads: {num_heads} (head_dim={head_dim})")
                        break

    # Analyze classifier
    print("\n4. CLASSIFIER LAYERS:")
    print("-" * 40)
    for layer in result["layers"]["classifier"]:
        print(f"  {layer['name']}: {layer['shape']} ({layer['params']:,} params)")

    # Determine hidden_dim and num_classes
    for layer in result["layers"]["classifier"]:
        if "weight" in layer["name"] and len(layer["shape"]) == 2:
            if "head.1" in layer["name"] or "head.0" in layer["name"]:
                # First linear: hidden_dim -> classifier_hidden
                result["token_info"]["hidden_dim"] = layer["shape"][1]
                result["token_info"]["classifier_hidden"] = layer["shape"][0]
            elif "head.4" in layer["name"] or "head.2" in layer["name"]:
                # Last linear: classifier_hidden -> num_classes
                result["token_info"]["num_classes"] = layer["shape"][0]

    print("\n5. TOKEN DIMENSIONS:")
    print("-" * 40)
    if result["token_info"]:
        print(f"  Hidden dim: {result['token_info'].get('hidden_dim', 'N/A')}")
        print(f"  Classifier hidden: {result['token_info'].get('classifier_hidden', 'N/A')}")
        print(f"  Num classes: {result['token_info'].get('num_classes', 'N/A')}")

    # V-JEPA token info
    print("\n  V-JEPA ViT-L Token Structure:")
    print("    - Patch size: 16x16 pixels")
    print("    - Spatial tokens: 256x256 / 16 = 16x16 = 256 patches")
    print("    - Temporal: fpc=16 means 16 frames per clip")
    print("    - Total tokens: 256 spatial * 16 temporal = 4096 (if no temporal pooling)")
    print("    - Hidden dim: 1024")

    # Save config if available
    if "config" in checkpoint:
        result["saved_config"] = checkpoint["config"]
        print("\n6. SAVED TRAINING CONFIG:")
        print("-" * 40)
        model_cfg = checkpoint["config"].get("model", {})
        for k, v in model_cfg.items():
            print(f"  {k}: {v}")

    return result


class AttentionPoolingWithWeights(nn.Module):
    """
    Modified AttentionPooling that captures attention weights.
    """

    def __init__(self, original_pooler):
        super().__init__()
        self.hidden_dim = original_pooler.hidden_dim
        self.num_heads = original_pooler.num_heads
        self.query = original_pooler.query
        self.attention = original_pooler.attention
        self.norm = original_pooler.norm
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)

        # Get attention output AND weights
        attn_out, self.attention_weights = self.attention(
            query, x, x, need_weights=True, average_attn_weights=False
        )
        # attention_weights shape: (B, num_heads, 1, num_tokens)

        pooled = self.norm(attn_out.squeeze(1))
        return pooled


def load_model_for_analysis(config_path: str, checkpoint_path: str, device: str = "cpu"):
    """
    Load model with modified attention pooler that captures weights.
    """
    print("\n" + "=" * 70)
    print("STEP 2: LOADING MODEL FOR ANALYSIS")
    print("=" * 70)

    # Import model utilities
    import sys
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    from utils import load_config
    from model import create_model

    config = load_config(config_path)
    print(f"Config: {config_path}")
    print(f"Pooling type: {config['model'].get('pooling_type', 'mean')}")
    print(f"Head type: {config['model'].get('head_type', 'mlp')}")

    # Force attention pooling for analysis (even if checkpoint was mean pooling)
    original_pooling = config["model"].get("pooling_type", "mean")
    if original_pooling == "mean":
        print("\nWARNING: Checkpoint uses mean pooling - no attention weights to visualize!")
        print("To analyze attention, train with pooling_type='attention'")

    # Create model
    print("\nCreating model...")
    model = create_model(config)

    # Load checkpoint
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle potential key mismatches
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"\nMissing keys (will use random init): {len(missing)}")
        for k in sorted(missing)[:5]:
            print(f"  - {k}")
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more")

    if unexpected:
        print(f"\nUnexpected keys (will be ignored): {len(unexpected)}")
        for k in sorted(unexpected)[:5]:
            print(f"  - {k}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    # Replace attention pooler with weight-capturing version
    if hasattr(model.pooler, 'attention'):
        print("\nReplacing AttentionPooling with weight-capturing version...")
        model.pooler = AttentionPoolingWithWeights(model.pooler)

    print(f"\nModel loaded on {device}")
    print(f"Total params: {model.get_num_total_params()/1e6:.1f}M")
    print(f"Trainable params: {model.get_num_trainable_params()/1e6:.1f}M")

    return model, config


def get_validation_samples(config: Dict, num_samples: int = 4, targeted: bool = True, criterion_filter: str = None):
    """
    Load validation samples for analysis.

    Args:
        config: Training config dict
        num_samples: Number of samples to return
        targeted: If True, sample specific label combinations
        criterion_filter: If specified, filter for specific criterion:
            - "c1_only": C1=1, C2=0, C3=0
            - "c2_only": C1=0, C2=1, C3=0
            - "c3_only": C1=0, C2=0, C3=1
            - "full_cvs": C1=1, C2=1, C3=1
            - None: mixed sampling (default)
    """
    print("\n" + "=" * 70)
    print("STEP 3: LOADING VALIDATION SAMPLES")
    print("=" * 70)

    from dataset import EndoscapesCVSDataset

    # Create validation dataset
    val_dataset = EndoscapesCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="val",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"].get("frame_step", 25),
        resolution=config["dataset"]["resolution"],
        augment=False,
    )

    samples = []

    # Criterion-specific filtering
    if criterion_filter:
        print(f"\nFiltering for criterion: {criterion_filter}")
        matching_indices = []

        for i, sample_data in enumerate(val_dataset.samples):
            labels = sample_data["labels"]
            c1, c2, c3 = labels[0], labels[1], labels[2]

            if criterion_filter == "c1_only" and c1 == 1 and c2 == 0 and c3 == 0:
                matching_indices.append(i)
            elif criterion_filter == "c2_only" and c1 == 0 and c2 == 1 and c3 == 0:
                matching_indices.append(i)
            elif criterion_filter == "c3_only" and c1 == 0 and c2 == 0 and c3 == 1:
                matching_indices.append(i)
            elif criterion_filter == "full_cvs" and c1 == 1 and c2 == 1 and c3 == 1:
                matching_indices.append(i)

        print(f"  Found {len(matching_indices)} matching samples")

        import random
        random.seed(42)
        selected = random.sample(matching_indices, min(num_samples, len(matching_indices)))

        for idx in selected:
            sample = val_dataset[idx]
            samples.append({
                "video": sample["video"],
                "label": sample["labels"],
                "video_id": f"{sample['meta']['video_id']}_{sample['meta']['center_frame']}",
                "category": criterion_filter
            })

        print(f"\nLoaded {len(samples)} samples for {criterion_filter}:")
        for s in samples:
            label_str = ", ".join([f"C{i+1}={int(s['label'][i])}" for i in range(3)])
            print(f"  {s['video_id']}: [{label_str}]")

        return samples

    if targeted:
        # Categorize all samples by label pattern
        full_cvs = []      # C1=1, C2=1, C3=1
        c1_only = []       # C1=1, C2=0, C3=0
        c1_neg_struct = [] # C1=0, but (C2=1 or C3=1)
        all_neg = []       # C1=0, C2=0, C3=0

        for i, sample_data in enumerate(val_dataset.samples):
            labels = sample_data["labels"]
            c1, c2, c3 = labels[0], labels[1], labels[2]

            if c1 == 1 and c2 == 1 and c3 == 1:
                full_cvs.append(i)
            elif c1 == 1 and c2 == 0 and c3 == 0:
                c1_only.append(i)
            elif c1 == 0 and (c2 == 1 or c3 == 1):
                c1_neg_struct.append(i)
            elif c1 == 0 and c2 == 0 and c3 == 0:
                all_neg.append(i)

        print(f"\nSample categories found:")
        print(f"  Full CVS (C1=C2=C3=1): {len(full_cvs)}")
        print(f"  C1-only (C1=1, others=0): {len(c1_only)}")
        print(f"  C1-neg with structures: {len(c1_neg_struct)}")
        print(f"  All-negative: {len(all_neg)}")

        # Sample from each category
        import random
        random.seed(42)

        samples_per_category = max(1, num_samples // 4)

        categories = [
            ("Full CVS", full_cvs),
            ("C1-positive only", c1_only),
            ("C1-neg with structures", c1_neg_struct),
            ("All-negative", all_neg),
        ]

        for cat_name, indices in categories:
            if indices:
                selected = random.sample(indices, min(samples_per_category, len(indices)))
                for idx in selected:
                    sample = val_dataset[idx]
                    samples.append({
                        "video": sample["video"],
                        "label": sample["labels"],
                        "video_id": f"{sample['meta']['video_id']}_{sample['meta']['center_frame']}",
                        "category": cat_name
                    })

    else:
        # Original simple sampling
        for i, sample_data in enumerate(val_dataset.samples):
            if len(samples) >= num_samples:
                break
            if sum(sample_data["labels"]) > 0 and len(samples) < num_samples // 2:
                sample = val_dataset[i]
                samples.append({
                    "video": sample["video"],
                    "label": sample["labels"],
                    "video_id": f"{sample['meta']['video_id']}_{sample['meta']['center_frame']}",
                    "category": "mixed"
                })

        for i in range(len(val_dataset)):
            if len(samples) >= num_samples:
                break
            sample = val_dataset[i]
            video_id = f"{sample['meta']['video_id']}_{sample['meta']['center_frame']}"
            if video_id not in [s["video_id"] for s in samples]:
                samples.append({
                    "video": sample["video"],
                    "label": sample["labels"],
                    "video_id": video_id,
                    "category": "mixed"
                })

    print(f"\nLoaded {len(samples)} validation samples:")
    for s in samples:
        label_str = ", ".join([f"C{i+1}={int(s['label'][i])}" for i in range(3)])
        cat = s.get('category', 'unknown')
        print(f"  [{cat}] {s['video_id']}: [{label_str}]")

    return samples


def extract_attention_weights(
    model,
    samples: List[Dict],
    device: str = "cpu",
) -> List[Dict]:
    """
    Run forward pass and extract attention weights.
    """
    print("\n" + "=" * 70)
    print("STEP 4: EXTRACTING ATTENTION WEIGHTS")
    print("=" * 70)

    results = []

    for i, sample in enumerate(samples):
        video = sample["video"]
        label = sample["label"]
        video_id = sample["video_id"]

        print(f"\nProcessing sample {i+1}/{len(samples)}: {video_id}")

        # Prepare input
        videos_batch = [video]  # Add batch dimension
        pixel_values = model.process_videos(videos_batch, device)

        print(f"  Input shape: {pixel_values.shape}")

        with torch.no_grad():
            # Get features from backbone
            features = model.backbone.get_vision_features(pixel_values_videos=pixel_values)
            features = features.float()
            print(f"  Features shape: {features.shape}")  # (1, num_tokens, hidden_dim)

            num_tokens = features.shape[1]

            # Get pooled and attention weights
            pooled = model.pooler(features)

            if hasattr(model.pooler, 'attention_weights') and model.pooler.attention_weights is not None:
                attn_weights = model.pooler.attention_weights
                print(f"  Attention weights shape: {attn_weights.shape}")
                # Shape: (1, num_heads, 1, num_tokens)

                # Average across heads and remove batch/query dims
                avg_attn = attn_weights.squeeze(0).squeeze(1).mean(dim=0)  # (num_tokens,)
                print(f"  Averaged attention shape: {avg_attn.shape}")

                attn_numpy = avg_attn.cpu().numpy()
            else:
                print("  No attention weights (using mean pooling)")
                # For mean pooling, all tokens have equal weight
                attn_numpy = np.ones(num_tokens) / num_tokens

            # Get prediction
            logits = model.classifier(pooled)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        print(f"  Predictions: C1={probs[0]:.3f}, C2={probs[1]:.3f}, C3={probs[2]:.3f}")
        print(f"  Ground truth: {label.tolist()}")

        results.append({
            "video_id": video_id,
            "video": video,
            "label": label.numpy(),
            "probs": probs,
            "attention_weights": attn_numpy,
            "num_tokens": num_tokens,
        })

    return results


def visualize_attention(
    results: List[Dict],
    output_dir: str,
    num_frames: int = 16,
    spatial_size: int = 16,  # 16x16 = 256 spatial tokens
):
    """
    Visualize attention weights over spatial and temporal dimensions.
    """
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZING ATTENTION")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for result in results:
        video_id = result["video_id"]
        video = result["video"]
        label = result["label"]
        probs = result["probs"]
        attn = result["attention_weights"]
        num_tokens = result["num_tokens"]

        print(f"\nVisualizing: {video_id}")
        print(f"  Total tokens: {num_tokens}")

        # Determine token layout
        # V-JEPA2 with fpc=16: could be (T*H*W) or pooled differently
        # Try to infer spatial/temporal structure

        if num_tokens == 256 * num_frames:
            # Full spatiotemporal tokens
            temporal_tokens = num_frames
            spatial_tokens = 256
            attn_reshaped = attn.reshape(temporal_tokens, spatial_tokens)
        elif num_tokens == 256:
            # Temporally pooled - only spatial tokens
            temporal_tokens = 1
            spatial_tokens = 256
            attn_reshaped = attn.reshape(1, spatial_tokens)
        elif num_tokens % 256 == 0:
            # Some temporal pooling
            temporal_tokens = num_tokens // 256
            spatial_tokens = 256
            attn_reshaped = attn.reshape(temporal_tokens, spatial_tokens)
        else:
            # Unknown structure - show as 1D
            print(f"  Warning: Unknown token structure ({num_tokens} tokens)")
            temporal_tokens = 1
            spatial_tokens = num_tokens
            attn_reshaped = attn.reshape(1, -1)

        print(f"  Reshaped to: ({temporal_tokens}, {spatial_tokens})")

        # Create figure
        fig = plt.figure(figsize=(16, 10))

        # Title with prediction info
        label_str = ", ".join([f"C{i+1}={'Y' if label[i]>0.5 else 'N'}" for i in range(3)])
        pred_str = ", ".join([f"C{i+1}={probs[i]:.2f}" for i in range(3)])
        fig.suptitle(f"Video: {video_id}\nGT: [{label_str}]  |  Pred: [{pred_str}]", fontsize=14)

        # --- Subplot 1: Temporal attention profile ---
        ax1 = fig.add_subplot(2, 3, 1)
        temporal_attn = attn_reshaped.mean(axis=1)  # Average over spatial
        ax1.bar(range(len(temporal_attn)), temporal_attn)
        ax1.set_xlabel("Frame Index")
        ax1.set_ylabel("Attention Weight")
        ax1.set_title("Temporal Attention Profile")
        ax1.set_xticks(range(0, len(temporal_attn), max(1, len(temporal_attn)//8)))

        # --- Subplot 2: Spatial attention heatmap (averaged over time) ---
        ax2 = fig.add_subplot(2, 3, 2)
        spatial_attn = attn_reshaped.mean(axis=0)  # Average over temporal
        if len(spatial_attn) == 256:
            spatial_map = spatial_attn.reshape(16, 16)
        else:
            side = int(np.sqrt(len(spatial_attn)))
            if side * side == len(spatial_attn):
                spatial_map = spatial_attn.reshape(side, side)
            else:
                spatial_map = spatial_attn.reshape(1, -1)

        im2 = ax2.imshow(spatial_map, cmap='hot', interpolation='bilinear')
        ax2.set_title("Spatial Attention (avg over time)")
        ax2.set_xlabel("Patch X")
        ax2.set_ylabel("Patch Y")
        plt.colorbar(im2, ax=ax2)

        # --- Subplot 3: Full spatiotemporal attention heatmap ---
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(attn_reshaped, cmap='hot', aspect='auto', interpolation='nearest')
        ax3.set_title("Spatiotemporal Attention")
        ax3.set_xlabel("Spatial Token Index")
        ax3.set_ylabel("Temporal Index")
        plt.colorbar(im3, ax=ax3)

        # --- Subplot 4-6: Sample video frames with attention overlay ---
        frame_indices = [0, num_frames//2, num_frames-1] if len(video) >= num_frames else [0]

        for idx, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(2, 3, 4 + idx)

            if frame_idx < len(video):
                frame = video[frame_idx]  # (H, W, C) uint8

                # Get spatial attention for this frame
                if temporal_tokens > 1 and frame_idx < temporal_tokens:
                    frame_attn = attn_reshaped[frame_idx]
                else:
                    frame_attn = spatial_attn

                # Reshape to spatial grid
                if len(frame_attn) == 256:
                    attn_map = frame_attn.reshape(16, 16)
                else:
                    attn_map = frame_attn.reshape(int(np.sqrt(len(frame_attn))), -1)

                # Resize attention map to match frame
                from scipy.ndimage import zoom
                h, w = frame.shape[:2]
                scale_h = h / attn_map.shape[0]
                scale_w = w / attn_map.shape[1]
                attn_resized = zoom(attn_map, (scale_h, scale_w), order=1)

                # Show frame
                ax.imshow(frame)
                ax.imshow(attn_resized, cmap='jet', alpha=0.4)
                ax.set_title(f"Frame {frame_idx}")
                ax.axis('off')

        plt.tight_layout()

        # Save
        save_path = output_path / f"attention_{video_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    # Create summary figure with all samples
    print("\nCreating summary figure...")
    n_samples = len(results)
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for i, result in enumerate(results):
        video_id = result["video_id"]
        video = result["video"]
        attn = result["attention_weights"]

        # Top row: middle frame
        ax_frame = axes[0, i]
        mid_frame = video[len(video)//2]
        ax_frame.imshow(mid_frame)
        ax_frame.set_title(f"{video_id}")
        ax_frame.axis('off')

        # Bottom row: attention heatmap
        ax_attn = axes[1, i]
        if len(attn) == 256:
            spatial_map = attn.reshape(16, 16)
        elif len(attn) % 256 == 0:
            spatial_map = attn.reshape(-1, 256).mean(axis=0).reshape(16, 16)
        else:
            spatial_map = attn.reshape(1, -1)

        ax_attn.imshow(spatial_map, cmap='hot')
        ax_attn.set_title(f"Spatial Attention")
        ax_attn.axis('off')

    plt.tight_layout()
    summary_path = output_path / "attention_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA Attention Diagnostic")
    parser.add_argument("--checkpoint", type=str, default="results/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="results/attention_analysis",
                        help="Output directory for visualizations")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of validation samples to analyze")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only inspect checkpoint, don't load full model")
    parser.add_argument("--targeted", action="store_true", default=True,
                        help="Sample specific label combinations (Full CVS, C1-only, etc.)")
    parser.add_argument("--criterion", type=str, default=None,
                        choices=["c1_only", "c2_only", "c3_only", "full_cvs"],
                        help="Filter for specific criterion pattern")
    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Step 1: Inspect checkpoint
    checkpoint_info = inspect_checkpoint(args.checkpoint)

    if args.inspect_only:
        print("\n" + "=" * 70)
        print("INSPECTION COMPLETE (--inspect-only mode)")
        print("=" * 70)
        return

    # Check if attention pooling is available
    if checkpoint_info["attention_info"].get("type") == "mean":
        print("\n" + "=" * 70)
        print("WARNING: This checkpoint uses MEAN POOLING")
        print("=" * 70)
        print("Mean pooling has no learnable attention weights to visualize.")
        print("Options:")
        print("  1. Train a model with pooling_type='attention'")
        print("  2. Use --inspect-only to just see checkpoint structure")
        print("\nContinuing anyway (will show uniform attention)...")

    # Step 2: Load model
    model, config = load_model_for_analysis(args.config, args.checkpoint, device)

    # Step 3: Get validation samples
    samples = get_validation_samples(config, args.num_samples, targeted=args.targeted, criterion_filter=args.criterion)

    if not samples:
        print("ERROR: No validation samples found!")
        return

    # Step 4: Extract attention weights
    results = extract_attention_weights(model, samples, device)

    # Step 5: Visualize
    visualize_attention(
        results,
        args.output,
        num_frames=config["dataset"]["num_frames"],
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Visualizations saved to: {args.output}")

    # Save numeric results
    results_file = Path(args.output) / "attention_results.json"
    json_results = []
    for r in results:
        json_results.append({
            "video_id": str(r["video_id"]),
            "label": r["label"].tolist(),
            "probs": r["probs"].tolist(),
            "num_tokens": r["num_tokens"],
            "attention_stats": {
                "mean": float(r["attention_weights"].mean()),
                "std": float(r["attention_weights"].std()),
                "max": float(r["attention_weights"].max()),
                "min": float(r["attention_weights"].min()),
                "top5_indices": r["attention_weights"].argsort()[-5:][::-1].tolist(),
            }
        })

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Numeric results saved to: {results_file}")


if __name__ == "__main__":
    main()
