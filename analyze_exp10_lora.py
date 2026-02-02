"""
Exp10 LoRA Model Analysis
=========================
Analyze our best model (53.75% mAP) to understand:

1. Did LoRA change V-JEPA's internal attention patterns?
2. Is attention now more focused on anatomy vs ~98% uniform baseline?
3. Are C2 predictions improved (was 0/5 before)?

Compares:
- V-JEPA internal attention BEFORE vs AFTER LoRA
- Predictions on specific samples
- Entropy analysis (focus vs uniformity)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent))

from dataset_multitask import MultiTaskCVSDataset
from utils import load_config


# Segmentation class info
CLASS_NAMES = {
    0: "Background",
    1: "Cystic Plate (C2)",
    2: "Calot's Triangle (C1)",
    3: "Cystic Artery (C3)",
    4: "Cystic Duct (C3)",
}

CLASS_COLORS = {
    0: [0.2, 0.2, 0.2],
    1: [0.0, 0.8, 0.0],
    2: [0.0, 0.5, 1.0],
    3: [1.0, 0.0, 0.0],
    4: [1.0, 0.8, 0.0],
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class indices to RGB."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)
    for class_idx, color in CLASS_COLORS.items():
        colored[mask == class_idx] = color
    return colored


def compute_entropy(attn):
    """Compute entropy of attention distribution (measure of focus)."""
    attn_flat = attn.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
    max_entropy = np.log(len(attn_flat))
    return entropy, max_entropy, entropy / max_entropy * 100  # normalized %


def get_vjepa_attention_from_last_layer(backbone, pixel_values, device):
    """
    Extract attention from V-JEPA's last transformer layer.

    For LoRA models, we need to access the base model through PEFT wrapper.
    """
    captured_qkv = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            captured_qkv[name] = output.detach().cpu()
        return hook_fn

    # Find the actual V-JEPA model (may be wrapped by PEFT)
    actual_backbone = backbone
    if hasattr(backbone, 'base_model'):
        actual_backbone = backbone.base_model
    if hasattr(actual_backbone, 'model'):
        actual_backbone = actual_backbone.model

    # Find attention layers
    target_layer = None
    last_layer_idx = -1

    if hasattr(actual_backbone, 'encoder') and hasattr(actual_backbone.encoder, 'layer'):
        last_layer_idx = len(actual_backbone.encoder.layer) - 1
        target_layer = actual_backbone.encoder.layer[last_layer_idx].attention

    if target_layer is None:
        print("  Could not find attention layer structure")
        return None

    # Register hooks on Q, K projections
    hooks = []

    if hasattr(target_layer, 'query'):
        hooks.append(target_layer.query.register_forward_hook(make_hook('query')))
    if hasattr(target_layer, 'key'):
        hooks.append(target_layer.key.register_forward_hook(make_hook('key')))

    if not hooks:
        print("  No Q/K projections found")
        return None

    try:
        with torch.no_grad():
            _ = backbone.get_vision_features(pixel_values_videos=pixel_values)
    finally:
        for hook in hooks:
            hook.remove()

    # Compute attention weights from Q and K
    if 'query' in captured_qkv and 'key' in captured_qkv:
        Q = captured_qkv['query']
        K = captured_qkv['key']

        B, seq_len, hidden_dim = Q.shape
        num_heads = 16  # ViT-L default
        head_dim = hidden_dim // num_heads

        # Reshape to (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        K = K.view(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Compute attention: softmax(QK^T / sqrt(d_k))
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        return attn_weights

    return None


class AttentionPoolingWithWeights(nn.Module):
    """Modified AttentionPooling that captures attention weights."""

    def __init__(self, original_pooler):
        super().__init__()
        self.query = original_pooler.query
        self.attention = original_pooler.attention
        self.norm = original_pooler.norm
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, self.attention_weights = self.attention(
            query, x, x, need_weights=True, average_attn_weights=False
        )
        return self.norm(attn_out.squeeze(1))


def load_lora_model(checkpoint_path: str, device: torch.device):
    """Load LoRA fine-tuned model."""
    print(f"Loading LoRA model from: {checkpoint_path}")

    # Import LoRA model class
    from train_lora import VJEPA_LoRA

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Create model with same config
    lora_cfg = config.get("lora", {})
    model_cfg = config.get("model", {})

    model = VJEPA_LoRA(
        model_name=model_cfg.get("name", "facebook/vjepa2-vitl-fpc16-256-ssv2"),
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        target_modules=lora_cfg.get("target_modules"),
        cvs_hidden=model_cfg.get("cvs_hidden", 512),
        cvs_dropout=model_cfg.get("cvs_dropout", 0.5),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        num_seg_classes=model_cfg.get("num_seg_classes", 5),
        seg_output_size=model_cfg.get("seg_output_size", 64),
        seg_dropout=model_cfg.get("seg_dropout", 0.1),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Replace pooler with attention-capturing version
    model.pooler = AttentionPoolingWithWeights(model.pooler)

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best mAP: {checkpoint.get('best_metric', 0)*100:.2f}%")

    return model, config


def load_baseline_model(device: torch.device):
    """Load baseline V-JEPA model (no fine-tuning)."""
    print("Loading baseline V-JEPA model (no fine-tuning)...")

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
    backbone = backbone.float()
    backbone = backbone.to(device)
    backbone.eval()

    return backbone


def find_specific_samples(dataset, target_video_ids: List[str]) -> Dict[str, int]:
    """Find specific samples by video_id pattern (e.g., '127_19475')."""
    found = {}

    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        vid = sample_info["video_id"]
        frame = sample_info["center_frame"]
        sample_id = f"{vid}_{frame}"

        if sample_id in target_video_ids:
            found[sample_id] = i

    print(f"Found {len(found)}/{len(target_video_ids)} target samples:")
    for sample_id in target_video_ids:
        if sample_id in found:
            print(f"  {sample_id}: index {found[sample_id]}")
        else:
            print(f"  {sample_id}: NOT FOUND")

    return found


def find_c2_only_sample(dataset) -> Optional[int]:
    """Find a sample with only C2 positive (Cystic Plate visible but not C1/C3)."""
    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        labels = sample_info["labels"]

        # C2-only: C1=0, C2=1, C3=0
        if labels[0] == 0 and labels[1] == 1 and labels[2] == 0:
            return i

    return None


def process_sample_lora(model, sample, device):
    """Process sample through LoRA model and extract attention."""
    video = sample["video"]
    labels = sample["labels"]
    masks = sample["masks"]
    mask_indices = sample["mask_indices"]
    meta = sample["meta"]

    # Get middle frame for visualization
    middle_idx = len(video) // 2
    frame = video[middle_idx]

    # Find mask for middle frame if available
    gt_mask = None
    mask_frame_idx = middle_idx

    for i, idx in enumerate(mask_indices.tolist()):
        if idx == middle_idx:
            gt_mask = masks[i].numpy()
            break

    if gt_mask is None and len(mask_indices) > 0:
        gt_mask = masks[0].numpy()
        mask_frame_idx = mask_indices[0].item()

    # Process video
    video_batch = [video]
    pixel_values = model.process_videos(video_batch, device)

    # Extract V-JEPA internal attention
    vjepa_attn = get_vjepa_attention_from_last_layer(model.backbone, pixel_values, device)

    # Forward pass
    if gt_mask is not None:
        frame_indices = torch.tensor([mask_frame_idx], device=device)
        batch_indices = torch.tensor([0], device=device)
    else:
        frame_indices = torch.tensor([], dtype=torch.long, device=device)
        batch_indices = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        features = model.backbone.get_vision_features(pixel_values_videos=pixel_values)
        features = features.float()

        # Get pooler attention
        pooled = model.pooler(features)
        pooler_attn = model.pooler.attention_weights

        # CVS predictions
        cvs_logits = model.cvs_head(pooled)
        cvs_pred = torch.sigmoid(cvs_logits[0]).cpu().numpy()

        # Segmentation predictions
        if len(frame_indices) > 0:
            seg_logits = model.seg_head(features, frame_indices, batch_indices)
            pred_mask = seg_logits[0].argmax(dim=0).cpu().numpy()
        else:
            pred_mask = None

    # Process pooler attention
    if pooler_attn is not None:
        pooler_attn = pooler_attn.squeeze(0).squeeze(1).mean(dim=0).cpu().numpy()

    return {
        "frame": frame,
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
        "cvs_pred": cvs_pred,
        "cvs_gt": labels.numpy(),
        "vjepa_attn": vjepa_attn,
        "pooler_attn": pooler_attn,
        "meta": meta,
    }


def process_sample_baseline(backbone, sample, device):
    """Process sample through baseline V-JEPA and extract attention."""
    video = sample["video"]
    labels = sample["labels"]

    # Get middle frame
    middle_idx = len(video) // 2
    frame = video[middle_idx]

    # Process video - need to manually normalize
    video_tensor = torch.from_numpy(np.stack([video]))  # (1, T, H, W, C)
    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    pixel_values = ((video_tensor - mean) / std).to(device)

    # Extract V-JEPA internal attention
    vjepa_attn = get_vjepa_attention_from_last_layer(backbone, pixel_values, device)

    return {
        "frame": frame,
        "cvs_gt": labels.numpy(),
        "vjepa_attn": vjepa_attn,
    }


def create_attention_comparison_figure(lora_results, baseline_results, output_path):
    """Create figure comparing LoRA vs baseline attention."""
    n_samples = len(lora_results)

    fig = plt.figure(figsize=(20, 5 * n_samples))

    # Columns: Frame | Baseline Attn | LoRA Attn | Baseline Heatmap | LoRA Heatmap | Seg Pred
    n_cols = 6

    baseline_entropies = []
    lora_entropies = []

    for row, (lora_res, base_res) in enumerate(zip(lora_results, baseline_results)):
        frame = lora_res["frame"]
        cvs_pred = lora_res["cvs_pred"]
        cvs_gt = lora_res["cvs_gt"]
        pred_mask = lora_res.get("pred_mask")
        lora_attn = lora_res["vjepa_attn"]
        base_attn = base_res["vjepa_attn"]
        meta = lora_res["meta"]

        # Normalize frame
        if frame.max() > 1:
            frame_disp = frame.astype(np.float32) / 255.0
        else:
            frame_disp = frame

        # Process baseline attention
        if base_attn is not None:
            base_mean = base_attn[0].mean(dim=0).mean(dim=0).numpy()
            seq_len = len(base_mean)
            spatial_tokens = 256

            if seq_len >= spatial_tokens:
                temporal_bins = seq_len // spatial_tokens
                base_spatial = base_mean[:temporal_bins * spatial_tokens].reshape(temporal_bins, spatial_tokens).mean(axis=0)
                base_spatial_map = base_spatial.reshape(16, 16)
            else:
                side = int(np.sqrt(seq_len))
                base_spatial_map = base_mean.reshape(side, side) if side * side == seq_len else base_mean.reshape(1, -1)

            _, _, base_norm_ent = compute_entropy(base_mean)
            baseline_entropies.append(base_norm_ent)
        else:
            base_spatial_map = None
            baseline_entropies.append(None)

        # Process LoRA attention
        if lora_attn is not None:
            lora_mean = lora_attn[0].mean(dim=0).mean(dim=0).numpy()
            seq_len = len(lora_mean)
            spatial_tokens = 256

            if seq_len >= spatial_tokens:
                temporal_bins = seq_len // spatial_tokens
                lora_spatial = lora_mean[:temporal_bins * spatial_tokens].reshape(temporal_bins, spatial_tokens).mean(axis=0)
                lora_spatial_map = lora_spatial.reshape(16, 16)
            else:
                side = int(np.sqrt(seq_len))
                lora_spatial_map = lora_mean.reshape(side, side) if side * side == seq_len else lora_mean.reshape(1, -1)

            _, _, lora_norm_ent = compute_entropy(lora_mean)
            lora_entropies.append(lora_norm_ent)
        else:
            lora_spatial_map = None
            lora_entropies.append(None)

        # Column 1: Original frame with CVS info
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 1)
        ax.imshow(frame_disp)
        cvs_text = f"Pred: {cvs_pred[0]:.2f}, {cvs_pred[1]:.2f}, {cvs_pred[2]:.2f}\n"
        cvs_text += f"GT: {int(cvs_gt[0])}, {int(cvs_gt[1])}, {int(cvs_gt[2])}"
        ax.set_title(f"Vid {meta['video_id']}\n{cvs_text}", fontsize=9)
        ax.axis('off')

        # Column 2: Baseline attention on frame
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 2)
        ax.imshow(frame_disp)
        if base_spatial_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(base_spatial_map.astype(np.float64),
                               (h / base_spatial_map.shape[0], w / base_spatial_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ax.set_title(f"BASELINE V-JEPA\nEntropy: {baseline_entropies[row]:.1f}%", fontsize=9)
        else:
            ax.set_title("BASELINE\n(not available)", fontsize=9)
        ax.axis('off')

        # Column 3: LoRA attention on frame
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 3)
        ax.imshow(frame_disp)
        if lora_spatial_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(lora_spatial_map.astype(np.float64),
                               (h / lora_spatial_map.shape[0], w / lora_spatial_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ax.set_title(f"LoRA V-JEPA\nEntropy: {lora_entropies[row]:.1f}%", fontsize=9)
        else:
            ax.set_title("LoRA\n(not available)", fontsize=9)
        ax.axis('off')

        # Column 4: Baseline heatmap
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 4)
        if base_spatial_map is not None:
            im = ax.imshow(base_spatial_map, cmap='hot')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Baseline Heatmap", fontsize=9)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        # Column 5: LoRA heatmap
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 5)
        if lora_spatial_map is not None:
            im = ax.imshow(lora_spatial_map, cmap='hot')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("LoRA Heatmap", fontsize=9)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        # Column 6: Segmentation prediction
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 6)
        if pred_mask is not None:
            pred_colored = colorize_mask(pred_mask)
            ax.imshow(pred_colored)
            ax.set_title("Seg Prediction", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No mask", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # Add summary statistics
    valid_baseline = [e for e in baseline_entropies if e is not None]
    valid_lora = [e for e in lora_entropies if e is not None]

    summary = f"ENTROPY ANALYSIS (lower = more focused)\n"
    summary += f"Baseline V-JEPA: {np.mean(valid_baseline):.1f}% avg (expected ~98% uniform)\n"
    summary += f"LoRA V-JEPA: {np.mean(valid_lora):.1f}% avg\n"
    improvement = np.mean(valid_baseline) - np.mean(valid_lora)
    if improvement > 0:
        summary += f"LoRA IMPROVED focus by {improvement:.1f}%"
    else:
        summary += f"No significant change in attention focus"

    fig.text(0.5, 0.01, summary, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add legend
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=legend_patches, loc='lower right', ncol=5, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nAttention comparison saved to: {output_path}")
    plt.close()

    return baseline_entropies, lora_entropies


def create_entropy_analysis_figure(baseline_entropies, lora_entropies, sample_names, output_path):
    """Create entropy comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-sample comparison
    ax = axes[0]
    x = np.arange(len(sample_names))
    width = 0.35

    valid_base = [e if e is not None else 0 for e in baseline_entropies]
    valid_lora = [e if e is not None else 0 for e in lora_entropies]

    bars1 = ax.bar(x - width/2, valid_base, width, label='Baseline', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, valid_lora, width, label='LoRA', color='#4CAF50', alpha=0.8)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Uniform (100%)')
    ax.axhline(y=98, color='red', linestyle='--', alpha=0.5, label='Expected Baseline (~98%)')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Entropy (%)')
    ax.set_title('Attention Entropy: Baseline vs LoRA')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('_')[0] for s in sample_names], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)

    # Summary statistics
    ax = axes[1]
    categories = ['Baseline\nV-JEPA', 'LoRA\nV-JEPA']
    valid_base_only = [e for e in baseline_entropies if e is not None]
    valid_lora_only = [e for e in lora_entropies if e is not None]

    means = [np.mean(valid_base_only), np.mean(valid_lora_only)]
    stds = [np.std(valid_base_only), np.std(valid_lora_only)]

    colors = ['#2196F3', '#4CAF50']
    bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.8, capsize=10)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=98, color='red', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Entropy (%)')
    ax.set_title('Average Entropy Comparison')
    ax.set_ylim(0, 105)

    # Add interpretation
    improvement = means[0] - means[1]
    if improvement > 5:
        interpretation = f"LoRA SIGNIFICANTLY improved attention focus\n(entropy reduced by {improvement:.1f}%)"
        color = 'green'
    elif improvement > 0:
        interpretation = f"LoRA slightly improved attention focus\n(entropy reduced by {improvement:.1f}%)"
        color = 'orange'
    else:
        interpretation = f"No significant change in attention focus"
        color = 'gray'

    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=12, fontweight='bold', color=color,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Entropy analysis saved to: {output_path}")
    plt.close()


def create_predictions_comparison_figure(lora_results, output_path):
    """Create predictions comparison figure."""
    n_samples = len(lora_results)

    fig = plt.figure(figsize=(16, 4 * n_samples))

    for row, result in enumerate(lora_results):
        frame = result["frame"]
        cvs_pred = result["cvs_pred"]
        cvs_gt = result["cvs_gt"]
        meta = result["meta"]

        if frame.max() > 1:
            frame_disp = frame.astype(np.float32) / 255.0
        else:
            frame_disp = frame

        # Frame
        ax = fig.add_subplot(n_samples, 3, row * 3 + 1)
        ax.imshow(frame_disp)
        ax.set_title(f"Video {meta['video_id']}", fontsize=11)
        ax.axis('off')

        # Predictions bar chart
        ax = fig.add_subplot(n_samples, 3, row * 3 + 2)

        x = np.arange(3)
        width = 0.35

        gt_colors = ['#4CAF50' if g > 0.5 else '#F44336' for g in cvs_gt]
        pred_colors = ['#4CAF50' if p > 0.5 else '#FFC107' for p in cvs_pred]

        bars_gt = ax.bar(x - width/2, cvs_gt, width, label='Ground Truth', color=gt_colors, alpha=0.8)
        bars_pred = ax.bar(x + width/2, cvs_pred, width, label='LoRA Prediction', edgecolor=pred_colors,
                          color='none', linewidth=3)

        # Fill predicted bars
        for bar, pred in zip(bars_pred, cvs_pred):
            bar.set_facecolor('#2196F3')
            bar.set_alpha(0.6)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['C1\n(Triangle)', 'C2\n(Plate)', 'C3\n(Artery/Duct)'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('CVS Predictions')
        ax.legend(loc='upper right')

        # Analysis text
        ax = fig.add_subplot(n_samples, 3, row * 3 + 3)
        ax.axis('off')

        # Compute correctness
        correct = []
        for i, (gt, pred) in enumerate(zip(cvs_gt, cvs_pred)):
            gt_class = gt > 0.5
            pred_class = pred > 0.5
            status = "correct" if gt_class == pred_class else "WRONG"
            correct.append(f"C{i+1}: GT={int(gt)}, Pred={pred:.2f} ({status})")

        text = "\n".join(correct)

        # Overall accuracy
        matches = sum(1 for gt, pred in zip(cvs_gt, cvs_pred) if (gt > 0.5) == (pred > 0.5))
        text += f"\n\nAccuracy: {matches}/3"

        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Predictions comparison saved to: {output_path}")
    plt.close()


def main():
    # Configuration
    lora_checkpoint = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\results\exp10_lora\run_20260201_203545\best_model.pt"
    data_root = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
    output_dir = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\visualizations\exp10_lora_analysis")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Target samples
    target_samples = [
        "127_19475",  # Full CVS - model previously missed
        "129_68650",  # C1-only
        "122_35150",  # Negative
    ]

    # Load LoRA model
    print("\n" + "=" * 70)
    print("LOADING LoRA MODEL (53.75% mAP)")
    print("=" * 70)
    lora_model, config = load_lora_model(lora_checkpoint, device)

    # Load baseline V-JEPA
    print("\n" + "=" * 70)
    print("LOADING BASELINE V-JEPA")
    print("=" * 70)
    baseline_backbone = load_baseline_model(device)

    # Load dataset
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    dataset = MultiTaskCVSDataset(
        root_dir=data_root,
        split="val",
        num_frames=16,
        resolution=256,
        mask_resolution=64,
        augment=False,
        use_synthetic_masks=True,
    )

    # Find specific samples
    found_samples = find_specific_samples(dataset, target_samples)

    # Also find a C2-only sample
    c2_only_idx = find_c2_only_sample(dataset)
    if c2_only_idx is not None:
        sample_info = dataset.samples[c2_only_idx]
        c2_sample_id = f"{sample_info['video_id']}_{sample_info['center_frame']}"
        print(f"Found C2-only sample: {c2_sample_id}")
        found_samples[c2_sample_id] = c2_only_idx

    # Process samples
    print("\n" + "=" * 70)
    print("PROCESSING SAMPLES")
    print("=" * 70)

    lora_results = []
    baseline_results = []
    sample_names = []

    for sample_id, idx in found_samples.items():
        print(f"\nProcessing: {sample_id} (index {idx})")
        sample = dataset[idx]

        # LoRA model
        print("  Processing through LoRA model...")
        lora_result = process_sample_lora(lora_model, sample, device)
        lora_results.append(lora_result)

        # Baseline model
        print("  Processing through baseline model...")
        baseline_result = process_sample_baseline(baseline_backbone, sample, device)
        baseline_results.append(baseline_result)

        sample_names.append(sample_id)

        # Print predictions
        print(f"  GT: C1={int(sample['labels'][0])}, C2={int(sample['labels'][1])}, C3={int(sample['labels'][2])}")
        print(f"  LoRA Pred: C1={lora_result['cvs_pred'][0]:.3f}, C2={lora_result['cvs_pred'][1]:.3f}, C3={lora_result['cvs_pred'][2]:.3f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Attention comparison
    baseline_entropies, lora_entropies = create_attention_comparison_figure(
        lora_results, baseline_results,
        output_dir / "attention_comparison.png"
    )

    # 2. Entropy analysis
    create_entropy_analysis_figure(
        baseline_entropies, lora_entropies, sample_names,
        output_dir / "entropy_analysis.png"
    )

    # 3. Predictions comparison
    create_predictions_comparison_figure(
        lora_results,
        output_dir / "predictions_comparison.png"
    )

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    valid_baseline = [e for e in baseline_entropies if e is not None]
    valid_lora = [e for e in lora_entropies if e is not None]

    print(f"\nATTENTION ENTROPY (lower = more focused):")
    print(f"  Baseline V-JEPA: {np.mean(valid_baseline):.1f}% avg")
    print(f"  LoRA V-JEPA: {np.mean(valid_lora):.1f}% avg")

    improvement = np.mean(valid_baseline) - np.mean(valid_lora)
    if improvement > 5:
        print(f"\n  LoRA SIGNIFICANTLY improved attention focus by {improvement:.1f}%!")
    elif improvement > 0:
        print(f"\n  LoRA slightly improved focus by {improvement:.1f}%")
    else:
        print(f"\n  No significant change in attention focus")

    print(f"\nPREDICTIONS:")
    for i, (result, name) in enumerate(zip(lora_results, sample_names)):
        cvs_pred = result["cvs_pred"]
        cvs_gt = result["cvs_gt"]
        matches = sum(1 for gt, pred in zip(cvs_gt, cvs_pred) if (gt > 0.5) == (pred > 0.5))
        print(f"  {name}: {matches}/3 correct")
        print(f"    GT: [{int(cvs_gt[0])}, {int(cvs_gt[1])}, {int(cvs_gt[2])}]")
        print(f"    Pred: [{cvs_pred[0]:.2f}, {cvs_pred[1]:.2f}, {cvs_pred[2]:.2f}]")

    print(f"\nVisualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
