"""
V-JEPA Internal Attention Analysis for Exp8 Fine-tuned Model
=============================================================
Compare V-JEPA's internal attention BEFORE vs AFTER fine-tuning.

Key question: Did fine-tuning make V-JEPA's internal attention more focused on anatomy?
Before fine-tuning it was ~95% entropy (nearly uniform).

This script:
1. Loads the fine-tuned Exp8 model
2. Extracts V-JEPA's internal self-attention from the last transformer block
3. Extracts our attention pooling weights
4. Shows segmentation predictions
5. Computes entropy to measure focus vs uniformity
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent))

from dataset_multitask import MultiTaskCVSDataset
from model_multitask import VJEPA_MultiTask
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


def get_vjepa_attention_from_last_layer(model, pixel_values, device):
    """
    Extract attention from V-JEPA's last transformer layer.

    Hooks into Q, K projections and computes attention = softmax(QK^T / sqrt(d_k)).
    """
    captured_qkv = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            captured_qkv[name] = output.detach().cpu()
        return hook_fn

    # Find the last transformer layer's attention
    backbone = model.backbone
    last_layer_idx = -1
    target_layer = None

    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
        last_layer_idx = len(backbone.encoder.layer) - 1
        target_layer = backbone.encoder.layer[last_layer_idx].attention
        print(f"  Found attention at encoder.layer.{last_layer_idx}.attention")

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
            _ = model.backbone.get_vision_features(pixel_values_videos=pixel_values)
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


def compute_entropy(attn):
    """Compute entropy of attention distribution (measure of focus)."""
    attn_flat = attn.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
    max_entropy = np.log(len(attn_flat))
    return entropy, max_entropy, entropy / max_entropy * 100  # normalized %


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


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load fine-tuned model."""
    print(f"Loading model from: {checkpoint_path}")

    model = VJEPA_MultiTask(
        model_name=config["model"]["name"],
        unfreeze_last_n_layers=config["model"].get("unfreeze_last_n_layers", 2),
        hidden_dim=config["model"].get("hidden_dim", 1024),
        cvs_hidden=config["model"].get("cvs_hidden", 512),
        cvs_dropout=config["model"].get("cvs_dropout", 0.5),
        attention_heads=config["model"].get("attention_heads", 8),
        attention_dropout=config["model"].get("attention_dropout", 0.1),
        num_seg_classes=config["model"].get("num_seg_classes", 5),
        seg_output_size=config["model"].get("seg_output_size", 64),
        seg_dropout=config["model"].get("seg_dropout", 0.1),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Replace pooler with attention-capturing version
    model.pooler = AttentionPoolingWithWeights(model.pooler)

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def find_diverse_samples(dataset, n_samples=6):
    """Find diverse samples with masks and positive CVS labels."""
    positive_cvs = []
    seen_videos = set()

    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        if sample_info["has_any_mask"]:
            labels = sample_info["labels"]
            vid = sample_info["video_id"]
            if labels.sum() > 0:
                positive_cvs.append((i, vid, labels.copy()))

    # Select from different videos
    selected = []
    for idx, vid, labels in positive_cvs:
        if vid not in seen_videos:
            selected.append(idx)
            seen_videos.add(vid)
            if len(selected) >= n_samples:
                break

    print(f"Selected {len(selected)} samples from {len(seen_videos)} videos")
    return selected


def process_sample(model, dataset, sample_idx, device):
    """Process a single sample and extract all attention info."""
    sample = dataset[sample_idx]

    video = sample["video"]
    labels = sample["labels"]
    masks = sample["masks"]
    mask_indices = sample["mask_indices"]
    meta = sample["meta"]

    # Get middle frame
    middle_idx = len(video) // 2
    frame = video[middle_idx]

    # Find GT mask for middle frame (or first available)
    gt_mask = None
    mask_frame_idx = middle_idx

    for i, idx in enumerate(mask_indices.tolist()):
        if idx == middle_idx:
            gt_mask = masks[i].numpy()
            break

    if gt_mask is None and len(mask_indices) > 0:
        gt_mask = masks[0].numpy()
        mask_frame_idx = mask_indices[0].item()
        frame = video[mask_frame_idx]

    # Process through model
    video_batch = [video]
    pixel_values = model.process_videos(video_batch, device)

    # Extract V-JEPA internal attention
    print(f"  Extracting V-JEPA internal attention...")
    vjepa_attn = get_vjepa_attention_from_last_layer(model, pixel_values, device)

    # Forward pass for predictions
    if gt_mask is not None:
        frame_indices = torch.tensor([mask_frame_idx], device=device)
        batch_indices = torch.tensor([0], device=device)
    else:
        frame_indices = torch.tensor([], dtype=torch.long, device=device)
        batch_indices = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        # Get features
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

    # Process pooler attention: (1, heads, 1, tokens) -> (tokens,)
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


def create_visualization(results_list, output_path):
    """Create comprehensive visualization comparing attention patterns."""
    n_samples = len(results_list)

    fig = plt.figure(figsize=(24, 5 * n_samples))

    # Column layout: Frame | V-JEPA Spatial | Pooler Spatial | V-JEPA Heatmap | Pooler Heatmap | Seg Pred
    n_cols = 6

    # Track entropy values for summary
    vjepa_entropies = []
    pooler_entropies = []

    for row, result in enumerate(results_list):
        frame = result["frame"]
        gt_mask = result["gt_mask"]
        pred_mask = result["pred_mask"]
        cvs_pred = result["cvs_pred"]
        cvs_gt = result["cvs_gt"]
        vjepa_attn = result["vjepa_attn"]
        pooler_attn = result["pooler_attn"]
        meta = result["meta"]

        # Normalize frame
        if frame.max() > 1:
            frame_disp = frame.astype(np.float32) / 255.0
        else:
            frame_disp = frame

        # Process V-JEPA attention
        if vjepa_attn is not None:
            # Average across heads: (B, heads, seq, seq) -> (seq,)
            vjepa_mean = vjepa_attn[0].mean(dim=0).mean(dim=0).numpy()

            # Try to reshape to spatial grid
            seq_len = len(vjepa_mean)
            spatial_tokens = 256  # 16x16 patches

            if seq_len >= spatial_tokens:
                temporal_bins = seq_len // spatial_tokens
                vjepa_spatial = vjepa_mean[:temporal_bins * spatial_tokens].reshape(temporal_bins, spatial_tokens).mean(axis=0)
                vjepa_spatial_map = vjepa_spatial.reshape(16, 16)
            else:
                side = int(np.sqrt(seq_len))
                vjepa_spatial_map = vjepa_mean.reshape(side, side) if side * side == seq_len else vjepa_mean.reshape(1, -1)

            # Compute entropy
            entropy, max_ent, norm_ent = compute_entropy(vjepa_mean)
            vjepa_entropies.append(norm_ent)
        else:
            vjepa_spatial_map = None
            vjepa_entropies.append(None)

        # Process pooler attention
        if pooler_attn is not None:
            seq_len = len(pooler_attn)
            spatial_tokens = 256

            if seq_len >= spatial_tokens:
                temporal_bins = seq_len // spatial_tokens
                pooler_spatial = pooler_attn[:temporal_bins * spatial_tokens].reshape(temporal_bins, spatial_tokens).mean(axis=0)
                pooler_spatial_map = pooler_spatial.reshape(16, 16)
            else:
                side = int(np.sqrt(seq_len))
                pooler_spatial_map = pooler_attn.reshape(side, side) if side * side == seq_len else pooler_attn.reshape(1, -1)

            entropy, max_ent, norm_ent = compute_entropy(pooler_attn)
            pooler_entropies.append(norm_ent)
        else:
            pooler_spatial_map = None
            pooler_entropies.append(None)

        # Column 1: Original frame with CVS info
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 1)
        ax.imshow(frame_disp)
        cvs_text = f"Pred: {cvs_pred[0]:.2f}, {cvs_pred[1]:.2f}, {cvs_pred[2]:.2f}\n"
        cvs_text += f"GT: {int(cvs_gt[0])}, {int(cvs_gt[1])}, {int(cvs_gt[2])}"
        ax.set_title(f"Vid {meta['video_id']}\n{cvs_text}", fontsize=9)
        ax.axis('off')

        # Column 2: V-JEPA attention on frame
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 2)
        ax.imshow(frame_disp)
        if vjepa_spatial_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(vjepa_spatial_map.astype(np.float64),
                               (h / vjepa_spatial_map.shape[0], w / vjepa_spatial_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ent_val = vjepa_entropies[row]
            ax.set_title(f"V-JEPA Internal\nEntropy: {ent_val:.1f}%", fontsize=9)
        else:
            ax.set_title("V-JEPA Internal\n(not available)", fontsize=9)
        ax.axis('off')

        # Column 3: Pooler attention on frame
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 3)
        ax.imshow(frame_disp)
        if pooler_spatial_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(pooler_spatial_map.astype(np.float64),
                               (h / pooler_spatial_map.shape[0], w / pooler_spatial_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ent_val = pooler_entropies[row]
            ax.set_title(f"Attention Pooler\nEntropy: {ent_val:.1f}%", fontsize=9)
        else:
            ax.set_title("Attention Pooler\n(not available)", fontsize=9)
        ax.axis('off')

        # Column 4: V-JEPA spatial heatmap
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 4)
        if vjepa_spatial_map is not None:
            im = ax.imshow(vjepa_spatial_map, cmap='hot')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("V-JEPA Heatmap", fontsize=9)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        # Column 5: Pooler spatial heatmap
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 5)
        if pooler_spatial_map is not None:
            im = ax.imshow(pooler_spatial_map, cmap='hot')
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Pooler Heatmap", fontsize=9)
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

    # Add column headers
    col_titles = ["Original + CVS", "V-JEPA Internal Attn", "Attention Pooler",
                  "V-JEPA Heatmap", "Pooler Heatmap", "Seg Prediction"]
    for col, title in enumerate(col_titles):
        fig.text(0.08 + col * 0.15, 0.98, title, ha='center', fontsize=11, fontweight='bold')

    # Add summary statistics
    valid_vjepa = [e for e in vjepa_entropies if e is not None]
    valid_pooler = [e for e in pooler_entropies if e is not None]

    summary = f"ENTROPY SUMMARY (lower = more focused)\n"
    summary += f"V-JEPA Internal: {np.mean(valid_vjepa):.1f}% avg (baseline was ~95%)\n"
    summary += f"Attention Pooler: {np.mean(valid_pooler):.1f}% avg"

    fig.text(0.5, 0.01, summary, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add legend
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=legend_patches, loc='lower right', ncol=5, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    return vjepa_entropies, pooler_entropies


def main():
    # Paths
    checkpoint_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\results\exp8_finetune_multitask\run_20260131_114120\best_model.pt"
    config_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\configs\exp8_finetune_multitask.yaml"
    data_root = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
    output_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\visualizations\exp8_vjepa_internal_attention.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and model
    config = load_config(config_path)
    model = load_model(checkpoint_path, config, device)

    # Load dataset
    print("\nLoading validation dataset...")
    dataset = MultiTaskCVSDataset(
        root_dir=data_root,
        split="val",
        num_frames=config["dataset"]["num_frames"],
        resolution=config["dataset"]["resolution"],
        mask_resolution=config["dataset"].get("mask_resolution", 64),
        augment=False,
        use_synthetic_masks=True,
    )

    # Find diverse samples
    sample_indices = find_diverse_samples(dataset, n_samples=6)

    # Process samples
    print("\nProcessing samples...")
    results_list = []
    for i, sample_idx in enumerate(sample_indices):
        print(f"\nSample {i+1}/{len(sample_indices)} (idx={sample_idx})")
        result = process_sample(model, dataset, sample_idx, device)
        results_list.append(result)

    # Create visualization
    print("\nGenerating visualization...")
    vjepa_ent, pooler_ent = create_visualization(results_list, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ATTENTION ENTROPY ANALYSIS")
    print("=" * 70)
    print("Entropy measures how uniform/focused attention is:")
    print("  100% = completely uniform (no focus)")
    print("  Lower % = more focused on specific regions")
    print("-" * 70)
    print(f"BASELINE (before fine-tuning): ~95% entropy (nearly uniform)")
    print("-" * 70)

    valid_vjepa = [e for e in vjepa_ent if e is not None]
    valid_pooler = [e for e in pooler_ent if e is not None]

    print(f"V-JEPA Internal (fine-tuned):")
    print(f"  Average entropy: {np.mean(valid_vjepa):.1f}%")
    print(f"  Range: {np.min(valid_vjepa):.1f}% - {np.max(valid_vjepa):.1f}%")

    print(f"\nAttention Pooler:")
    print(f"  Average entropy: {np.mean(valid_pooler):.1f}%")
    print(f"  Range: {np.min(valid_pooler):.1f}% - {np.max(valid_pooler):.1f}%")

    print("-" * 70)
    baseline_entropy = 95.0
    improvement = baseline_entropy - np.mean(valid_vjepa)
    if improvement > 5:
        print(f"RESULT: Fine-tuning IMPROVED V-JEPA focus by {improvement:.1f}%")
    elif improvement > 0:
        print(f"RESULT: Fine-tuning slightly improved focus by {improvement:.1f}%")
    else:
        print(f"RESULT: No significant change in V-JEPA attention focus")
    print("=" * 70)

    print("\nDone!")


if __name__ == "__main__":
    main()
