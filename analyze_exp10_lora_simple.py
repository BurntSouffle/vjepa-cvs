"""
Exp10 LoRA Model Analysis (Simplified - No PEFT required)
=========================================================
Analyze our best model (53.75% mAP) by loading state dict directly.

This version bypasses PEFT library requirements by:
1. Loading the model architecture components separately
2. Extracting attention directly from Q/K projections

Key analysis:
- V-JEPA internal attention BEFORE vs AFTER LoRA
- Entropy comparison (focus vs uniformity)
- Predictions on specific samples
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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


def get_vjepa_attention_from_backbone(backbone, pixel_values, device):
    """
    Extract attention from V-JEPA's last transformer layer.
    Works with both baseline and PEFT-wrapped models.
    """
    captured_qkv = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            captured_qkv[name] = output.detach().cpu()
        return hook_fn

    # Navigate through possible wrappers
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

    # Register hooks - handle both regular and LoRA-wrapped linear layers
    hooks = []

    # For PEFT models, the actual linear layer might be wrapped
    query_module = target_layer.query if hasattr(target_layer, 'query') else None
    key_module = target_layer.key if hasattr(target_layer, 'key') else None

    # Try to find the actual linear layer if wrapped
    if query_module is not None:
        if hasattr(query_module, 'base_layer'):
            query_module = query_module.base_layer
        hooks.append(query_module.register_forward_hook(make_hook('query')))

    if key_module is not None:
        if hasattr(key_module, 'base_layer'):
            key_module = key_module.base_layer
        hooks.append(key_module.register_forward_hook(make_hook('key')))

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


class AttentionPooling(nn.Module):
    """Learnable attention pooling over token dimension."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, self.attention_weights = self.attention(
            query, x, x, need_weights=True, average_attn_weights=False
        )
        return self.norm(attn_out.squeeze(1))


class LightweightSegDecoder(nn.Module):
    """Lightweight segmentation decoder for V-JEPA features."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_classes: int = 5,
        output_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_size = output_size

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        frame_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        num_temporal_bins: int = 8,
        spatial_size: int = 16,
    ) -> torch.Tensor:
        if len(frame_indices) == 0:
            return torch.zeros(0, self.num_classes, self.output_size, self.output_size,
                             device=features.device)

        B, num_tokens, D = features.shape
        spatial_tokens = spatial_size * spatial_size

        temporal_indices = frame_indices // 2

        frame_features = []
        for i, (batch_idx, temp_idx) in enumerate(zip(batch_indices, temporal_indices)):
            start_idx = temp_idx * spatial_tokens
            end_idx = start_idx + spatial_tokens

            if end_idx > num_tokens:
                end_idx = num_tokens
                start_idx = max(0, end_idx - spatial_tokens)

            spatial_feats = features[batch_idx, start_idx:end_idx]

            if spatial_feats.shape[0] < spatial_tokens:
                padding = torch.zeros(spatial_tokens - spatial_feats.shape[0], D,
                                     device=features.device)
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)

            frame_features.append(spatial_feats)

        frame_features = torch.stack(frame_features, dim=0)
        frame_features = self.proj(frame_features)

        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()

        logits = self.decoder(frame_features)

        return logits


class LoRAModelForInference(nn.Module):
    """
    Wrapper to load LoRA-trained model for inference without PEFT library.
    Loads backbone + heads from checkpoint state dict.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})

        self.config = config
        model_cfg = config.get("model", {})

        # Load backbone with PEFT if available, otherwise skip
        print("Loading V-JEPA backbone...")
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            model_cfg.get("name", "facebook/vjepa2-vitl-fpc16-256-ssv2")
        )
        self.backbone = self.backbone.float()

        # Try to apply LoRA weights if PEFT is available
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = config.get("lora", {})

            # Freeze base
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Find target modules
            target_modules = []
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear) and ('query' in name or 'value' in name):
                    target_modules.append(name)

            if target_modules:
                lora_config = LoraConfig(
                    r=lora_cfg.get("r", 16),
                    lora_alpha=lora_cfg.get("lora_alpha", 32),
                    target_modules=target_modules,
                    lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_config)
                print(f"Applied LoRA to {len(target_modules)} modules")

        except ImportError:
            print("PEFT not available - using base backbone without LoRA weights")
            print("Note: Attention patterns will be from base model, not fine-tuned")

        hidden_dim = model_cfg.get("hidden_dim", 1024)

        # Create heads
        self.pooler = AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=model_cfg.get("attention_heads", 8),
            dropout=model_cfg.get("attention_dropout", 0.1),
        )

        self.cvs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_cfg.get("cvs_hidden", 512)),
            nn.GELU(),
            nn.Dropout(model_cfg.get("cvs_dropout", 0.5)),
            nn.Linear(model_cfg.get("cvs_hidden", 512), 3),
        )

        self.seg_head = LightweightSegDecoder(
            hidden_dim=hidden_dim,
            num_classes=model_cfg.get("num_seg_classes", 5),
            output_size=model_cfg.get("seg_output_size", 64),
            dropout=model_cfg.get("seg_dropout", 0.1),
        )

        # Load state dict (partial - just the parts we created)
        self._load_partial_state_dict(state_dict)

        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best mAP: {checkpoint.get('best_metric', 0)*100:.2f}%")

    def _load_partial_state_dict(self, state_dict):
        """Load matching keys from state dict."""
        own_state = self.state_dict()
        loaded = 0
        skipped = 0

        for name, param in state_dict.items():
            if name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
                else:
                    print(f"  Shape mismatch for {name}: {own_state[name].shape} vs {param.shape}")
                    skipped += 1
            else:
                # Try without backbone prefix
                alt_name = name.replace("backbone.base_model.model.", "backbone.")
                if alt_name in own_state and own_state[alt_name].shape == param.shape:
                    own_state[alt_name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1

        print(f"  Loaded {loaded} parameters, skipped {skipped}")

    def process_videos(self, videos, device):
        """Process videos for V-JEPA input."""
        if isinstance(videos, torch.Tensor):
            video_tensor = videos
        elif isinstance(videos, list):
            video_tensor = torch.from_numpy(np.stack(videos))
        else:
            raise ValueError(f"Unsupported type: {type(videos)}")

        if video_tensor.dim() == 5:
            B, T, H, W, C = video_tensor.shape
            if C == 3:
                video_tensor = video_tensor.permute(0, 1, 4, 2, 3)

        video_tensor = video_tensor.float() / 255.0 if video_tensor.max() > 1 else video_tensor.float()
        video_tensor = video_tensor.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

        return (video_tensor - mean) / std

    def forward(self, pixel_values, frame_indices=None, batch_indices=None):
        features = self.backbone.get_vision_features(pixel_values_videos=pixel_values)
        features = features.float()

        pooled = self.pooler(features)
        cvs_logits = self.cvs_head(pooled)

        result = {"cvs_logits": cvs_logits, "features": features}

        if frame_indices is not None and batch_indices is not None and len(frame_indices) > 0:
            seg_logits = self.seg_head(features, frame_indices, batch_indices)
            result["seg_logits"] = seg_logits

        return result


def load_baseline_model(device: torch.device):
    """Load baseline V-JEPA model (no fine-tuning)."""
    print("Loading baseline V-JEPA model...")
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


def find_c2_only_sample(dataset) -> Optional[Tuple[int, str]]:
    """Find a sample with only C2 positive."""
    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        labels = sample_info["labels"]

        if labels[0] == 0 and labels[1] == 1 and labels[2] == 0:
            vid = sample_info["video_id"]
            frame = sample_info["center_frame"]
            return i, f"{vid}_{frame}"

    return None, None


def process_sample(model, sample, device, extract_attention=True):
    """Process sample through model."""
    video = sample["video"]
    labels = sample["labels"]
    masks = sample["masks"]
    mask_indices = sample["mask_indices"]
    meta = sample["meta"]

    middle_idx = len(video) // 2
    frame = video[middle_idx]

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

    # Extract attention
    vjepa_attn = None
    if extract_attention:
        vjepa_attn = get_vjepa_attention_from_backbone(model.backbone, pixel_values, device)

    # Forward pass
    if gt_mask is not None:
        frame_indices = torch.tensor([mask_frame_idx], device=device)
        batch_indices = torch.tensor([0], device=device)
    else:
        frame_indices = torch.tensor([], dtype=torch.long, device=device)
        batch_indices = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(pixel_values, frame_indices, batch_indices)

        cvs_pred = torch.sigmoid(outputs["cvs_logits"][0]).cpu().numpy()
        pooler_attn = model.pooler.attention_weights
        if pooler_attn is not None:
            pooler_attn = pooler_attn.squeeze(0).squeeze(1).mean(dim=0).cpu().numpy()

        pred_mask = None
        if "seg_logits" in outputs and outputs["seg_logits"].shape[0] > 0:
            pred_mask = outputs["seg_logits"][0].argmax(dim=0).cpu().numpy()

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


def process_baseline_sample(backbone, sample, device):
    """Process sample through baseline backbone."""
    video = sample["video"]
    labels = sample["labels"]

    middle_idx = len(video) // 2
    frame = video[middle_idx]

    # Normalize video
    video_tensor = torch.from_numpy(np.stack([video]))
    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    pixel_values = ((video_tensor - mean) / std).to(device)

    vjepa_attn = get_vjepa_attention_from_backbone(backbone, pixel_values, device)

    return {
        "frame": frame,
        "cvs_gt": labels.numpy(),
        "vjepa_attn": vjepa_attn,
    }


def create_attention_comparison_figure(lora_results, baseline_results, sample_names, output_path):
    """Create figure comparing LoRA vs baseline attention."""
    n_samples = len(lora_results)

    fig = plt.figure(figsize=(20, 5 * n_samples))
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

        if frame.max() > 1:
            frame_disp = frame.astype(np.float32) / 255.0
        else:
            frame_disp = frame

        # Process baseline attention
        base_spatial_map = None
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
            baseline_entropies.append(None)

        # Process LoRA attention
        lora_spatial_map = None
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
            lora_entropies.append(None)

        # Column 1: Original frame
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 1)
        ax.imshow(frame_disp)
        cvs_text = f"Pred: {cvs_pred[0]:.2f}, {cvs_pred[1]:.2f}, {cvs_pred[2]:.2f}\n"
        cvs_text += f"GT: {int(cvs_gt[0])}, {int(cvs_gt[1])}, {int(cvs_gt[2])}"
        ax.set_title(f"Vid {meta['video_id']}\n{cvs_text}", fontsize=9)
        ax.axis('off')

        # Column 2: Baseline attention
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

        # Column 3: LoRA attention
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

        # Column 6: Segmentation
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 6)
        if pred_mask is not None:
            pred_colored = colorize_mask(pred_mask)
            ax.imshow(pred_colored)
            ax.set_title("Seg Prediction", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No mask", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # Summary
    valid_baseline = [e for e in baseline_entropies if e is not None]
    valid_lora = [e for e in lora_entropies if e is not None]

    if valid_baseline and valid_lora:
        summary = f"ENTROPY ANALYSIS (lower = more focused)\n"
        summary += f"Baseline V-JEPA: {np.mean(valid_baseline):.1f}% avg\n"
        summary += f"LoRA V-JEPA: {np.mean(valid_lora):.1f}% avg\n"
        improvement = np.mean(valid_baseline) - np.mean(valid_lora)
        if improvement > 0:
            summary += f"LoRA IMPROVED focus by {improvement:.1f}%"
        else:
            summary += f"No significant change"

        fig.text(0.5, 0.01, summary, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Legend
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=legend_patches, loc='lower right', ncol=5, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
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

    ax.bar(x - width/2, valid_base, width, label='Baseline', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, valid_lora, width, label='LoRA', color='#4CAF50', alpha=0.8)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Uniform (100%)')
    ax.axhline(y=98, color='red', linestyle='--', alpha=0.5, label='Expected Baseline (~98%)')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Entropy (%)')
    ax.set_title('Attention Entropy: Baseline vs LoRA')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('_')[0] for s in sample_names], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)

    # Summary
    ax = axes[1]
    valid_base_only = [e for e in baseline_entropies if e is not None]
    valid_lora_only = [e for e in lora_entropies if e is not None]

    if valid_base_only and valid_lora_only:
        categories = ['Baseline\nV-JEPA', 'LoRA\nV-JEPA']
        means = [np.mean(valid_base_only), np.mean(valid_lora_only)]
        stds = [np.std(valid_base_only), np.std(valid_lora_only)]

        colors = ['#2196F3', '#4CAF50']
        bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.8, capsize=10)

        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=98, color='red', linestyle='--', alpha=0.5)

        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Entropy (%)')
        ax.set_title('Average Entropy Comparison')
        ax.set_ylim(0, 105)

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


def create_predictions_comparison_figure(lora_results, sample_names, output_path):
    """Create predictions comparison figure."""
    n_samples = len(lora_results)

    fig = plt.figure(figsize=(16, 4 * n_samples))

    for row, (result, name) in enumerate(zip(lora_results, sample_names)):
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
        ax.set_title(f"Video {meta['video_id']}\n{name}", fontsize=11)
        ax.axis('off')

        # Predictions
        ax = fig.add_subplot(n_samples, 3, row * 3 + 2)
        x = np.arange(3)
        width = 0.35

        gt_colors = ['#4CAF50' if g > 0.5 else '#F44336' for g in cvs_gt]
        ax.bar(x - width/2, cvs_gt, width, label='Ground Truth', color=gt_colors, alpha=0.8)
        ax.bar(x + width/2, cvs_pred, width, label='LoRA Prediction', color='#2196F3', alpha=0.6)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['C1\n(Triangle)', 'C2\n(Plate)', 'C3\n(Art/Duct)'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('CVS Predictions')
        ax.legend(loc='upper right')

        # Analysis
        ax = fig.add_subplot(n_samples, 3, row * 3 + 3)
        ax.axis('off')

        correct = []
        for i, (gt, pred) in enumerate(zip(cvs_gt, cvs_pred)):
            gt_class = gt > 0.5
            pred_class = pred > 0.5
            status = "correct" if gt_class == pred_class else "WRONG"
            correct.append(f"C{i+1}: GT={int(gt)}, Pred={pred:.2f} ({status})")

        text = "\n".join(correct)
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
    lora_model = LoRAModelForInference(lora_checkpoint, device)
    lora_model = lora_model.to(device)
    lora_model.eval()

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

    # Find samples
    found_samples = find_specific_samples(dataset, target_samples)

    # Find C2-only sample
    c2_idx, c2_id = find_c2_only_sample(dataset)
    if c2_idx is not None:
        print(f"Found C2-only sample: {c2_id}")
        found_samples[c2_id] = c2_idx

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
        lora_result = process_sample(lora_model, sample, device)
        lora_results.append(lora_result)

        # Baseline model
        print("  Processing through baseline model...")
        baseline_result = process_baseline_sample(baseline_backbone, sample, device)
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
        lora_results, baseline_results, sample_names,
        output_dir / "attention_comparison.png"
    )

    # 2. Entropy analysis
    create_entropy_analysis_figure(
        baseline_entropies, lora_entropies, sample_names,
        output_dir / "entropy_analysis.png"
    )

    # 3. Predictions comparison
    create_predictions_comparison_figure(
        lora_results, sample_names,
        output_dir / "predictions_comparison.png"
    )

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    valid_baseline = [e for e in baseline_entropies if e is not None]
    valid_lora = [e for e in lora_entropies if e is not None]

    if valid_baseline and valid_lora:
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
    for result, name in zip(lora_results, sample_names):
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
