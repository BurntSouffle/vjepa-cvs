"""
Exp10b LoRA Attention Analysis
==============================
Compare attention patterns across ALL LoRA experiments:
- Baseline V-JEPA (no LoRA): ~98.4% entropy
- Exp10a (r=16, q+v): ~97.9% entropy
- Exp10b (r=32, q+k+v): ???

Key question: Did adding k_proj and higher rank change WHERE the model attends?
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent))

from dataset_multitask import MultiTaskCVSDataset


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


def compute_entropy(attn):
    """Compute normalized entropy of attention distribution."""
    attn_flat = attn.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
    max_entropy = np.log(len(attn_flat))
    return entropy / max_entropy * 100  # normalized %


def get_vjepa_attention_from_backbone(backbone, pixel_values, device, include_k_proj=False):
    """
    Extract attention from V-JEPA's transformer layers.
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

    # Find attention layers - check last few layers
    hooks = []
    target_layers = []

    if hasattr(actual_backbone, 'encoder') and hasattr(actual_backbone.encoder, 'layer'):
        num_layers = len(actual_backbone.encoder.layer)
        # Get last 3 layers for analysis
        for layer_idx in [num_layers - 1, num_layers - 2, num_layers - 3]:
            if layer_idx >= 0:
                target_layer = actual_backbone.encoder.layer[layer_idx].attention
                target_layers.append((layer_idx, target_layer))

    if not target_layers:
        print("  Could not find attention layer structure")
        return None, {}

    # Register hooks for Q, K projections on last layer
    last_layer_idx, last_layer = target_layers[0]

    query_module = last_layer.query if hasattr(last_layer, 'query') else None
    key_module = last_layer.key if hasattr(last_layer, 'key') else None

    if query_module is not None:
        # Handle LoRA-wrapped layers
        if hasattr(query_module, 'base_layer'):
            query_module = query_module.base_layer
        hooks.append(query_module.register_forward_hook(make_hook('query')))

    if key_module is not None:
        if hasattr(key_module, 'base_layer'):
            key_module = key_module.base_layer
        hooks.append(key_module.register_forward_hook(make_hook('key')))

    if not hooks:
        print("  No Q/K projections found")
        return None, {}

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

        # Also return raw Q, K for analysis
        info = {
            'Q_norm': torch.norm(Q).item(),
            'K_norm': torch.norm(K).item(),
            'seq_len': seq_len,
            'num_heads': num_heads,
        }

        return attn_weights, info

    return None, {}


class AttentionPooling(nn.Module):
    """Learnable attention pooling."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
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
    """Lightweight segmentation decoder."""

    def __init__(self, hidden_dim: int = 1024, num_classes: int = 5,
                 output_size: int = 64, dropout: float = 0.1):
        super().__init__()
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

    def forward(self, features, frame_indices, batch_indices,
                num_temporal_bins=8, spatial_size=16):
        if len(frame_indices) == 0:
            return torch.zeros(0, 5, self.output_size, self.output_size, device=features.device)

        B, num_tokens, D = features.shape
        spatial_tokens = spatial_size * spatial_size
        temporal_indices = frame_indices // 2

        frame_features = []
        for batch_idx, temp_idx in zip(batch_indices, temporal_indices):
            start_idx = temp_idx * spatial_tokens
            end_idx = start_idx + spatial_tokens
            if end_idx > num_tokens:
                end_idx = num_tokens
                start_idx = max(0, end_idx - spatial_tokens)

            spatial_feats = features[batch_idx, start_idx:end_idx]
            if spatial_feats.shape[0] < spatial_tokens:
                padding = torch.zeros(spatial_tokens - spatial_feats.shape[0], D, device=features.device)
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)
            frame_features.append(spatial_feats)

        frame_features = torch.stack(frame_features, dim=0)
        frame_features = self.proj(frame_features)

        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()

        return self.decoder(frame_features)


class LoRAModelWrapper(nn.Module):
    """Wrapper for LoRA-trained models."""

    def __init__(self, checkpoint_path: str, device: torch.device, model_name: str = ""):
        super().__init__()
        self.model_name = model_name

        print(f"Loading {model_name}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})

        self.config = config
        model_cfg = config.get("model", {})
        lora_cfg = config.get("lora", {})

        # Store LoRA info
        self.lora_r = lora_cfg.get("r", 16)
        self.lora_alpha = lora_cfg.get("lora_alpha", 32)
        self.target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

        # Load backbone
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            model_cfg.get("name", "facebook/vjepa2-vitl-fpc16-256-ssv2")
        ).float()

        # Apply LoRA if available
        try:
            from peft import LoraConfig, get_peft_model

            for param in self.backbone.parameters():
                param.requires_grad = False

            # Find target modules
            target_module_names = []
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    # Check if this module matches any target
                    for target in self.target_modules:
                        if target in name or target.replace("_proj", "") in name:
                            target_module_names.append(name)
                            break

            if target_module_names:
                lora_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=target_module_names,
                    lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_config)
                print(f"  Applied LoRA r={self.lora_r} to {len(target_module_names)} modules")
                print(f"  Target modules: {self.target_modules}")

        except ImportError:
            print("  PEFT not available - using base backbone")

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

        # Load state dict
        self._load_state_dict(state_dict)

        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best mAP: {checkpoint.get('best_metric', 0)*100:.2f}%")

    def _load_state_dict(self, state_dict):
        """Load matching keys from state dict."""
        own_state = self.state_dict()
        loaded = 0

        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                # Try alternative names
                alt_name = name.replace("backbone.base_model.model.", "backbone.")
                if alt_name in own_state and own_state[alt_name].shape == param.shape:
                    own_state[alt_name].copy_(param)
                    loaded += 1

        print(f"  Loaded {loaded} parameters")

    def process_videos(self, videos, device):
        """Process videos for V-JEPA input."""
        if isinstance(videos, torch.Tensor):
            video_tensor = videos
        else:
            video_tensor = torch.from_numpy(np.stack(videos))

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
        features = self.backbone.get_vision_features(pixel_values_videos=pixel_values).float()
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
    backbone = backbone.float().to(device)
    backbone.eval()

    return backbone


def find_samples(dataset, target_ids: List[str]) -> Dict[str, int]:
    """Find specific samples by video_id pattern."""
    found = {}

    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        vid = sample_info["video_id"]
        frame = sample_info["center_frame"]
        sample_id = f"{vid}_{frame}"

        if sample_id in target_ids:
            found[sample_id] = i

    return found


def find_c2_only_samples(dataset, n_samples: int = 3) -> List[Tuple[int, str]]:
    """Find samples with only C2 positive."""
    found = []

    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        labels = sample_info["labels"]

        if labels[0] == 0 and labels[1] == 1 and labels[2] == 0:
            vid = sample_info["video_id"]
            frame = sample_info["center_frame"]
            found.append((i, f"{vid}_{frame}"))

            if len(found) >= n_samples:
                break

    return found


def process_sample_with_model(model, sample, device):
    """Process sample through a model."""
    video = sample["video"]
    labels = sample["labels"]
    meta = sample["meta"]
    masks = sample["masks"]
    mask_indices = sample["mask_indices"]

    middle_idx = len(video) // 2
    frame = video[middle_idx]

    # Get mask if available
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
    vjepa_attn, attn_info = get_vjepa_attention_from_backbone(model.backbone, pixel_values, device)

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
        "attn_info": attn_info,
        "meta": meta,
    }


def process_baseline_sample(backbone, sample, device):
    """Process sample through baseline backbone."""
    video = sample["video"]
    labels = sample["labels"]
    meta = sample["meta"]

    middle_idx = len(video) // 2
    frame = video[middle_idx]

    # Normalize video
    video_tensor = torch.from_numpy(np.stack([video]))
    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    pixel_values = ((video_tensor - mean) / std).to(device)

    vjepa_attn, attn_info = get_vjepa_attention_from_backbone(backbone, pixel_values, device)

    return {
        "frame": frame,
        "cvs_gt": labels.numpy(),
        "vjepa_attn": vjepa_attn,
        "attn_info": attn_info,
        "meta": meta,
    }


def attention_to_spatial_map(attn_weights, spatial_size=16):
    """Convert attention weights to spatial heatmap."""
    if attn_weights is None:
        return None, None

    # Average over heads and aggregate across tokens
    mean_attn = attn_weights[0].mean(dim=0).mean(dim=0).numpy()

    seq_len = len(mean_attn)
    spatial_tokens = spatial_size * spatial_size  # 256

    if seq_len >= spatial_tokens:
        temporal_bins = seq_len // spatial_tokens
        spatial = mean_attn[:temporal_bins * spatial_tokens].reshape(temporal_bins, spatial_tokens).mean(axis=0)
        spatial_map = spatial.reshape(spatial_size, spatial_size)
    else:
        side = int(np.sqrt(seq_len))
        if side * side == seq_len:
            spatial_map = mean_attn.reshape(side, side)
        else:
            spatial_map = mean_attn.reshape(1, -1)

    entropy = compute_entropy(mean_attn)

    return spatial_map, entropy


def create_all_lora_comparison_figure(
    baseline_results, exp10a_results, exp10b_results,
    sample_names, output_path
):
    """Create figure comparing all three models."""
    n_samples = len(sample_names)

    fig = plt.figure(figsize=(24, 5 * n_samples))
    n_cols = 7

    entropies = {"baseline": [], "exp10a": [], "exp10b": []}

    for row, name in enumerate(sample_names):
        base_res = baseline_results[row]
        a_res = exp10a_results[row]
        b_res = exp10b_results[row]

        frame = b_res["frame"]
        if frame.max() > 1:
            frame_disp = frame.astype(np.float32) / 255.0
        else:
            frame_disp = frame

        # Get attention maps
        base_map, base_ent = attention_to_spatial_map(base_res["vjepa_attn"])
        a_map, a_ent = attention_to_spatial_map(a_res["vjepa_attn"])
        b_map, b_ent = attention_to_spatial_map(b_res["vjepa_attn"])

        entropies["baseline"].append(base_ent)
        entropies["exp10a"].append(a_ent)
        entropies["exp10b"].append(b_ent)

        # Column 1: Original frame with GT and predictions
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 1)
        ax.imshow(frame_disp)

        cvs_gt = b_res["cvs_gt"]
        cvs_pred_a = a_res["cvs_pred"]
        cvs_pred_b = b_res["cvs_pred"]

        title = f"Vid {b_res['meta']['video_id']}\n"
        title += f"GT: [{int(cvs_gt[0])},{int(cvs_gt[1])},{int(cvs_gt[2])}]"
        ax.set_title(title, fontsize=9)
        ax.axis('off')

        # Column 2: Baseline attention
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 2)
        ax.imshow(frame_disp)
        if base_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(base_map.astype(np.float64),
                               (h / base_map.shape[0], w / base_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ax.set_title(f"BASELINE\n{base_ent:.1f}% entropy", fontsize=9)
        else:
            ax.set_title("BASELINE\nN/A", fontsize=9)
        ax.axis('off')

        # Column 3: Exp10a attention
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 3)
        ax.imshow(frame_disp)
        if a_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(a_map.astype(np.float64),
                               (h / a_map.shape[0], w / a_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ax.set_title(f"Exp10a (r=16)\n{a_ent:.1f}% entropy", fontsize=9)
        else:
            ax.set_title("Exp10a\nN/A", fontsize=9)
        ax.axis('off')

        # Column 4: Exp10b attention
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 4)
        ax.imshow(frame_disp)
        if b_map is not None:
            h, w = frame_disp.shape[:2]
            attn_resized = zoom(b_map.astype(np.float64),
                               (h / b_map.shape[0], w / b_map.shape[1]), order=1)
            ax.imshow(attn_resized, cmap='hot', alpha=0.6)
            ax.set_title(f"Exp10b (r=32+k)\n{b_ent:.1f}% entropy", fontsize=9)
        else:
            ax.set_title("Exp10b\nN/A", fontsize=9)
        ax.axis('off')

        # Column 5: Heatmap comparison (baseline vs exp10b)
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 5)
        if base_map is not None and b_map is not None:
            diff_map = b_map - base_map
            im = ax.imshow(diff_map, cmap='RdBu_r', vmin=-0.01, vmax=0.01)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Diff: Exp10b - Base\n(blue=less, red=more)", fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        # Column 6: Predictions comparison
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 6)
        x = np.arange(3)
        width = 0.25

        ax.bar(x - width, cvs_gt, width, label='GT', color='#4CAF50', alpha=0.8)
        ax.bar(x, cvs_pred_a, width, label='10a', color='#2196F3', alpha=0.7)
        ax.bar(x + width, cvs_pred_b, width, label='10b', color='#FF5722', alpha=0.7)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        ax.set_ylim(0, 1.1)
        ax.set_title('Predictions', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')

        # Column 7: Analysis text
        ax = fig.add_subplot(n_samples, n_cols, row * n_cols + 7)
        ax.axis('off')

        # Compute accuracy for each model
        def compute_acc(pred, gt):
            return sum(1 for p, g in zip(pred, gt) if (p > 0.5) == (g > 0.5))

        acc_a = compute_acc(cvs_pred_a, cvs_gt)
        acc_b = compute_acc(cvs_pred_b, cvs_gt)

        text = f"Exp10a: {acc_a}/3 correct\n"
        text += f"  C1={cvs_pred_a[0]:.2f}, C2={cvs_pred_a[1]:.2f}, C3={cvs_pred_a[2]:.2f}\n\n"
        text += f"Exp10b: {acc_b}/3 correct\n"
        text += f"  C1={cvs_pred_b[0]:.2f}, C2={cvs_pred_b[1]:.2f}, C3={cvs_pred_b[2]:.2f}\n\n"

        if acc_b > acc_a:
            text += "Exp10b BETTER"
        elif acc_b < acc_a:
            text += "Exp10a better"
        else:
            text += "Same accuracy"

        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Summary at bottom
    valid_base = [e for e in entropies["baseline"] if e is not None]
    valid_a = [e for e in entropies["exp10a"] if e is not None]
    valid_b = [e for e in entropies["exp10b"] if e is not None]

    if valid_base and valid_a and valid_b:
        summary = "ENTROPY COMPARISON (lower = more focused attention)\n"
        summary += f"Baseline V-JEPA: {np.mean(valid_base):.1f}%  |  "
        summary += f"Exp10a (r=16, q+v): {np.mean(valid_a):.1f}%  |  "
        summary += f"Exp10b (r=32, q+k+v): {np.mean(valid_b):.1f}%"

        fig.text(0.5, 0.01, summary, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    return entropies


def create_entropy_comparison_figure(entropies, sample_names, output_path):
    """Create entropy comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-sample comparison
    ax = axes[0]
    x = np.arange(len(sample_names))
    width = 0.25

    base = [e if e is not None else 0 for e in entropies["baseline"]]
    a = [e if e is not None else 0 for e in entropies["exp10a"]]
    b = [e if e is not None else 0 for e in entropies["exp10b"]]

    ax.bar(x - width, base, width, label='Baseline', color='#9E9E9E', alpha=0.8)
    ax.bar(x, a, width, label='Exp10a (r=16)', color='#2196F3', alpha=0.8)
    ax.bar(x + width, b, width, label='Exp10b (r=32+k)', color='#FF5722', alpha=0.8)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Uniform (100%)')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Entropy (%)')
    ax.set_title('Attention Entropy Per Sample')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('_')[0] for s in sample_names], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(90, 102)

    # Summary comparison
    ax = axes[1]

    valid_base = [e for e in entropies["baseline"] if e is not None]
    valid_a = [e for e in entropies["exp10a"] if e is not None]
    valid_b = [e for e in entropies["exp10b"] if e is not None]

    if valid_base and valid_a and valid_b:
        categories = ['Baseline\nV-JEPA', 'Exp10a\n(r=16, q+v)', 'Exp10b\n(r=32, q+k+v)']
        means = [np.mean(valid_base), np.mean(valid_a), np.mean(valid_b)]
        stds = [np.std(valid_base), np.std(valid_a), np.std(valid_b)]

        colors = ['#9E9E9E', '#2196F3', '#FF5722']
        bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.8, capsize=10)

        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Entropy (%)')
        ax.set_title('Average Entropy Comparison')
        ax.set_ylim(90, 102)

        # Analysis
        improvement_a = means[0] - means[1]
        improvement_b = means[0] - means[2]
        improvement_b_over_a = means[1] - means[2]

        analysis = f"Baseline → Exp10a: {improvement_a:+.1f}%\n"
        analysis += f"Baseline → Exp10b: {improvement_b:+.1f}%\n"
        analysis += f"Exp10a → Exp10b: {improvement_b_over_a:+.1f}%"

        fig.text(0.5, 0.02, analysis, ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_c2_detection_analysis(exp10a_results, exp10b_results, sample_names, output_path):
    """Create C2 detection comparison figure."""
    n_samples = len(sample_names)

    fig = plt.figure(figsize=(16, 4 * n_samples))

    for row, name in enumerate(sample_names):
        a_res = exp10a_results[row]
        b_res = exp10b_results[row]

        frame = b_res["frame"]
        if frame.max() > 1:
            frame_disp = frame.astype(np.float32) / 255.0
        else:
            frame_disp = frame

        cvs_gt = b_res["cvs_gt"]
        cvs_pred_a = a_res["cvs_pred"]
        cvs_pred_b = b_res["cvs_pred"]

        # Column 1: Frame
        ax = fig.add_subplot(n_samples, 3, row * 3 + 1)
        ax.imshow(frame_disp)
        ax.set_title(f"{name}\nGT: C1={int(cvs_gt[0])}, C2={int(cvs_gt[1])}, C3={int(cvs_gt[2])}", fontsize=10)
        ax.axis('off')

        # Column 2: C2 comparison bar chart
        ax = fig.add_subplot(n_samples, 3, row * 3 + 2)

        x = [0, 1, 2]
        labels = ['GT', 'Exp10a', 'Exp10b']
        values = [cvs_gt[1], cvs_pred_a[1], cvs_pred_b[1]]
        colors = ['#4CAF50', '#2196F3', '#FF5722']

        bars = ax.bar(x, values, color=colors, alpha=0.8)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('C2 Score')
        ax.set_title('C2 (Cystic Plate) Detection')
        ax.legend()

        # Column 3: Analysis
        ax = fig.add_subplot(n_samples, 3, row * 3 + 3)
        ax.axis('off')

        c2_gt = cvs_gt[1] > 0.5
        c2_pred_a = cvs_pred_a[1] > 0.5
        c2_pred_b = cvs_pred_b[1] > 0.5

        text = f"C2 Ground Truth: {'POSITIVE' if c2_gt else 'NEGATIVE'}\n\n"
        text += f"Exp10a prediction: {cvs_pred_a[1]:.3f}\n"
        text += f"  Result: {'CORRECT' if c2_pred_a == c2_gt else 'WRONG'}\n\n"
        text += f"Exp10b prediction: {cvs_pred_b[1]:.3f}\n"
        text += f"  Result: {'CORRECT' if c2_pred_b == c2_gt else 'WRONG'}\n\n"

        improvement = cvs_pred_b[1] - cvs_pred_a[1]
        if c2_gt:
            if improvement > 0:
                text += f"Exp10b improved C2 by +{improvement:.2f}"
                color = 'green'
            else:
                text += f"No improvement"
                color = 'gray'
        else:
            if improvement < 0:
                text += f"Exp10b reduced false positive by {-improvement:.2f}"
                color = 'green'
            else:
                text += f"Both correctly reject"
                color = 'gray'

        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_report(entropies, baseline_results, exp10a_results, exp10b_results,
                          sample_names, output_path):
    """Create a markdown summary report."""
    valid_base = [e for e in entropies["baseline"] if e is not None]
    valid_a = [e for e in entropies["exp10a"] if e is not None]
    valid_b = [e for e in entropies["exp10b"] if e is not None]

    report = """# Exp10b Attention Analysis Summary
## Comparing: Baseline vs Exp10a (r=16) vs Exp10b (r=32 + k_proj)

---

## Key Question: Did r=32 + k_proj change WHERE the model attends?

"""

    if valid_base and valid_a and valid_b:
        base_mean = np.mean(valid_base)
        a_mean = np.mean(valid_a)
        b_mean = np.mean(valid_b)

        report += f"""### Entropy Comparison (Lower = More Focused)

| Model | Avg Entropy | vs Baseline | Interpretation |
|-------|-------------|-------------|----------------|
| Baseline V-JEPA | {base_mean:.1f}% | - | Nearly uniform |
| Exp10a (r=16, q+v) | {a_mean:.1f}% | {base_mean - a_mean:+.1f}% | """

        if base_mean - a_mean > 1:
            report += "Slightly more focused |\n"
        else:
            report += "Still uniform |\n"

        report += f"| **Exp10b (r=32, q+k+v)** | **{b_mean:.1f}%** | **{base_mean - b_mean:+.1f}%** | "

        if base_mean - b_mean > 2:
            report += "**More focused!** |\n"
        elif base_mean - b_mean > 0.5:
            report += "Slightly more focused |\n"
        else:
            report += "Still uniform |\n"

        report += f"""
### Answer: """

        if base_mean - b_mean > 2:
            report += f"""**YES - Exp10b shows measurably reduced entropy!**

Adding k_proj and using r=32 DID change attention patterns:
- Entropy reduced by {base_mean - b_mean:.1f}% compared to baseline
- This suggests the model is attending more selectively to certain regions
"""
        elif base_mean - b_mean > 0.5:
            report += f"""**MINOR - Exp10b shows slightly reduced entropy**

Adding k_proj had a small effect:
- Entropy reduced by only {base_mean - b_mean:.1f}%
- Attention patterns are still largely uniform
- Performance gains likely came from better value extraction, not focused attention
"""
        else:
            report += f"""**NO - Attention patterns remain uniform**

Despite adding k_proj and doubling the rank:
- Entropy only changed by {base_mean - b_mean:.1f}%
- V-JEPA's uniform attention is deeply ingrained
- Performance gains came from VALUE projection modifications, not attention focus
"""

    # Predictions comparison
    report += "\n\n---\n\n## Predictions Comparison\n\n"
    report += "| Sample | GT | Exp10a Pred | Exp10a Acc | Exp10b Pred | Exp10b Acc | Winner |\n"
    report += "|--------|----|-----------|-----------|-----------|-----------|---------|\n"

    total_a = 0
    total_b = 0

    for i, name in enumerate(sample_names):
        a_res = exp10a_results[i]
        b_res = exp10b_results[i]

        gt = b_res["cvs_gt"]
        pred_a = a_res["cvs_pred"]
        pred_b = b_res["cvs_pred"]

        acc_a = sum(1 for p, g in zip(pred_a, gt) if (p > 0.5) == (g > 0.5))
        acc_b = sum(1 for p, g in zip(pred_b, gt) if (p > 0.5) == (g > 0.5))

        total_a += acc_a
        total_b += acc_b

        winner = "Exp10b" if acc_b > acc_a else ("Exp10a" if acc_a > acc_b else "Tie")

        gt_str = f"[{int(gt[0])},{int(gt[1])},{int(gt[2])}]"
        pred_a_str = f"[{pred_a[0]:.2f},{pred_a[1]:.2f},{pred_a[2]:.2f}]"
        pred_b_str = f"[{pred_b[0]:.2f},{pred_b[1]:.2f},{pred_b[2]:.2f}]"

        report += f"| {name} | {gt_str} | {pred_a_str} | {acc_a}/3 | {pred_b_str} | {acc_b}/3 | {winner} |\n"

    report += f"\n**Total Accuracy:** Exp10a: {total_a}/{len(sample_names)*3}, Exp10b: {total_b}/{len(sample_names)*3}\n"

    # C2 specific analysis
    report += "\n\n---\n\n## C2 Detection Analysis\n\n"
    report += "C2 (Cystic Plate) showed the biggest improvement in Exp10b (60.34% vs 55.03% AP).\n\n"
    report += "| Sample | C2 GT | Exp10a C2 | Exp10b C2 | Change |\n"
    report += "|--------|-------|-----------|-----------|--------|\n"

    for i, name in enumerate(sample_names):
        a_res = exp10a_results[i]
        b_res = exp10b_results[i]

        c2_gt = int(b_res["cvs_gt"][1])
        c2_a = a_res["cvs_pred"][1]
        c2_b = b_res["cvs_pred"][1]
        change = c2_b - c2_a

        change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
        report += f"| {name} | {c2_gt} | {c2_a:.3f} | {c2_b:.3f} | {change_str} |\n"

    # Conclusions
    report += """

---

## Conclusions

### What Exp10b Changed
1. **Higher LoRA rank (32 vs 16):** More capacity to modify projections
2. **Added k_proj:** Can now modify key vectors, affecting attention computation
3. **Result:** """

    if valid_b and valid_base and (np.mean(valid_base) - np.mean(valid_b)) > 1:
        report += "Measurable reduction in attention entropy\n"
    else:
        report += "Minimal change in attention patterns\n"

    report += """
### Why Performance Improved
"""

    if valid_b and valid_base and (np.mean(valid_base) - np.mean(valid_b)) > 1:
        report += """- **Attention became more focused** - model attends more selectively
- **k_proj modifications** changed WHERE the model looks
- **Both attention AND value extraction improved**
"""
    else:
        report += """- **Value extraction improved** - better features from q_proj, k_proj, v_proj
- **Attention patterns stayed uniform** - V-JEPA's design is robust
- **Task heads benefit** from more discriminative features
"""

    report += """
### Implications
- V-JEPA's uniform attention is fundamental to its architecture
- LoRA modifications primarily improve WHAT gets extracted, not WHERE
- Higher rank + k_proj helps, but doesn't fundamentally change attention
- For focused attention, may need architectural changes (window attention, etc.)

---

*Generated from Exp10b analysis*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved: {output_path}")


def main():
    # Configuration
    exp10a_checkpoint = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\results\exp10_lora\run_20260201_203545\best_model.pt"
    exp10b_checkpoint = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\results\exp10b_lora_r32\run_20260202_204110\best_model.pt"
    data_root = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
    output_dir = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\visualizations\exp10b_attention_analysis")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Target samples
    target_samples = [
        "127_19475",  # Full CVS
        "129_68650",  # C1-only
        "122_35150",  # Negative
    ]

    # Load models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    # Baseline
    baseline_backbone = load_baseline_model(device)

    # Exp10a
    exp10a_model = LoRAModelWrapper(exp10a_checkpoint, device, "Exp10a (r=16, q+v)")
    exp10a_model = exp10a_model.to(device)
    exp10a_model.eval()

    # Exp10b
    exp10b_model = LoRAModelWrapper(exp10b_checkpoint, device, "Exp10b (r=32, q+k+v)")
    exp10b_model = exp10b_model.to(device)
    exp10b_model.eval()

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
    found_samples = find_samples(dataset, target_samples)
    print(f"Found {len(found_samples)}/{len(target_samples)} target samples")

    # Find C2-only samples
    c2_samples = find_c2_only_samples(dataset, n_samples=2)
    for idx, sample_id in c2_samples:
        found_samples[sample_id] = idx
        print(f"Found C2-only sample: {sample_id}")

    # Process samples
    print("\n" + "=" * 70)
    print("PROCESSING SAMPLES")
    print("=" * 70)

    baseline_results = []
    exp10a_results = []
    exp10b_results = []
    sample_names = []

    for sample_id, idx in found_samples.items():
        print(f"\nProcessing: {sample_id}")
        sample = dataset[idx]

        # Baseline
        print("  Baseline...")
        baseline_result = process_baseline_sample(baseline_backbone, sample, device)
        baseline_results.append(baseline_result)

        # Exp10a
        print("  Exp10a...")
        exp10a_result = process_sample_with_model(exp10a_model, sample, device)
        exp10a_results.append(exp10a_result)

        # Exp10b
        print("  Exp10b...")
        exp10b_result = process_sample_with_model(exp10b_model, sample, device)
        exp10b_results.append(exp10b_result)

        sample_names.append(sample_id)

        # Print predictions
        gt = sample['labels'].numpy()
        print(f"  GT: C1={int(gt[0])}, C2={int(gt[1])}, C3={int(gt[2])}")
        print(f"  Exp10a: C1={exp10a_result['cvs_pred'][0]:.3f}, C2={exp10a_result['cvs_pred'][1]:.3f}, C3={exp10a_result['cvs_pred'][2]:.3f}")
        print(f"  Exp10b: C1={exp10b_result['cvs_pred'][0]:.3f}, C2={exp10b_result['cvs_pred'][1]:.3f}, C3={exp10b_result['cvs_pred'][2]:.3f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. All LoRA comparison
    entropies = create_all_lora_comparison_figure(
        baseline_results, exp10a_results, exp10b_results,
        sample_names, output_dir / "attention_comparison_all_lora.png"
    )

    # 2. Entropy comparison
    create_entropy_comparison_figure(
        entropies, sample_names,
        output_dir / "entropy_comparison.png"
    )

    # 3. C2 detection analysis
    create_c2_detection_analysis(
        exp10a_results, exp10b_results, sample_names,
        output_dir / "c2_detection_analysis.png"
    )

    # 4. Summary report
    create_summary_report(
        entropies, baseline_results, exp10a_results, exp10b_results,
        sample_names, output_dir / "ANALYSIS_SUMMARY.md"
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_base = [e for e in entropies["baseline"] if e is not None]
    valid_a = [e for e in entropies["exp10a"] if e is not None]
    valid_b = [e for e in entropies["exp10b"] if e is not None]

    if valid_base and valid_a and valid_b:
        print(f"\nATTENTION ENTROPY (lower = more focused):")
        print(f"  Baseline V-JEPA:     {np.mean(valid_base):.2f}%")
        print(f"  Exp10a (r=16, q+v):  {np.mean(valid_a):.2f}%")
        print(f"  Exp10b (r=32, q+k+v): {np.mean(valid_b):.2f}%")

        improvement_b = np.mean(valid_base) - np.mean(valid_b)
        if improvement_b > 2:
            print(f"\n  Exp10b REDUCED entropy by {improvement_b:.1f}% - attention is more focused!")
        elif improvement_b > 0.5:
            print(f"\n  Exp10b slightly reduced entropy by {improvement_b:.1f}%")
        else:
            print(f"\n  No significant change in attention focus")

    print(f"\nVisualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
