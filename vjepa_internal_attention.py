"""
V-JEPA Internal Attention Extraction
=====================================
Extract self-attention from V-JEPA's transformer blocks to compare
with our attention pooling.

This helps diagnose: Does V-JEPA "see" the right regions, but our pooling
ignores them? Or is V-JEPA already confused?

Usage:
    python vjepa_internal_attention.py --config configs/exp2_local_attention.yaml
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


class AttentionExtractor:
    """
    Extract attention weights from V-JEPA transformer blocks using hooks.
    """

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []

    def _get_attention_hook(self, name):
        """Create a hook function that captures attention weights."""
        def hook(module, input, output):
            # For HuggingFace ViT attention, output is typically:
            # (attn_output, attn_weights) if output_attentions=True
            # or just attn_output
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_weights[name] = attn_weights.detach().cpu()
        return hook

    def register_hooks(self):
        """Register hooks on all attention layers."""
        # Find all attention modules in the backbone
        backbone = self.model.backbone

        # Try to find attention layers - different architectures use different names
        for name, module in backbone.named_modules():
            # Look for MultiheadAttention or similar
            if 'attn' in name.lower() or 'attention' in name.lower():
                if isinstance(module, nn.MultiheadAttention):
                    print(f"Registering hook on: {name}")
                    hook = module.register_forward_hook(self._get_attention_hook(name))
                    self.hooks.append(hook)
                # Also check for custom attention modules
                elif hasattr(module, 'attn') or hasattr(module, 'attention'):
                    print(f"Found attention container: {name}")

        # If no hooks registered, try a different approach
        if not self.hooks:
            print("No direct attention hooks found. Trying encoder layers...")
            # Try accessing encoder layers directly
            if hasattr(backbone, 'encoder'):
                encoder = backbone.encoder
                if hasattr(encoder, 'layer'):
                    for i, layer in enumerate(encoder.layer):
                        if hasattr(layer, 'attention'):
                            attn_module = layer.attention
                            if hasattr(attn_module, 'self'):
                                # BERT-style attention
                                hook = attn_module.self.register_forward_hook(
                                    self._get_attention_hook(f'layer_{i}_attention')
                                )
                                self.hooks.append(hook)
                                print(f"Registered hook on encoder.layer.{i}.attention.self")

        print(f"Registered {len(self.hooks)} attention hooks")
        return len(self.hooks)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_attention(self):
        """Clear stored attention weights."""
        self.attention_weights = {}


def extract_vjepa_attention_manual(model, pixel_values, device):
    """
    Manually extract attention by modifying forward pass.

    V-JEPA uses a custom forward pass, so we need to intercept it.
    Note: V-JEPA with SDPA doesn't support output_attentions directly.
    """
    backbone = model.backbone

    # Check if model supports output_attentions
    if hasattr(backbone.config, '_attn_implementation') and backbone.config._attn_implementation == 'sdpa':
        print("  Model uses SDPA - cannot extract attention via output_attentions")
        print("  Will use hook-based extraction instead")
        return None

    try:
        # Enable output_attentions if the model supports it
        original_output_attentions = getattr(backbone.config, 'output_attentions', False)
        backbone.config.output_attentions = True

        # Run forward pass with attention output
        with torch.no_grad():
            outputs = backbone(pixel_values_videos=pixel_values, output_attentions=True)

        # Check what we got back
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            return outputs.attentions
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            # Some models return (features, attentions)
            return outputs[1] if outputs[1] is not None else None

    except Exception as e:
        print(f"  Could not extract attention via output_attentions: {e}")
    finally:
        if 'original_output_attentions' in dir():
            backbone.config.output_attentions = original_output_attentions

    return None


def get_vjepa_attention_from_last_layer(model, pixel_values, device):
    """
    Get attention from the last transformer layer by hooking into Q, K, V projections.

    V-JEPA uses separate Q, K, V linear projections instead of nn.MultiheadAttention.
    We capture Q and K, then compute attention = softmax(QK^T / sqrt(d_k)).
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

    # V-JEPA uses encoder.layer.X.attention structure
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
        print(f"  Hooked query projection")
    if hasattr(target_layer, 'key'):
        hooks.append(target_layer.key.register_forward_hook(make_hook('key')))
        print(f"  Hooked key projection")
    if hasattr(target_layer, 'value'):
        hooks.append(target_layer.value.register_forward_hook(make_hook('value')))
        print(f"  Hooked value projection")

    if not hooks:
        print("  No Q/K/V projections found")
        return None

    try:
        with torch.no_grad():
            _ = model.backbone.get_vision_features(pixel_values_videos=pixel_values)
    finally:
        for hook in hooks:
            hook.remove()

    # Compute attention weights from Q and K
    if 'query' in captured_qkv and 'key' in captured_qkv:
        Q = captured_qkv['query']  # (B, seq_len, hidden_dim)
        K = captured_qkv['key']    # (B, seq_len, hidden_dim)

        print(f"  Q shape: {Q.shape}, K shape: {K.shape}")

        # Reshape for multi-head attention
        # V-JEPA ViT-L: hidden_dim=1024, num_heads=16, head_dim=64
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

        print(f"  Computed attention shape: {attn_weights.shape}")
        return attn_weights

    return None


def compute_cls_attention(attention_weights, cls_idx=0):
    """
    Extract attention from CLS token (or first token) to all other tokens.

    Args:
        attention_weights: (B, num_heads, seq_len, seq_len)
        cls_idx: Index of CLS token (usually 0)

    Returns:
        cls_attention: (B, num_heads, seq_len) - attention from CLS to all tokens
    """
    if attention_weights is None:
        return None
    return attention_weights[:, :, cls_idx, :]


def compute_mean_attention(attention_weights):
    """
    Compute mean attention across all query positions.

    Args:
        attention_weights: (B, num_heads, seq_len, seq_len)

    Returns:
        mean_attention: (B, num_heads, seq_len) - mean attention to each key position
    """
    if attention_weights is None:
        return None
    return attention_weights.mean(dim=2)  # Average over query positions


def load_specific_samples(config: Dict, video_ids: List[str]):
    """
    Load specific validation samples by video ID.
    """
    from dataset import EndoscapesCVSDataset

    val_dataset = EndoscapesCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="val",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"].get("frame_step", 25),
        resolution=config["dataset"]["resolution"],
        augment=False,
    )

    samples = []

    # Build lookup by video_id
    for i, sample_data in enumerate(val_dataset.samples):
        vid_id = f"{sample_data['video_id']}_{sample_data['center_frame']}"
        if vid_id in video_ids:
            sample = val_dataset[i]
            samples.append({
                "video": sample["video"],
                "label": sample["labels"],
                "video_id": vid_id,
                "index": i
            })

    print(f"Found {len(samples)}/{len(video_ids)} requested samples")
    for s in samples:
        label_str = ", ".join([f"C{i+1}={int(s['label'][i])}" for i in range(3)])
        print(f"  {s['video_id']}: [{label_str}]")

    return samples


def visualize_vjepa_attention(
    sample,
    attention_weights,
    pooler_attention,
    output_dir: Path,
    num_frames: int = 16,
):
    """
    Visualize V-JEPA internal attention vs our pooling attention.
    """
    video_id = sample["video_id"]
    video = sample["video"]
    label = sample["label"]

    print(f"\nVisualizing: {video_id}")

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # Title
    label_str = ", ".join([f"C{i+1}={'Y' if label[i]>0.5 else 'N'}" for i in range(3)])
    fig.suptitle(f"V-JEPA Internal vs Pooling Attention\nVideo: {video_id} | GT: [{label_str}]", fontsize=14)

    # Row 1: Video frames
    frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(4, 5, idx + 1)
        if frame_idx < len(video):
            ax.imshow(video[frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')

    # Row 2: V-JEPA internal attention (if available)
    if attention_weights is not None:
        # Average across heads
        if len(attention_weights.shape) == 4:
            # (B, heads, seq, seq) -> (seq, seq)
            attn = attention_weights[0].mean(dim=0).numpy()
        else:
            attn = attention_weights[0].numpy() if len(attention_weights.shape) == 3 else attention_weights.numpy()

        # Mean attention to each position
        mean_attn = attn.mean(axis=0)  # (seq_len,)

        # Determine spatial/temporal structure
        seq_len = len(mean_attn)

        # V-JEPA typically has: num_patches = (T/temporal_stride) * (H/patch_size) * (W/patch_size)
        # For ViT-L with 256x256 input and 16x16 patches: 16*16 = 256 spatial per frame
        # With temporal pooling, might be fewer

        spatial_tokens = 256  # 16x16 patches
        temporal_tokens = seq_len // spatial_tokens if seq_len >= spatial_tokens else 1

        if temporal_tokens > 1:
            # Has temporal dimension
            try:
                attn_reshaped = mean_attn[:temporal_tokens*spatial_tokens].reshape(temporal_tokens, spatial_tokens)
            except:
                attn_reshaped = mean_attn.reshape(1, -1)
        else:
            attn_reshaped = mean_attn.reshape(1, -1)

        # Plot temporal profile
        ax = fig.add_subplot(4, 5, 6)
        temporal_attn = attn_reshaped.mean(axis=1)
        ax.bar(range(len(temporal_attn)), temporal_attn)
        ax.set_title("V-JEPA Temporal Attn")
        ax.set_xlabel("Frame")

        # Plot spatial attention (averaged over time)
        ax = fig.add_subplot(4, 5, 7)
        spatial_attn = attn_reshaped.mean(axis=0)
        if len(spatial_attn) == 256:
            spatial_map = spatial_attn.reshape(16, 16)
        else:
            side = int(np.sqrt(len(spatial_attn)))
            spatial_map = spatial_attn.reshape(side, -1) if side*side == len(spatial_attn) else spatial_attn.reshape(1, -1)
        ax.imshow(spatial_map, cmap='hot')
        ax.set_title("V-JEPA Spatial Attn")
        ax.axis('off')

        # Full attention matrix
        ax = fig.add_subplot(4, 5, 8)
        ax.imshow(attn_reshaped, cmap='hot', aspect='auto')
        ax.set_title("V-JEPA Full Attn")
        ax.set_xlabel("Spatial")
        ax.set_ylabel("Temporal")

        # Attention head diversity
        if len(attention_weights.shape) == 4:
            ax = fig.add_subplot(4, 5, 9)
            head_means = attention_weights[0].mean(dim=1).mean(dim=1).numpy()
            ax.bar(range(len(head_means)), head_means)
            ax.set_title("Per-Head Attn Mean")
            ax.set_xlabel("Head")

        # Entropy of attention (measure of focus)
        ax = fig.add_subplot(4, 5, 10)
        attn_flat = mean_attn / (mean_attn.sum() + 1e-8)
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
        max_entropy = np.log(len(mean_attn))
        ax.bar(['Attn', 'Max'], [entropy, max_entropy])
        ax.set_title(f"Entropy: {entropy:.2f}/{max_entropy:.2f}")
    else:
        ax = fig.add_subplot(4, 5, 6)
        ax.text(0.5, 0.5, "V-JEPA attention\nnot available", ha='center', va='center')
        ax.axis('off')

    # Row 3: Our pooling attention
    if pooler_attention is not None:
        pooler_attn = pooler_attention
        seq_len = len(pooler_attn)

        spatial_tokens = 256
        temporal_tokens = seq_len // spatial_tokens if seq_len >= spatial_tokens else 1

        if temporal_tokens > 1:
            try:
                pooler_reshaped = pooler_attn[:temporal_tokens*spatial_tokens].reshape(temporal_tokens, spatial_tokens)
            except:
                pooler_reshaped = pooler_attn.reshape(1, -1)
        else:
            pooler_reshaped = pooler_attn.reshape(1, -1)

        # Temporal profile
        ax = fig.add_subplot(4, 5, 11)
        temporal_attn = pooler_reshaped.mean(axis=1)
        ax.bar(range(len(temporal_attn)), temporal_attn)
        ax.set_title("Pooler Temporal Attn")
        ax.set_xlabel("Frame")

        # Spatial attention
        ax = fig.add_subplot(4, 5, 12)
        spatial_attn = pooler_reshaped.mean(axis=0)
        if len(spatial_attn) == 256:
            spatial_map = spatial_attn.reshape(16, 16)
        else:
            side = int(np.sqrt(len(spatial_attn)))
            spatial_map = spatial_attn.reshape(side, -1) if side*side == len(spatial_attn) else spatial_attn.reshape(1, -1)
        ax.imshow(spatial_map, cmap='hot')
        ax.set_title("Pooler Spatial Attn")
        ax.axis('off')

        # Full attention
        ax = fig.add_subplot(4, 5, 13)
        ax.imshow(pooler_reshaped, cmap='hot', aspect='auto')
        ax.set_title("Pooler Full Attn")
        ax.set_xlabel("Spatial")
        ax.set_ylabel("Temporal")

        # Statistics
        ax = fig.add_subplot(4, 5, 14)
        stats = [pooler_attn.mean(), pooler_attn.std(), pooler_attn.max()]
        ax.bar(['Mean', 'Std', 'Max'], stats)
        ax.set_title("Pooler Attn Stats")

        # Entropy
        ax = fig.add_subplot(4, 5, 15)
        attn_flat = pooler_attn / (pooler_attn.sum() + 1e-8)
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
        max_entropy = np.log(len(pooler_attn))
        ax.bar(['Attn', 'Max'], [entropy, max_entropy])
        ax.set_title(f"Entropy: {entropy:.2f}/{max_entropy:.2f}")

    # Row 4: Comparison overlays on video frames
    mid_frame = video[num_frames//2]

    for col, (attn_name, attn_data) in enumerate([
        ("V-JEPA", attention_weights[0].mean(dim=0).mean(dim=0).numpy() if attention_weights is not None else None),
        ("Pooler", pooler_attention)
    ]):
        if attn_data is not None:
            ax = fig.add_subplot(4, 5, 16 + col)

            # Get spatial attention
            spatial_tokens = 256
            if len(attn_data) >= spatial_tokens:
                spatial_attn = attn_data[:spatial_tokens] if len(attn_data) == 256 else attn_data.reshape(-1, spatial_tokens).mean(axis=0)
            else:
                spatial_attn = attn_data

            if len(spatial_attn) == 256:
                attn_map = spatial_attn.reshape(16, 16)
            else:
                side = int(np.sqrt(len(spatial_attn)))
                attn_map = spatial_attn.reshape(side, -1) if side*side == len(spatial_attn) else spatial_attn.reshape(1, -1)

            # Resize to match frame
            from scipy.ndimage import zoom
            h, w = mid_frame.shape[:2]
            scale_h = h / attn_map.shape[0]
            scale_w = w / attn_map.shape[1]
            # Ensure float64 for scipy
            attn_map_float = np.array(attn_map, dtype=np.float64)
            attn_resized = zoom(attn_map_float, (scale_h, scale_w), order=1)

            ax.imshow(mid_frame)
            ax.imshow(attn_resized, cmap='jet', alpha=0.5)
            ax.set_title(f"{attn_name} on Frame")
            ax.axis('off')

    plt.tight_layout()

    # Save
    save_path = output_dir / f"vjepa_vs_pooler_{video_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


class AttentionPoolingWithWeights(nn.Module):
    """Modified AttentionPooling that captures attention weights."""

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

        attn_out, self.attention_weights = self.attention(
            query, x, x, need_weights=True, average_attn_weights=False
        )

        pooled = self.norm(attn_out.squeeze(1))
        return pooled


def main():
    parser = argparse.ArgumentParser(description="V-JEPA Internal Attention Extraction")
    parser.add_argument("--config", type=str, default="configs/exp2_local_attention.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (optional, for pooler attention)")
    parser.add_argument("--output", type=str, default="visualizations/vjepa_internal_attention")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load config
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_config
    from model import create_model

    config = load_config(args.config)

    # Target samples
    video_ids = [
        "127_19475",   # Full CVS, model missed
        "129_68650",   # C1-only
        "122_35150",   # Negative
        "161_21275",   # C2-only
    ]

    # Load samples
    samples = load_specific_samples(config, video_ids)

    if not samples:
        print("ERROR: No samples found!")
        return

    # Create model
    print("\nLoading model...")
    model = create_model(config)

    # Load checkpoint if provided (for trained pooler weights)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    # Replace pooler with weight-capturing version
    if hasattr(model.pooler, 'attention'):
        model.pooler = AttentionPoolingWithWeights(model.pooler)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    results = []

    for sample in samples:
        video_id = sample["video_id"]
        video = sample["video"]
        label = sample["label"]

        print(f"\nProcessing: {video_id}")

        # Prepare input
        pixel_values = model.process_videos([video], device)

        # Try to extract V-JEPA internal attention
        vjepa_attention = None

        # Method 1: Try output_attentions parameter
        vjepa_attention = extract_vjepa_attention_manual(model, pixel_values, device)

        # Method 2: Try hooking last layer
        if vjepa_attention is None:
            vjepa_attention = get_vjepa_attention_from_last_layer(model, pixel_values, device)

        if vjepa_attention is not None:
            if isinstance(vjepa_attention, (list, tuple)):
                vjepa_attention = vjepa_attention[-1]  # Take last layer
            print(f"  V-JEPA attention shape: {vjepa_attention.shape}")
        else:
            print("  WARNING: Could not extract V-JEPA internal attention")

        # Get pooler attention
        with torch.no_grad():
            features = model.backbone.get_vision_features(pixel_values_videos=pixel_values)
            features = features.float()
            pooled = model.pooler(features)

            if hasattr(model.pooler, 'attention_weights') and model.pooler.attention_weights is not None:
                pooler_attn = model.pooler.attention_weights
                # Average across heads: (1, heads, 1, tokens) -> (tokens,)
                pooler_attn = pooler_attn.squeeze(0).squeeze(1).mean(dim=0).cpu().numpy()
                print(f"  Pooler attention shape: {pooler_attn.shape}")
            else:
                pooler_attn = None
                print("  No pooler attention (mean pooling?)")

            # Get predictions
            logits = model.classifier(pooled)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        print(f"  Predictions: C1={probs[0]:.3f}, C2={probs[1]:.3f}, C3={probs[2]:.3f}")
        print(f"  Ground truth: {label.tolist()}")

        # Visualize
        visualize_vjepa_attention(
            sample,
            vjepa_attention,
            pooler_attn,
            output_dir,
            num_frames=config["dataset"]["num_frames"],
        )

        # Store results
        results.append({
            "video_id": video_id,
            "label": label.numpy().tolist(),
            "probs": probs.tolist(),
            "vjepa_attention_available": vjepa_attention is not None,
            "pooler_attention_stats": {
                "mean": float(pooler_attn.mean()) if pooler_attn is not None else None,
                "std": float(pooler_attn.std()) if pooler_attn is not None else None,
                "max": float(pooler_attn.max()) if pooler_attn is not None else None,
            }
        })

    # Save results
    results_file = output_dir / "attention_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
