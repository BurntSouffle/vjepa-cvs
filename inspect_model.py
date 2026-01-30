"""
Inspect the best model checkpoint to understand layer structure.
"""

import torch
import sys


def inspect_checkpoint(checkpoint_path: str):
    """Load and inspect checkpoint structure."""
    print("=" * 70)
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Check what's in the checkpoint
    print("\n1. CHECKPOINT KEYS:")
    print("-" * 40)
    for key in checkpoint.keys():
        if key == "model_state_dict":
            print(f"  {key}: dict with {len(checkpoint[key])} entries")
        elif key == "optimizer_state_dict":
            print(f"  {key}: optimizer state")
        else:
            print(f"  {key}: {checkpoint[key]}")

    # Get model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # Might be direct state dict

    # Analyze layer structure
    print("\n2. LAYER NAMES AND SHAPES:")
    print("-" * 70)

    # Group by component
    backbone_layers = []
    pooler_layers = []
    classifier_layers = []

    for name, param in state_dict.items():
        shape = tuple(param.shape)
        size = param.numel()

        if name.startswith("backbone."):
            backbone_layers.append((name, shape, size))
        elif name.startswith("pooler."):
            pooler_layers.append((name, shape, size))
        elif name.startswith("classifier."):
            classifier_layers.append((name, shape, size))
        else:
            print(f"  [OTHER] {name}: {shape}")

    # Report backbone summary (too many layers to show all)
    print(f"\n  BACKBONE: {len(backbone_layers)} layers")
    print(f"  Total backbone params: {sum(s for _, _, s in backbone_layers) / 1e6:.1f}M")

    # Show first and last few backbone layers
    if backbone_layers:
        print("\n  First 5 backbone layers:")
        for name, shape, _ in backbone_layers[:5]:
            print(f"    {name}: {shape}")
        print("    ...")
        print("\n  Last 5 backbone layers:")
        for name, shape, _ in backbone_layers[-5:]:
            print(f"    {name}: {shape}")

    # Show all pooler layers
    print(f"\n  POOLER: {len(pooler_layers)} layers")
    for name, shape, size in pooler_layers:
        print(f"    {name}: {shape} ({size:,} params)")

    # Show all classifier layers
    print(f"\n  CLASSIFIER: {len(classifier_layers)} layers")
    for name, shape, size in classifier_layers:
        print(f"    {name}: {shape} ({size:,} params)")

    # Determine attention pooling type
    print("\n3. ATTENTION POOLING ANALYSIS:")
    print("-" * 40)

    pooler_params = {name: shape for name, shape, _ in pooler_layers}

    if not pooler_layers:
        print("  No pooler layers found - likely using MeanPooling (no learnable params)")
    elif "pooler.query" in pooler_params:
        query_shape = pooler_params["pooler.query"]
        print(f"  Query shape: {query_shape}")
        print(f"  -> Single learnable query token")

        # Check for multi-head attention
        if "pooler.attention.in_proj_weight" in pooler_params:
            in_proj_shape = pooler_params["pooler.attention.in_proj_weight"]
            print(f"  Attention in_proj_weight shape: {in_proj_shape}")

            # in_proj_weight shape is (3 * embed_dim, embed_dim) for Q, K, V projections
            embed_dim = in_proj_shape[1]
            print(f"  Embedding dimension: {embed_dim}")

            # Check num_heads from out_proj
            if "pooler.attention.out_proj.weight" in pooler_params:
                out_proj_shape = pooler_params["pooler.attention.out_proj.weight"]
                print(f"  Attention out_proj shape: {out_proj_shape}")

                # head_dim = embed_dim // num_heads
                # Common head_dim values: 64, 128
                for num_heads in [1, 2, 4, 8, 16, 32]:
                    if embed_dim % num_heads == 0:
                        head_dim = embed_dim // num_heads
                        if head_dim in [64, 128, 256]:
                            print(f"  -> Likely {num_heads} attention heads (head_dim={head_dim})")
    else:
        print("  Unknown pooler structure")

    # Token dimension analysis
    print("\n4. TOKEN DIMENSION ANALYSIS:")
    print("-" * 40)

    # Find hidden dimension from various sources
    hidden_dim = None

    # From pooler query
    if "pooler.query" in pooler_params:
        hidden_dim = pooler_params["pooler.query"][-1]
        print(f"  From pooler.query: hidden_dim = {hidden_dim}")

    # From classifier first layer norm
    for name, shape, _ in classifier_layers:
        if "LayerNorm" in name or "norm" in name.lower():
            if len(shape) == 1:
                hidden_dim = shape[0]
                print(f"  From {name}: hidden_dim = {hidden_dim}")
                break

    # From first Linear layer
    for name, shape, _ in classifier_layers:
        if "Linear" in name or "weight" in name:
            if len(shape) == 2:
                print(f"  From {name}: input_dim={shape[1]}, output_dim={shape[0]}")
                hidden_dim = shape[1]
                break

    if hidden_dim:
        # V-JEPA ViT-L: 256 spatial tokens (16x16 patches from 256x256 image with patch_size=16)
        # Plus CLS token if used, multiplied by temporal patches
        print(f"\n  V-JEPA ViT-L hidden_dim: {hidden_dim}")
        print(f"  Spatial patches: 16x16 = 256 (for 256x256 input with patch_size=16)")
        print(f"  With 16 frames at fpc=16: 256 * 16 = 4096 tokens, or varies with temporal pooling")

    # Config from checkpoint if available
    if "config" in checkpoint:
        print("\n5. TRAINING CONFIG (from checkpoint):")
        print("-" * 40)
        config = checkpoint["config"]
        if isinstance(config, dict):
            import json
            print(json.dumps(config.get("model", config), indent=2)[:500])

    return state_dict


if __name__ == "__main__":
    checkpoint_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\results\best_model.pt"

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    inspect_checkpoint(checkpoint_path)
