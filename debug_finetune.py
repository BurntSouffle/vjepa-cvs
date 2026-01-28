"""
Debug script for V-JEPA fine-tuning NaN issues.

Systematically checks each stage to identify where NaN is introduced.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import create_model
from dataset_combined import CombinedCVSDataset, collate_fn
from utils import load_config, set_seed


def check_tensor(name: str, tensor: torch.Tensor) -> bool:
    """Check tensor for NaN/Inf and print stats. Returns True if clean."""
    if tensor is None:
        print(f"  {name}: None")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"  {name}: PROBLEM! NaN={has_nan}, Inf={has_inf}")
        print(f"    shape={tensor.shape}, dtype={tensor.dtype}")
        return False
    else:
        print(f"  {name}: OK - min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        return True


def check_gradients(model: nn.Module) -> dict:
    """Check gradients for all parameters."""
    results = {
        "backbone_unfrozen": {"total": 0, "nan": 0, "inf": 0, "zero": 0},
        "head": {"total": 0, "nan": 0, "inf": 0, "zero": 0},
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            category = "backbone_unfrozen" if "backbone" in name else "head"
            results[category]["total"] += 1

            if torch.isnan(param.grad).any():
                results[category]["nan"] += 1
                print(f"    NaN grad in: {name}")
            if torch.isinf(param.grad).any():
                results[category]["inf"] += 1
                print(f"    Inf grad in: {name}")
            if (param.grad == 0).all():
                results[category]["zero"] += 1

    return results


def main():
    print("=" * 70)
    print("V-JEPA Fine-tuning Debug Script")
    print("=" * 70)

    # Load config
    config = load_config("configs/exp5_finetune_attention_local.yaml")
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ========== STEP 1: Create Model ==========
    print("\n" + "=" * 70)
    print("STEP 1: Create Model")
    print("=" * 70)

    model = create_model(config)
    model = model.to(device)

    # Check model state
    print("\nModel configuration:")
    print(f"  freeze_backbone: {config['model']['freeze_backbone']}")
    print(f"  unfreeze_last_n_layers: {config['model'].get('unfreeze_last_n_layers', 0)}")

    # Count trainable params
    backbone_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'backbone' in n)
    head_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n)
    print(f"  Backbone trainable params: {backbone_trainable/1e6:.1f}M")
    print(f"  Head trainable params: {head_trainable/1e6:.1f}M")

    # ========== STEP 2: Create Optimizer with Differential LR ==========
    print("\n" + "=" * 70)
    print("STEP 2: Create Optimizer")
    print("=" * 70)

    backbone_lr = config["training"].get("backbone_lr", 1e-6)
    head_lr = config["training"].get("head_lr", 1e-4)

    print(f"\nConfigured LRs:")
    print(f"  backbone_lr: {backbone_lr}")
    print(f"  head_lr: {head_lr}")

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': head_params, 'lr': head_lr, 'name': 'head'},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config["training"]["weight_decay"],
    )

    print(f"\nOptimizer param groups:")
    for i, pg in enumerate(optimizer.param_groups):
        print(f"  Group {i} ({pg.get('name', 'unknown')}): LR={pg['lr']:.2e}, params={len(pg['params'])}")

    # ========== STEP 3: Create Scheduler ==========
    print("\n" + "=" * 70)
    print("STEP 3: Create Scheduler")
    print("=" * 70)

    # Simple scheduler test - let's check how LambdaLR handles multiple groups
    from torch.optim.lr_scheduler import LambdaLR
    import math

    num_warmup_steps = 100
    num_training_steps = 1000

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"\nScheduler LRs at different steps:")
    for step in [0, 1, 10, 50, 100, 500]:
        # Simulate stepping to this point
        test_optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)
        test_scheduler = LambdaLR(test_optimizer, lr_lambda)
        for _ in range(step):
            test_scheduler.step()
        lrs = test_scheduler.get_last_lr()
        print(f"  Step {step:4d}: backbone_lr={lrs[0]:.2e}, head_lr={lrs[1]:.2e}")

    # ========== STEP 4: Load One Batch ==========
    print("\n" + "=" * 70)
    print("STEP 4: Load Data")
    print("=" * 70)

    dataset = CombinedCVSDataset(
        sages_root=config["data"]["sages_root"],
        endoscapes_root=config["data"]["endoscapes_root"],
        split="train",
        num_frames=config["dataset"]["num_frames"],
        resolution=config["dataset"]["resolution"],
        augment=False,  # No augmentation for debugging
        sages_val_ratio=config["dataset"].get("sages_val_ratio", 0.2),
        seed=config["seed"],
    )

    # Get one batch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    videos = batch["videos"]
    labels = batch["labels"].to(device)

    print(f"\nBatch info:")
    print(f"  Number of videos: {len(videos)}")
    print(f"  Video shape: {videos[0].shape}")
    print(f"  Video dtype: {videos[0].dtype}")
    print(f"  Video range: [{videos[0].min()}, {videos[0].max()}]")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels: {labels}")

    # Check for NaN in input
    for i, v in enumerate(videos):
        if np.isnan(v).any():
            print(f"  WARNING: NaN in video {i}!")

    # ========== STEP 5: Forward Pass - Stage by Stage ==========
    print("\n" + "=" * 70)
    print("STEP 5: Forward Pass (Stage by Stage)")
    print("=" * 70)

    model.train()

    # 5a: Process videos
    print("\n5a. Process videos through processor:")
    pixel_values = model.process_videos(videos, device)
    check_tensor("pixel_values", pixel_values)
    print(f"    dtype: {pixel_values.dtype}")

    # 5b: V-JEPA backbone forward
    print("\n5b. V-JEPA backbone forward:")

    # We need to manually call backbone to inspect intermediate outputs
    with torch.set_grad_enabled(True):
        # The backbone outputs features
        features = model.backbone.get_vision_features(pixel_values_videos=pixel_values)
        check_tensor("backbone_features", features)
        print(f"    dtype: {features.dtype}")

        # Convert to float32 for classifier
        features_f32 = features.float()
        check_tensor("features_float32", features_f32)

    # 5c: Pooling
    print("\n5c. Attention pooling:")
    with torch.set_grad_enabled(True):
        pooled = model.pooler(features_f32)
        check_tensor("pooled", pooled)

    # 5d: Classifier
    print("\n5d. Classifier head:")
    with torch.set_grad_enabled(True):
        logits = model.classifier(pooled)
        check_tensor("logits", logits)

    # 5e: Full forward pass
    print("\n5e. Full forward pass (model(pixel_values)):")
    logits_full = model(pixel_values)
    check_tensor("logits_full", logits_full)

    # ========== STEP 6: Loss Computation ==========
    print("\n" + "=" * 70)
    print("STEP 6: Loss Computation")
    print("=" * 70)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits_full, labels)
    check_tensor("loss", loss)
    print(f"  Loss value: {loss.item()}")

    # ========== STEP 7: Backward Pass ==========
    print("\n" + "=" * 70)
    print("STEP 7: Backward Pass")
    print("=" * 70)

    optimizer.zero_grad()
    loss.backward()

    print("\nGradient check:")
    grad_results = check_gradients(model)

    for category, stats in grad_results.items():
        print(f"\n  {category}:")
        print(f"    Total params with grad: {stats['total']}")
        print(f"    NaN grads: {stats['nan']}")
        print(f"    Inf grads: {stats['inf']}")
        print(f"    Zero grads: {stats['zero']}")

    # Check gradient norms
    print("\nGradient norms:")
    backbone_grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in backbone_params if p.grad is not None],
        float('inf')
    )
    head_grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in head_params if p.grad is not None],
        float('inf')
    )
    print(f"  Backbone grad norm: {backbone_grad_norm:.4f}")
    print(f"  Head grad norm: {head_grad_norm:.4f}")

    # ========== STEP 8: Optimizer Step ==========
    print("\n" + "=" * 70)
    print("STEP 8: Optimizer Step")
    print("=" * 70)

    # Clip gradients first
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Total grad norm (after clipping to 1.0): {total_grad_norm:.4f}")

    # Check a few weights before
    print("\nSample weights BEFORE optimizer step:")
    for name, param in list(model.named_parameters())[:3]:
        if param.requires_grad:
            check_tensor(f"  {name[:50]}", param.data)
            break

    optimizer.step()
    scheduler.step()

    # Check weights after
    print("\nSample weights AFTER optimizer step:")
    for name, param in list(model.named_parameters())[:3]:
        if param.requires_grad:
            check_tensor(f"  {name[:50]}", param.data)
            break

    # Check LR after step
    print(f"\nLR after step:")
    for i, pg in enumerate(optimizer.param_groups):
        print(f"  Group {i} ({pg.get('name', 'unknown')}): LR={pg['lr']:.2e}")

    # ========== STEP 9: Second Forward Pass ==========
    print("\n" + "=" * 70)
    print("STEP 9: Second Forward Pass (after one update)")
    print("=" * 70)

    with torch.no_grad():
        logits2 = model(pixel_values)
        check_tensor("logits2", logits2)
        loss2 = criterion(logits2, labels)
        check_tensor("loss2", loss2)
        print(f"  Loss after update: {loss2.item()}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_ok = True
    issues = []

    if torch.isnan(loss) or torch.isinf(loss):
        all_ok = False
        issues.append("Loss is NaN/Inf")

    if grad_results["backbone_unfrozen"]["nan"] > 0 or grad_results["head"]["nan"] > 0:
        all_ok = False
        issues.append("NaN gradients detected")

    if all_ok:
        print("\nAll checks passed! No NaN/Inf detected in single batch test.")
        print("The issue might be:")
        print("  - Accumulating over multiple batches")
        print("  - Specific problematic samples in the dataset")
        print("  - Scheduler interaction over many steps")
    else:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")

    print("\n" + "=" * 70)
    print("Debug complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
