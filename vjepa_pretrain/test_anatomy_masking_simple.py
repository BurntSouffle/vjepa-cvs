"""
Simple test of anatomy-guided masking using GT masks directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from masks.anatomy_guided_multiblock3d import AnatomyGuidedMaskCollator, segmentation_to_anatomy_map


def test_collator_directly():
    """Test the mask collator with synthetic data first."""
    print("=" * 60)
    print("Testing Mask Collator Directly")
    print("=" * 60)

    # Create collator
    mask_cfgs = [
        # Short masks (8 blocks, 15% spatial scale)
        {
            'aspect_ratio': (0.75, 1.5),
            'num_blocks': 8,
            'spatial_scale': (0.15, 0.15),
            'temporal_scale': (1.0, 1.0),
            'max_temporal_keep': 1.0,
            'max_keep': None,
        },
        # Long masks (2 blocks, 70% spatial scale)
        {
            'aspect_ratio': (0.75, 1.5),
            'num_blocks': 2,
            'spatial_scale': (0.7, 0.7),
            'temporal_scale': (1.0, 1.0),
            'max_temporal_keep': 1.0,
            'max_keep': None,
        },
    ]

    collator = AnatomyGuidedMaskCollator(
        cfgs_mask=mask_cfgs,
        crop_size=256,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        anatomy_bias=0.7,
    )

    print("\n1. Testing with synthetic data (no anatomy map)...")

    # Create dummy batch of 2 videos
    batch = [torch.randn(16, 3, 256, 256) for _ in range(2)]

    # Generate masks without anatomy maps
    _, masks_enc, masks_pred = collator(batch, anatomy_maps=None)

    print(f"   Batch size: {len(batch)}")
    print(f"   Total patches per sample: 8 * 16 * 16 = 2048")
    print(f"\n   Short masks (config 0):")
    print(f"      Encoder (kept): {masks_enc[0].shape} -> {masks_enc[0].shape[1]} patches/sample")
    print(f"      Predictor (predict): {masks_pred[0].shape} -> {masks_pred[0].shape[1]} patches/sample")
    print(f"   Long masks (config 1):")
    print(f"      Encoder (kept): {masks_enc[1].shape} -> {masks_enc[1].shape[1]} patches/sample")
    print(f"      Predictor (predict): {masks_pred[1].shape} -> {masks_pred[1].shape[1]} patches/sample")

    # Verify mask coverage
    short_pred = masks_pred[0].shape[1]
    long_pred = masks_pred[1].shape[1]
    print(f"\n   Short mask coverage: {short_pred/2048*100:.1f}% of patches masked")
    print(f"   Long mask coverage: {long_pred/2048*100:.1f}% of patches masked")

    print("\n2. Testing with anatomy maps...")

    # Create synthetic anatomy map (center has high priority)
    anatomy_map = torch.zeros(256, 256)
    # High priority in center (simulating anatomy)
    anatomy_map[80:176, 80:176] = 1.0
    # Medium priority around (gallbladder area)
    anatomy_map[40:216, 40:216] = torch.maximum(
        anatomy_map[40:216, 40:216],
        torch.full((176, 176), 0.5)
    )

    anatomy_maps = [anatomy_map, anatomy_map]

    _, masks_enc_anat, masks_pred_anat = collator(batch, anatomy_maps=anatomy_maps)

    print(f"   Short masks with anatomy: {masks_pred_anat[0].shape[1]} patches/sample")
    print(f"   Long masks with anatomy: {masks_pred_anat[1].shape[1]} patches/sample")

    print("\n3. Visualizing mask placement...")

    output_dir = Path(__file__).parent.parent / "vjepa" / "visualizations" / "anatomy_guided_masking"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze where masks land
    h, w, t = 16, 16, 8  # patch grid dimensions

    # Compare random vs anatomy-guided mask placement
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Random masking
    for col, (masks_p, title) in enumerate([
        (masks_pred[0][0], "Short (random)"),
        (masks_pred[1][0], "Long (random)"),
    ]):
        # Convert flat indices to spatial grid
        mask_grid = torch.zeros(t, h, w)
        for idx in masks_p:
            ti = idx.item() // (h * w)
            hi = (idx.item() % (h * w)) // w
            wi = idx.item() % w
            if ti < t:
                mask_grid[ti, hi, wi] = 1

        # Average over time
        mask_spatial = mask_grid.mean(dim=0)
        axes[0, col].imshow(mask_spatial, cmap='Reds', vmin=0, vmax=1)
        axes[0, col].set_title(f"{title}\n{len(masks_p)} patches")
        axes[0, col].axis('off')

    # Anatomy map
    axes[0, 2].imshow(anatomy_map[::16, ::16], cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title("Anatomy Map (patch-level)")
    axes[0, 2].axis('off')

    # Row 2: Anatomy-guided masking
    for col, (masks_p, title) in enumerate([
        (masks_pred_anat[0][0], "Short (anatomy)"),
        (masks_pred_anat[1][0], "Long (anatomy)"),
    ]):
        mask_grid = torch.zeros(t, h, w)
        for idx in masks_p:
            ti = idx.item() // (h * w)
            hi = (idx.item() % (h * w)) // w
            wi = idx.item() % w
            if ti < t:
                mask_grid[ti, hi, wi] = 1

        mask_spatial = mask_grid.mean(dim=0)
        axes[1, col].imshow(mask_spatial, cmap='Reds', vmin=0, vmax=1)
        axes[1, col].set_title(f"{title}\n{len(masks_p)} patches")
        axes[1, col].axis('off')

    # Overlap analysis
    # Compute how much masked area overlaps with anatomy
    anatomy_grid = anatomy_map[::16, ::16]  # Downsample to patch grid

    random_overlap = 0
    anat_overlap = 0

    for masks_p in [masks_pred[0][0], masks_pred[1][0]]:
        mask_grid = torch.zeros(h, w)
        for idx in masks_p:
            hi = (idx.item() % (h * w)) // w
            wi = idx.item() % w
            mask_grid[hi, wi] = 1
        random_overlap += (mask_grid * anatomy_grid).sum().item()

    for masks_p in [masks_pred_anat[0][0], masks_pred_anat[1][0]]:
        mask_grid = torch.zeros(h, w)
        for idx in masks_p:
            hi = (idx.item() % (h * w)) // w
            wi = idx.item() % w
            mask_grid[hi, wi] = 1
        anat_overlap += (mask_grid * anatomy_grid).sum().item()

    axes[1, 2].text(0.5, 0.5,
        f"Anatomy Overlap Analysis\n\n"
        f"Random masks:\n  {random_overlap:.1f} weighted patches\n\n"
        f"Anatomy-guided:\n  {anat_overlap:.1f} weighted patches\n\n"
        f"Improvement: {anat_overlap/random_overlap:.1f}x",
        ha='center', va='center', fontsize=12,
        transform=axes[1, 2].transAxes
    )
    axes[1, 2].axis('off')

    plt.suptitle("Anatomy-Guided Masking Test", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "collator_test.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir / 'collator_test.png'}")

    print(f"\n   Random overlap score: {random_overlap:.1f}")
    print(f"   Anatomy overlap score: {anat_overlap:.1f}")
    print(f"   Improvement: {anat_overlap/random_overlap:.1f}x")

    plt.close()

    print("\n4. Testing with real GT masks...")

    gt_masks_dir = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\Masks\synthetic_masks\semantic")
    gt_masks = list(gt_masks_dir.glob("*.png"))[:10]

    if not gt_masks:
        print("   No GT masks found, skipping real data test")
        return True

    real_overlaps = {'random': [], 'anatomy': []}

    for mask_path in gt_masks:
        # Load and process mask
        mask = Image.open(mask_path)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)

        # Convert to anatomy map
        anatomy_map = segmentation_to_anatomy_map(mask_tensor)

        # Generate masks
        batch = [torch.randn(16, 3, 256, 256)]
        _, _, masks_pred_rand = collator(batch, anatomy_maps=None)
        _, _, masks_pred_anat = collator(batch, anatomy_maps=[anatomy_map])

        # Compute overlap
        anatomy_grid = anatomy_map[::16, ::16]

        for masks_p in [masks_pred_rand[0][0], masks_pred_rand[1][0]]:
            mask_grid = torch.zeros(h, w)
            for idx in masks_p:
                hi = (idx.item() % (h * w)) // w
                wi = idx.item() % w
                mask_grid[hi, wi] = 1
            real_overlaps['random'].append((mask_grid * anatomy_grid).sum().item())

        for masks_p in [masks_pred_anat[0][0], masks_pred_anat[1][0]]:
            mask_grid = torch.zeros(h, w)
            for idx in masks_p:
                hi = (idx.item() % (h * w)) // w
                wi = idx.item() % w
                mask_grid[hi, wi] = 1
            real_overlaps['anatomy'].append((mask_grid * anatomy_grid).sum().item())

    random_mean = np.mean(real_overlaps['random'])
    anat_mean = np.mean(real_overlaps['anatomy'])
    print(f"   Random overlap (real masks): {random_mean:.2f}")
    print(f"   Anatomy overlap (real masks): {anat_mean:.2f}")
    print(f"   Improvement: {anat_mean/random_mean:.2f}x")

    # Visualize one real example
    mask = Image.open(gt_masks[0])
    mask = mask.resize((256, 256), Image.NEAREST)
    mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
    anatomy_map = segmentation_to_anatomy_map(mask_tensor)

    batch = [torch.randn(16, 3, 256, 256)]
    _, _, masks_pred_anat = collator(batch, anatomy_maps=[anatomy_map])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Segmentation mask
    axes[0].imshow(mask_tensor, cmap='tab10')
    axes[0].set_title(f"GT Segmentation\n{gt_masks[0].stem}")
    axes[0].axis('off')

    # Anatomy probability map
    axes[1].imshow(anatomy_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Anatomy Priority Map")
    axes[1].axis('off')

    # Mask placement
    mask_grid = torch.zeros(t, h, w)
    for masks_p in [masks_pred_anat[0][0], masks_pred_anat[1][0]]:
        for idx in masks_p:
            ti = idx.item() // (h * w)
            hi = (idx.item() % (h * w)) // w
            wi = idx.item() % w
            if ti < t:
                mask_grid[ti, hi, wi] = 1
    mask_spatial = mask_grid.mean(dim=0)

    # Upsample for visualization
    mask_upsampled = torch.nn.functional.interpolate(
        mask_spatial.unsqueeze(0).unsqueeze(0),
        size=(256, 256),
        mode='nearest'
    ).squeeze()

    axes[2].imshow(anatomy_map, cmap='hot', alpha=0.5)
    axes[2].imshow(mask_upsampled, cmap='Blues', alpha=0.5)
    axes[2].set_title("Mask Placement (blue) on Anatomy (red)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "real_mask_test.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir / 'real_mask_test.png'}")

    # Also test with 100% anatomy bias to see max effect
    print("\n5. Testing with 100% anatomy bias...")

    collator_full = AnatomyGuidedMaskCollator(
        cfgs_mask=mask_cfgs,
        crop_size=256,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        anatomy_bias=1.0,  # 100% anatomy-guided
    )

    # SAGES class weights (class 1=gallbladder, 5=anatomy, 6=tool)
    sages_weights = {
        0: 0.1,   # background
        1: 1.0,   # gallbladder - HIGH
        5: 1.0,   # anatomy - HIGH
        6: 0.3,   # tool - medium
    }

    full_overlaps = []
    for mask_path in gt_masks[:5]:
        mask = Image.open(mask_path)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        anatomy_map = segmentation_to_anatomy_map(mask_tensor, class_weights=sages_weights)

        batch = [torch.randn(16, 3, 256, 256)]
        _, _, masks_pred_full = collator_full(batch, anatomy_maps=[anatomy_map])

        anatomy_grid = anatomy_map[::16, ::16]
        for masks_p in [masks_pred_full[0][0], masks_pred_full[1][0]]:
            mask_grid = torch.zeros(h, w)
            for idx in masks_p:
                hi = (idx.item() % (h * w)) // w
                wi = idx.item() % w
                mask_grid[hi, wi] = 1
            full_overlaps.append((mask_grid * anatomy_grid).sum().item())

    full_mean = np.mean(full_overlaps)
    print(f"   100% anatomy bias overlap: {full_mean:.2f}")
    print(f"   Improvement over random: {full_mean/random_mean:.2f}x")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_collator_directly()
    sys.exit(0 if success else 1)
