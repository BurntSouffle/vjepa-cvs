"""
Test anatomy-guided masking locally.

Loads a few samples, generates anatomy-guided masks, and visualizes
to verify masks land on anatomy regions.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import our custom modules
from masks.anatomy_guided_multiblock3d import AnatomyGuidedMaskCollator
from datasets.surgical_video_dataset import SurgicalVideoDataset


def visualize_mask_on_anatomy(
    frame: torch.Tensor,
    anatomy_map: torch.Tensor,
    mask_indices: torch.Tensor,
    patch_size: int = 16,
    tubelet_size: int = 2,
    num_frames: int = 16,
    save_path: str = None
):
    """
    Visualize where masks land relative to anatomy.

    Args:
        frame: (C, H, W) single frame tensor
        anatomy_map: (H, W) anatomy probability map
        mask_indices: 1D tensor of masked patch indices
        patch_size: Spatial patch size
        tubelet_size: Temporal tubelet size
        num_frames: Number of frames in clip
        save_path: Where to save visualization
    """
    H, W = frame.shape[1], frame.shape[2]
    h = H // patch_size  # 16
    w = W // patch_size  # 16
    t = num_frames // tubelet_size  # 8

    # Create mask grid (t, h, w)
    mask_grid = torch.zeros(t, h, w)
    for idx in mask_indices:
        idx = idx.item()
        ti = idx // (h * w)
        hi = (idx % (h * w)) // w
        wi = idx % w
        if ti < t:
            mask_grid[ti, hi, wi] = 1

    # Average mask across time for visualization
    mask_spatial = mask_grid.mean(dim=0)  # (h, w)

    # Upsample mask to image size
    mask_upsampled = torch.nn.functional.interpolate(
        mask_spatial.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze()

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Original frame
    frame_np = frame.permute(1, 2, 0).numpy()
    axes[0].imshow(frame_np)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')

    # 2. Anatomy map
    if anatomy_map is not None:
        axes[1].imshow(anatomy_map.numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Anatomy Priority Map')
    else:
        axes[1].text(0.5, 0.5, 'No anatomy map', ha='center', va='center')
        axes[1].set_title('Anatomy Priority Map (None)')
    axes[1].axis('off')

    # 3. Mask overlay on frame
    frame_with_mask = frame_np.copy()
    mask_np = mask_upsampled.numpy()
    # Red overlay where masked
    overlay = np.zeros_like(frame_np)
    overlay[:, :, 0] = mask_np  # Red channel
    axes[2].imshow(frame_np)
    axes[2].imshow(overlay, alpha=0.5)
    axes[2].set_title(f'Masked Regions ({mask_indices.shape[0]} patches)')
    axes[2].axis('off')

    # 4. Mask grid
    axes[3].imshow(mask_spatial.numpy(), cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title(f'Mask Grid ({h}x{w})')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()

    # Compute mask-anatomy overlap
    if anatomy_map is not None:
        # Downsample anatomy map to patch grid
        anatomy_pooled = torch.nn.functional.avg_pool2d(
            anatomy_map.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=patch_size
        ).squeeze()

        # Compute overlap score
        masked_anatomy = (mask_spatial * anatomy_pooled).sum()
        total_masked = mask_spatial.sum()
        mean_anatomy_in_mask = (masked_anatomy / total_masked).item() if total_masked > 0 else 0
        mean_anatomy_overall = anatomy_pooled.mean().item()

        return {
            'mean_anatomy_in_mask': mean_anatomy_in_mask,
            'mean_anatomy_overall': mean_anatomy_overall,
            'anatomy_ratio': mean_anatomy_in_mask / mean_anatomy_overall if mean_anatomy_overall > 0 else 0
        }

    return None


def test_anatomy_guided_masking():
    """Test the anatomy-guided mask collator."""

    print("=" * 60)
    print("Testing Anatomy-Guided Masking")
    print("=" * 60)

    # Paths - using local Endoscapes data with GT masks
    endoscapes_frames = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes")
    # Use GT masks (semseg/) for local testing since synthetic masks aren't available
    endoscapes_masks = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes\semseg")

    # Output directory
    output_dir = Path(__file__).parent.parent / "vjepa" / "visualizations" / "anatomy_guided_masking"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not endoscapes_frames.exists():
        print(f"ERROR: Endoscapes frames not found at {endoscapes_frames}")
        return False

    print(f"\n1. Loading surgical video dataset...")
    print(f"   Frames: {endoscapes_frames}")
    print(f"   Masks: {endoscapes_masks}")

    # Create dataset with just Endoscapes (SAGES may not be available locally)
    try:
        dataset = SurgicalVideoDataset(
            endoscapes_frames_dir=str(endoscapes_frames),
            endoscapes_masks_dir=str(endoscapes_masks),
            sages_frames_dir=None,
            sages_masks_dir=None,
            frames_per_clip=16,
            crop_size=256,
            centre_crop=480,
        )
        print(f"   Loaded {len(dataset)} clips")
    except Exception as e:
        print(f"   ERROR creating dataset: {e}")
        return False

    if len(dataset) == 0:
        print("   ERROR: No clips found in dataset")
        return False

    # 2. Create anatomy-guided mask collator
    print(f"\n2. Creating anatomy-guided mask collator...")

    # Short and long mask configs (matching surgical_vitl16.yaml)
    mask_cfgs = [
        # Short masks
        {
            'aspect_ratio': (0.75, 1.5),
            'num_blocks': 8,
            'spatial_scale': (0.15, 0.15),
            'temporal_scale': (1.0, 1.0),
            'max_temporal_keep': 1.0,
            'max_keep': None,
        },
        # Long masks
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
        anatomy_bias=0.7,  # 70% anatomy-guided
    )
    print(f"   Created collator with anatomy_bias=0.7")

    # 3. Test on a few samples
    print(f"\n3. Testing on samples...")

    num_samples = min(5, len(dataset))
    anatomy_stats = []

    for i in range(num_samples):
        print(f"\n   Sample {i+1}/{num_samples}:")

        # Get sample
        clip, anatomy_map, clip_info = dataset[i]
        print(f"      Clip: {clip.shape} from {clip_info}")
        print(f"      Anatomy map: {anatomy_map.shape if anatomy_map is not None else 'None'}")

        # Generate masks
        clips_batch = [clip]
        anatomy_maps_batch = [anatomy_map] if anatomy_map is not None else None

        try:
            _, masks_enc, masks_pred = collator(clips_batch, anatomy_maps=anatomy_maps_batch)

            print(f"      Encoder masks: {len(masks_enc)} configs")
            for j, m in enumerate(masks_enc):
                print(f"         Config {j}: {m.shape[1]} patches kept (encoder)")

            print(f"      Predictor masks: {len(masks_pred)} configs")
            for j, m in enumerate(masks_pred):
                print(f"         Config {j}: {m.shape[1]} patches to predict")

            # Visualize first encoder mask
            save_path = output_dir / f"sample_{i+1}_mask_visualization.png"
            stats = visualize_mask_on_anatomy(
                frame=clip[8],  # Middle frame
                anatomy_map=anatomy_map,
                mask_indices=masks_pred[0][0],  # First predictor mask, first sample
                patch_size=16,
                tubelet_size=2,
                num_frames=16,
                save_path=str(save_path)
            )

            if stats:
                anatomy_stats.append(stats)
                print(f"      Anatomy overlap:")
                print(f"         Mean anatomy in mask: {stats['mean_anatomy_in_mask']:.3f}")
                print(f"         Mean anatomy overall: {stats['mean_anatomy_overall']:.3f}")
                print(f"         Ratio (should be >1 if biased): {stats['anatomy_ratio']:.2f}x")

        except Exception as e:
            print(f"      ERROR generating masks: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. Summary
    print(f"\n4. Summary:")
    print(f"   Samples processed: {len(anatomy_stats)}")

    if anatomy_stats:
        mean_ratio = np.mean([s['anatomy_ratio'] for s in anatomy_stats])
        print(f"   Mean anatomy ratio: {mean_ratio:.2f}x")

        if mean_ratio > 1.0:
            print(f"   ✓ Masks are biased toward anatomy regions!")
        else:
            print(f"   ✗ Masks not biased toward anatomy (ratio should be >1)")

    print(f"\n   Visualizations saved to: {output_dir}")

    # 5. Compare with random masking
    print(f"\n5. Comparing with random masking (anatomy_bias=0)...")

    random_collator = AnatomyGuidedMaskCollator(
        cfgs_mask=mask_cfgs,
        crop_size=256,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        anatomy_bias=0.0,  # Pure random
    )

    random_stats = []
    for i in range(min(3, len(dataset))):
        clip, anatomy_map, _ = dataset[i]
        clips_batch = [clip]
        anatomy_maps_batch = [anatomy_map] if anatomy_map is not None else None

        try:
            _, masks_enc, masks_pred = random_collator(clips_batch, anatomy_maps=anatomy_maps_batch)

            stats = visualize_mask_on_anatomy(
                frame=clip[8],
                anatomy_map=anatomy_map,
                mask_indices=masks_pred[0][0],
                patch_size=16,
                tubelet_size=2,
                num_frames=16,
                save_path=str(output_dir / f"random_sample_{i+1}.png")
            )

            if stats:
                random_stats.append(stats)
        except:
            continue

    if random_stats:
        random_mean_ratio = np.mean([s['anatomy_ratio'] for s in random_stats])
        print(f"   Random masking mean ratio: {random_mean_ratio:.2f}x")
        print(f"   Anatomy-guided ratio: {mean_ratio:.2f}x")
        print(f"   Improvement: {mean_ratio / random_mean_ratio:.1f}x better anatomy targeting")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_anatomy_guided_masking()
    sys.exit(0 if success else 1)
