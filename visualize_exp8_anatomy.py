"""
Visualization script to check if Exp8 learned surgical anatomy.

Loads the trained multi-task model and visualizes:
- Original frames
- Ground truth segmentation masks
- Predicted segmentation masks
- Overlay of predictions on frames
- CVS predictions vs ground truth
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_multitask import MultiTaskCVSDataset
from model_multitask import VJEPA_MultiTask
from utils import load_config


# Segmentation class names and colors
CLASS_NAMES = {
    0: "Background",
    1: "Cystic Plate (C2)",
    2: "Calot's Triangle (C1)",
    3: "Cystic Artery (C3)",
    4: "Cystic Duct (C3)",
}

# Colors for visualization (RGB, 0-1 scale)
CLASS_COLORS = {
    0: [0.2, 0.2, 0.2],      # Dark gray - Background
    1: [0.0, 0.8, 0.0],      # Green - Cystic Plate
    2: [0.0, 0.5, 1.0],      # Blue - Calot's Triangle
    3: [1.0, 0.0, 0.0],      # Red - Cystic Artery
    4: [1.0, 0.8, 0.0],      # Yellow - Cystic Duct
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class indices to RGB color image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    for class_idx, color in CLASS_COLORS.items():
        colored[mask == class_idx] = color

    return colored


def create_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of colored mask on frame."""
    # Normalize frame to 0-1 if needed
    if frame.max() > 1:
        frame = frame.astype(np.float32) / 255.0

    # Resize mask to match frame if needed
    if mask.shape[:2] != frame.shape[:2]:
        from PIL import Image
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((frame.shape[1], frame.shape[0]), Image.NEAREST)
        mask = np.array(mask_pil)

    # Colorize mask
    colored_mask = colorize_mask(mask)

    # Create overlay (only blend non-background regions)
    overlay = frame.copy()
    non_bg = mask > 0
    overlay[non_bg] = (1 - alpha) * frame[non_bg] + alpha * colored_mask[non_bg]

    return overlay


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> VJEPA_MultiTask:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Create model architecture
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metric: {checkpoint.get('best_metric', 0)*100:.2f}% mAP")

    return model


def find_samples_with_masks(dataset: MultiTaskCVSDataset, n_samples: int = 6) -> list:
    """Find diverse samples that have segmentation masks.

    Tries to get a mix of:
    - Samples with positive CVS labels (C1=1, C2=1, C3=1)
    - Samples from different videos
    """
    # Categorize samples
    positive_cvs = []  # At least one CVS criterion positive
    negative_cvs = []  # All CVS criteria negative
    seen_videos = set()

    for i in range(len(dataset)):
        sample_info = dataset.samples[i]
        if sample_info["has_any_mask"]:
            labels = sample_info["labels"]
            vid = sample_info["video_id"]

            # Check if any CVS criterion is positive
            if labels.sum() > 0:
                positive_cvs.append((i, vid, labels.copy()))
            else:
                negative_cvs.append((i, vid, labels.copy()))

    print(f"Samples with masks: {len(positive_cvs)} positive CVS, {len(negative_cvs)} negative CVS")

    # Select diverse samples: prioritize positive CVS, diverse videos
    selected = []
    seen_videos = set()

    # First, add positive CVS samples from different videos
    for idx, vid, labels in positive_cvs:
        if vid not in seen_videos:
            selected.append(idx)
            seen_videos.add(vid)
            if len(selected) >= n_samples:
                break

    # If we need more, add from remaining positive samples
    if len(selected) < n_samples:
        for idx, vid, labels in positive_cvs:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= n_samples:
                    break

    # If still need more, add negative samples from different videos
    if len(selected) < n_samples:
        for idx, vid, labels in negative_cvs:
            if vid not in seen_videos:
                selected.append(idx)
                seen_videos.add(vid)
                if len(selected) >= n_samples:
                    break

    print(f"Selected {len(selected)} samples from {len(seen_videos)} different videos")
    return selected


def visualize_sample(
    model: VJEPA_MultiTask,
    dataset: MultiTaskCVSDataset,
    sample_idx: int,
    device: torch.device,
) -> tuple:
    """
    Process a single sample and return visualization data.

    Returns:
        tuple: (frame, gt_mask, pred_mask, cvs_pred, cvs_gt, meta)
    """
    # Get sample
    sample = dataset[sample_idx]

    video = sample["video"]  # (T, H, W, C)
    labels = sample["labels"]  # (3,)
    masks = sample["masks"]  # (N, H, W) or empty
    mask_indices = sample["mask_indices"]  # (N,)
    meta = sample["meta"]

    # Get middle frame for visualization
    middle_frame_idx = len(video) // 2
    frame = video[middle_frame_idx]  # (H, W, C)

    # Find if middle frame has a mask
    gt_mask = None
    mask_frame_rel_idx = None

    for i, idx in enumerate(mask_indices.tolist()):
        if idx == middle_frame_idx:
            gt_mask = masks[i].numpy()
            mask_frame_rel_idx = i
            break

    # If middle frame doesn't have mask, use first available
    if gt_mask is None and len(mask_indices) > 0:
        mask_frame_rel_idx = 0
        gt_mask = masks[0].numpy()
        # Update frame to match the mask
        frame_idx = mask_indices[0].item()
        frame = video[frame_idx]
        middle_frame_idx = frame_idx

    # Prepare batch for model
    video_batch = [video]  # List of 1 video
    pixel_values = model.process_videos(video_batch, device)

    # Prepare mask indices for model (single frame)
    if gt_mask is not None:
        frame_indices = torch.tensor([middle_frame_idx], device=device)
        batch_indices = torch.tensor([0], device=device)
    else:
        frame_indices = torch.tensor([], dtype=torch.long, device=device)
        batch_indices = torch.tensor([], dtype=torch.long, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values, frame_indices, batch_indices)

    # CVS predictions
    cvs_logits = outputs["cvs_logits"][0]  # (3,)
    cvs_pred = torch.sigmoid(cvs_logits).cpu().numpy()
    cvs_gt = labels.numpy()

    # Segmentation predictions
    pred_mask = None
    if "seg_logits" in outputs and outputs["seg_logits"].shape[0] > 0:
        seg_logits = outputs["seg_logits"][0]  # (C, H, W)
        pred_mask = seg_logits.argmax(dim=0).cpu().numpy()

    return frame, gt_mask, pred_mask, cvs_pred, cvs_gt, meta


def create_visualization(
    model: VJEPA_MultiTask,
    dataset: MultiTaskCVSDataset,
    sample_indices: list,
    device: torch.device,
    output_path: str,
):
    """Create visualization figure for multiple samples."""
    n_samples = len(sample_indices)
    n_cols = 4  # Original, GT mask, Pred mask, Overlay

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Original Frame", "GT Segmentation", "Predicted Segmentation", "Prediction Overlay"]

    for row, sample_idx in enumerate(sample_indices):
        print(f"Processing sample {sample_idx}...")

        frame, gt_mask, pred_mask, cvs_pred, cvs_gt, meta = visualize_sample(
            model, dataset, sample_idx, device
        )

        # Original frame
        axes[row, 0].imshow(frame)
        axes[row, 0].set_title(f"Video {meta['video_id']}, Frame {meta['center_frame']}")
        axes[row, 0].axis("off")

        # GT mask
        if gt_mask is not None:
            gt_colored = colorize_mask(gt_mask)
            axes[row, 1].imshow(gt_colored)
        else:
            axes[row, 1].text(0.5, 0.5, "No GT mask", ha='center', va='center',
                            transform=axes[row, 1].transAxes)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        # Predicted mask
        if pred_mask is not None:
            pred_colored = colorize_mask(pred_mask)
            axes[row, 2].imshow(pred_colored)
        else:
            axes[row, 2].text(0.5, 0.5, "No prediction", ha='center', va='center',
                            transform=axes[row, 2].transAxes)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

        # Overlay
        if pred_mask is not None:
            overlay = create_overlay(frame, pred_mask, alpha=0.6)
            axes[row, 3].imshow(overlay)
        else:
            axes[row, 3].imshow(frame)
        axes[row, 3].axis("off")

        # Add CVS predictions as text
        cvs_text = f"CVS Pred: C1={cvs_pred[0]:.2f}, C2={cvs_pred[1]:.2f}, C3={cvs_pred[2]:.2f}\n"
        cvs_text += f"CVS GT:   C1={cvs_gt[0]:.0f}, C2={cvs_gt[1]:.0f}, C3={cvs_gt[2]:.0f}"
        axes[row, 3].set_title(f"Overlay\n{cvs_text}", fontsize=9)

    # Add column titles
    for col, title in enumerate(col_titles):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                         fontsize=12, fontweight='bold', ha='center')

    # Add legend
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                     for i in range(5)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()


def print_cvs_summary(
    model: VJEPA_MultiTask,
    dataset: MultiTaskCVSDataset,
    sample_indices: list,
    device: torch.device,
):
    """Print CVS predictions vs ground truth for samples."""
    print("\n" + "=" * 70)
    print("CVS Predictions vs Ground Truth")
    print("=" * 70)
    print(f"{'Sample':<10} {'Vid':<6} {'Frame':<8} {'C1 Pred':<10} {'C1 GT':<8} {'C2 Pred':<10} {'C2 GT':<8} {'C3 Pred':<10} {'C3 GT':<8}")
    print("-" * 70)

    for sample_idx in sample_indices:
        frame, gt_mask, pred_mask, cvs_pred, cvs_gt, meta = visualize_sample(
            model, dataset, sample_idx, device
        )

        print(f"{sample_idx:<10} {meta['video_id']:<6} {meta['center_frame']:<8} "
              f"{cvs_pred[0]:.3f}{'*' if abs(cvs_pred[0]-cvs_gt[0])>0.5 else ' ':<5} {int(cvs_gt[0]):<8} "
              f"{cvs_pred[1]:.3f}{'*' if abs(cvs_pred[1]-cvs_gt[1])>0.5 else ' ':<5} {int(cvs_gt[1]):<8} "
              f"{cvs_pred[2]:.3f}{'*' if abs(cvs_pred[2]-cvs_gt[2])>0.5 else ' ':<5} {int(cvs_gt[2]):<8}")

    print("-" * 70)
    print("* indicates prediction differs from ground truth by >0.5")
    print("=" * 70)


def main():
    # Paths
    checkpoint_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\results\exp8_finetune_multitask\run_20260131_114120\best_model.pt"
    config_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\configs\exp8_finetune_multitask.yaml"
    data_root = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
    output_path = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\vjepa\visualizations\exp8_segmentation_check.png"

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config(config_path)

    # Load model
    model = load_model(checkpoint_path, config, device)

    # Load validation dataset
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

    print(f"Dataset size: {len(dataset)}")
    print(f"Clips with masks: {dataset.clips_with_masks}")

    # Find samples with masks
    sample_indices = find_samples_with_masks(dataset, n_samples=6)

    if len(sample_indices) == 0:
        print("ERROR: No samples with masks found in validation set")
        return

    # Print CVS summary
    print_cvs_summary(model, dataset, sample_indices, device)

    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(model, dataset, sample_indices, device, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
