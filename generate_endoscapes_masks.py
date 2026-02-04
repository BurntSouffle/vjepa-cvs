"""
Generate Synthetic Segmentation Masks for Endoscapes Dataset
============================================================
Uses the SAM2 + UNet decoder (coarse v2, 3-class) model to generate
semantic segmentation masks for Endoscapes frames that don't have GT masks.

Key differences from SAGES mask generation:
1. Endoscapes already has 494 GT masks - we skip these
2. Endoscapes has train/val/test splits - we respect these
3. Output goes to endoscapes/synthetic_masks/{split}/semantic/

The coarse v2 model predicts: background(0), gallbladder(1), tool(2).
Uncertain regions (where no class is confident) are labeled as anatomy.

Output masks use Endoscapes 7-class format:
    0: background
    1: cystic_plate  (used as generic "anatomy" for uncertain regions)
    2: calot_triangle
    3: cystic_artery
    4: cystic_duct
    5: gallbladder
    6: tool

Usage:
    # Activate sam2_finetune environment first:
    # conda activate sam2_finetune

    python generate_endoscapes_masks.py                    # Full run
    python generate_endoscapes_masks.py --test 10          # Test on 10 frames
    python generate_endoscapes_masks.py --batch_size 8     # Larger batch
    python generate_endoscapes_masks.py --split train      # Only train split
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add SAM2 finetune scripts to path so we can import the model
# ---------------------------------------------------------------------------
SAM2_FINETUNE_ROOT = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune")
SAM2_SCRIPTS_DIR = SAM2_FINETUNE_ROOT / "scripts"
SAM2_COARSE_V2_DIR = SAM2_SCRIPTS_DIR / "coarse_segmentation_v2"

# NOTE: Do NOT add SAM2_FINETUNE_ROOT to sys.path - it contains a sam2/
# subdirectory that would shadow the installed sam2 package.
sys.path.insert(0, str(SAM2_SCRIPTS_DIR))
sys.path.insert(0, str(SAM2_COARSE_V2_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENDOSCAPES_ROOT = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes")
GT_MASKS_DIR = ENDOSCAPES_ROOT / "semseg"
OUTPUT_DIR = ENDOSCAPES_ROOT / "synthetic_masks_sam2"  # New dir to avoid overwriting existing

CHECKPOINT_PATH = SAM2_FINETUNE_ROOT / "checkpoints" / "coarse_segmentation_v2" / "run_20260120_234049" / "best_model.pt"

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Mapping from coarse v2 (3-class) to Endoscapes (7-class) format
# Coarse v2: 0=background, 1=gallbladder, 2=tool
# Endoscapes: 0=bg, 1=cystic_plate, 2=calot_tri, 3=cystic_art, 4=cystic_duct, 5=gb, 6=tool
COARSE_TO_ENDOSCAPES = {
    0: 0,   # background -> background
    1: 5,   # gallbladder -> gallbladder (class 5)
    2: 6,   # tool -> tool (class 6)
}
ANATOMY_CLASS = 1  # uncertain regions -> cystic_plate (generic anatomy marker)


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load the SAM2 + UNet coarse v2 model."""
    from inference_coarse_v2 import load_v2_model
    model = load_v2_model(str(checkpoint_path), device=device)
    return model


# ============================================================================
# Image Preprocessing
# ============================================================================

MODEL_INPUT_SIZE = (512, 512)  # model was trained at 512x512


def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Preprocess a single RGB image for the model.
    Resizes to 512x512 (model training resolution).

    Args:
        image_rgb: (H, W, 3) uint8 RGB image

    Returns:
        (1, 3, 512, 512) normalized float tensor
    """
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]),
                               interpolation=cv2.INTER_LINEAR)
    image = image_resized.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def preprocess_batch(images_rgb: List[np.ndarray], device: str = "cuda") -> torch.Tensor:
    """
    Preprocess a batch of RGB images.
    All images are resized to 512x512.

    Args:
        images_rgb: list of (H, W, 3) uint8 RGB images
        device: target device

    Returns:
        (B, 3, 512, 512) normalized float tensor
    """
    tensors = []
    for img in images_rgb:
        t = preprocess_image(img).squeeze(0)  # (3, 512, 512)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0).to(device)
    return batch


# ============================================================================
# Post-processing
# ============================================================================

def postprocess_mask(
    mask: np.ndarray,
    original_size: Tuple[int, int],
    close_kernel: int = 5,
    min_component_area: int = 100,
) -> np.ndarray:
    """
    Post-process a segmentation mask.

    Args:
        mask: (H, W) uint8 class-index mask
        original_size: (H_orig, W_orig) to resize back to
        close_kernel: kernel size for morphological closing
        min_component_area: remove connected components smaller than this

    Returns:
        (H_orig, W_orig) uint8 cleaned mask
    """
    # Resize to original image resolution (nearest to preserve class IDs)
    if mask.shape[:2] != original_size:
        mask = cv2.resize(mask, (original_size[1], original_size[0]),
                         interpolation=cv2.INTER_NEAREST)

    # Process each non-background class separately
    cleaned = np.zeros_like(mask)

    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue

        binary = (mask == cls_id).astype(np.uint8)

        # Morphological closing: fill small holes
        if close_kernel > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (close_kernel, close_kernel))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove small connected components
        if min_component_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_component_area:
                    binary[labels == i] = 0

        cleaned[binary > 0] = cls_id

    return cleaned.astype(np.uint8)


# ============================================================================
# Mask Generation
# ============================================================================

@torch.no_grad()
def generate_masks_batch(
    model: torch.nn.Module,
    images_rgb: List[np.ndarray],
    device: str = "cuda",
    threshold: float = 0.8,
) -> List[np.ndarray]:
    """
    Generate masks for a batch of same-sized images.

    Args:
        model: loaded SAM2+UNet model
        images_rgb: list of (H, W, 3) uint8 RGB images (must be same size)
        device: cuda/cpu
        threshold: confidence threshold

    Returns:
        list of (H, W) uint8 masks in Endoscapes 7-class format
    """
    if len(images_rgb) == 0:
        return []

    original_sizes = [(img.shape[0], img.shape[1]) for img in images_rgb]

    # Preprocess batch
    batch = preprocess_batch(images_rgb, device)

    # Forward pass
    output = model(batch)
    logits = output["logits"]  # (B, 3, model_H, model_W)

    probs = F.softmax(logits, dim=1)  # (B, 3, model_H, model_W)
    predictions = probs.argmax(dim=1).cpu().numpy()  # (B, model_H, model_W)
    max_probs = probs.max(dim=1)[0].cpu().numpy()  # (B, model_H, model_W)

    masks = []
    for i in range(len(images_rgb)):
        pred = predictions[i]
        max_prob = max_probs[i]

        # Build Endoscapes-format mask
        mask = np.zeros_like(pred, dtype=np.uint8)
        for coarse_cls, endo_cls in COARSE_TO_ENDOSCAPES.items():
            mask[pred == coarse_cls] = endo_cls

        # Uncertain background -> anatomy
        uncertain = max_prob < threshold
        anatomy_mask = uncertain & (pred == 0)
        mask[anatomy_mask] = ANATOMY_CLASS

        # Post-process
        mask = postprocess_mask(mask, original_sizes[i])
        masks.append(mask)

    return masks


# ============================================================================
# Frame Discovery
# ============================================================================

def find_gt_mask_stems(gt_masks_dir: Path) -> Set[str]:
    """Find all frame stems that have GT masks."""
    if not gt_masks_dir.exists():
        return set()

    gt_masks = gt_masks_dir.glob("*.png")
    return {m.stem for m in gt_masks}


def find_frames_by_split(endoscapes_root: Path) -> Dict[str, List[Path]]:
    """
    Find all frames organized by split.

    Returns:
        {"train": [frame_paths], "val": [frame_paths], "test": [frame_paths]}
    """
    frames_by_split = {}

    for split in ["train", "val", "test"]:
        split_dir = endoscapes_root / split
        if split_dir.exists():
            frames = sorted(split_dir.glob("*.jpg"))
            frames_by_split[split] = frames
            print(f"  {split}: {len(frames):,} frames")
        else:
            print(f"  {split}: directory not found at {split_dir}")
            frames_by_split[split] = []

    return frames_by_split


def find_existing_synthetic_masks(output_dir: Path) -> Set[str]:
    """Find frames that already have synthetic masks (for resume)."""
    existing = set()

    for split in ["train", "val", "test"]:
        semantic_dir = output_dir / split / "semantic"
        if semantic_dir.exists():
            for mask_path in semantic_dir.glob("*.png"):
                existing.add(mask_path.stem)

    return existing


# ============================================================================
# Progress Tracking
# ============================================================================

PROGRESS_FILE = "endoscapes_mask_progress.json"


def load_progress(output_dir: Path) -> dict:
    """Load progress from checkpoint file."""
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)
    return {"completed": [], "total": 0, "elapsed_seconds": 0.0, "by_split": {}}


def save_progress(output_dir: Path, progress: dict):
    """Save progress to checkpoint file."""
    progress_path = output_dir / PROGRESS_FILE
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# Main Generation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic segmentation masks for Endoscapes dataset"
    )
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH),
                       help="Path to SAM2+UNet coarse v2 checkpoint")
    parser.add_argument("--endoscapes_root", type=str, default=str(ENDOSCAPES_ROOT),
                       help="Path to Endoscapes dataset root")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                       help="Output directory for masks")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Confidence threshold for anatomy detection")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--test", type=int, default=0,
                       help="Test on N frames only per split (0 = full run)")
    parser.add_argument("--split", type=str, default="all",
                       choices=["all", "train", "val", "test"],
                       help="Which split to process")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't resume from previous progress")
    parser.add_argument("--save_every", type=int, default=1000,
                       help="Save progress every N frames")
    parser.add_argument("--skip_gt", action="store_true", default=True,
                       help="Skip frames that have GT masks (default: True)")
    args = parser.parse_args()

    endoscapes_root = Path(args.endoscapes_root)
    gt_masks_dir = endoscapes_root / "semseg"
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print(" Endoscapes Mask Generation - SAM2 + UNet (Coarse V2)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Find GT masks to skip
    # -----------------------------------------------------------------------
    print(f"\nChecking GT masks in: {gt_masks_dir}")
    gt_mask_stems = find_gt_mask_stems(gt_masks_dir)
    print(f"Found {len(gt_mask_stems)} GT masks (will skip these)")

    # -----------------------------------------------------------------------
    # Find frames by split
    # -----------------------------------------------------------------------
    print(f"\nScanning frames in: {endoscapes_root}")
    frames_by_split = find_frames_by_split(endoscapes_root)

    total_frames = sum(len(frames) for frames in frames_by_split.values())
    print(f"Total frames: {total_frames:,}")

    # Filter to requested split(s)
    if args.split != "all":
        frames_by_split = {args.split: frames_by_split.get(args.split, [])}

    # -----------------------------------------------------------------------
    # Resume from progress
    # -----------------------------------------------------------------------
    progress = load_progress(output_dir)
    completed_set = set(progress.get("completed", []))
    elapsed_prev = progress.get("elapsed_seconds", 0.0)

    if not args.no_resume and len(completed_set) > 0:
        print(f"\nResuming: {len(completed_set):,} frames already processed")

    # -----------------------------------------------------------------------
    # Calculate frames to process
    # -----------------------------------------------------------------------
    frames_to_process_by_split = {}

    for split, frames in frames_by_split.items():
        # Filter out GT masks and already completed
        to_process = []
        skipped_gt = 0
        skipped_done = 0

        for frame_path in frames:
            stem = frame_path.stem

            # Skip if has GT mask
            if args.skip_gt and stem in gt_mask_stems:
                skipped_gt += 1
                continue

            # Skip if already processed (resume)
            if not args.no_resume and stem in completed_set:
                skipped_done += 1
                continue

            to_process.append(frame_path)

        # Limit for testing
        if args.test > 0:
            to_process = to_process[:args.test]

        frames_to_process_by_split[split] = to_process
        print(f"\n{split} split:")
        print(f"  Total frames: {len(frames):,}")
        print(f"  Skipped (GT mask): {skipped_gt:,}")
        print(f"  Skipped (done): {skipped_done:,}")
        print(f"  To process: {len(to_process):,}")

    total_to_process = sum(len(f) for f in frames_to_process_by_split.values())
    print(f"\nTotal frames to process: {total_to_process:,}")

    if total_to_process == 0:
        print("All frames already processed!")
        return

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\nLoading model from: {args.checkpoint}")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device=device)
    print("Model loaded successfully")

    # -----------------------------------------------------------------------
    # Generate masks for each split
    # -----------------------------------------------------------------------
    print(f"\nThreshold: {args.threshold}")
    print(f"Batch size: {args.batch_size}")

    print("\n" + "=" * 70)
    print(" Starting mask generation")
    print("=" * 70)

    start_time = time.time()
    all_newly_completed = []
    all_errors = []
    stats_by_split = {}

    for split, frames_to_process in frames_to_process_by_split.items():
        if len(frames_to_process) == 0:
            continue

        print(f"\n--- Processing {split} split ({len(frames_to_process):,} frames) ---")

        # Set up output directory for this split
        semantic_dir = output_dir / split / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)

        newly_completed = []
        errors = []
        batch_images = []
        batch_paths = []

        pbar = tqdm(frames_to_process, desc=f"{split}", unit="frame")

        for frame_path in pbar:
            try:
                # Load image
                image_bgr = cv2.imread(str(frame_path))
                if image_bgr is None:
                    errors.append((str(frame_path), "Failed to read image"))
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Accumulate for batch processing
                batch_images.append(image_rgb)
                batch_paths.append(frame_path)

                # Process batch when full or last frame
                is_last = (frame_path == frames_to_process[-1])
                if len(batch_images) >= args.batch_size or is_last:
                    masks = generate_masks_batch(
                        model, batch_images, device=device,
                        threshold=args.threshold,
                    )

                    # Save masks
                    for mask, fpath in zip(masks, batch_paths):
                        out_path = semantic_dir / f"{fpath.stem}.png"
                        Image.fromarray(mask).save(out_path)
                        newly_completed.append(fpath.stem)
                        completed_set.add(fpath.stem)

                    # Clear batch
                    batch_images = []
                    batch_paths = []

                # Save progress periodically
                total_done = len(completed_set)
                if total_done % args.save_every == 0 and len(newly_completed) > 0:
                    elapsed = time.time() - start_time + elapsed_prev
                    progress = {
                        "completed": list(completed_set),
                        "total": total_frames,
                        "elapsed_seconds": elapsed,
                        "errors": len(all_errors) + len(errors),
                    }
                    save_progress(output_dir, progress)

                # Update progress bar
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = (len(all_newly_completed) + len(newly_completed)) / elapsed
                    pbar.set_postfix({
                        "speed": f"{speed:.1f}fps",
                        "errs": len(errors),
                    })

            except Exception as e:
                errors.append((str(frame_path), str(e)))
                batch_images = []
                batch_paths = []
                continue

        # Track stats
        stats_by_split[split] = {
            "processed": len(newly_completed),
            "errors": len(errors),
        }
        all_newly_completed.extend(newly_completed)
        all_errors.extend(errors)

        print(f"  Completed: {len(newly_completed):,} masks")
        print(f"  Errors: {len(errors)}")

    # -----------------------------------------------------------------------
    # Final progress save
    # -----------------------------------------------------------------------
    elapsed_total = time.time() - start_time + elapsed_prev
    progress = {
        "completed": list(completed_set),
        "total": total_frames,
        "elapsed_seconds": elapsed_total,
        "errors": len(all_errors),
        "by_split": stats_by_split,
    }
    save_progress(output_dir, progress)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" MASK GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total frames processed: {len(all_newly_completed):,}")
    print(f"Total errors: {len(all_errors)}")
    print(f"Time: {elapsed_total/60:.1f} minutes")

    if len(all_newly_completed) > 0:
        speed = len(all_newly_completed) / (time.time() - start_time)
        print(f"Speed: {speed:.1f} frames/sec")

    print(f"\nOutput: {output_dir}")
    print(f"Mask format: PNG, uint8, Endoscapes 7-class indices")
    print(f"  0=background, 1=anatomy(uncertain), 5=gallbladder, 6=tool")

    print(f"\nPer-split summary:")
    for split, stats in stats_by_split.items():
        print(f"  {split}: {stats['processed']:,} masks, {stats['errors']} errors")

    if all_errors:
        print(f"\nFirst 10 errors:")
        for path, err in all_errors[:10]:
            print(f"  {path}: {err}")

    # Verify a few masks
    if len(all_newly_completed) > 0:
        print(f"\nVerifying sample masks...")
        for split in frames_to_process_by_split.keys():
            semantic_dir = output_dir / split / "semantic"
            sample_masks = list(semantic_dir.glob("*.png"))[:3]
            for mask_path in sample_masks:
                mask = np.array(Image.open(mask_path))
                unique_vals = np.unique(mask)
                print(f"  {mask_path.name}: shape={mask.shape}, classes={unique_vals}")


if __name__ == "__main__":
    main()
