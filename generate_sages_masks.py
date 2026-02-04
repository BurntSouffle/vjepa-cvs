"""
Generate Synthetic Segmentation Masks for SAGES Dataset
========================================================
Uses the SAM2 + UNet decoder (coarse v2, 3-class) model to generate
semantic segmentation masks for all SAGES frames.

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

    python generate_sages_masks.py                    # Full run
    python generate_sages_masks.py --test 10          # Test on 10 frames
    python generate_sages_masks.py --batch_size 8     # Larger batch
    python generate_sages_masks.py --threshold 0.7    # Lower confidence threshold
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
SAGES_ROOT = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sages_cvs_challenge_2025_r1\sages_cvs_challenge_2025")
SAGES_FRAMES_DIR = SAGES_ROOT / "frames"
OUTPUT_DIR = SAGES_ROOT / "synthetic_masks"

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
def generate_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: str = "cuda",
    threshold: float = 0.8,
) -> np.ndarray:
    """
    Generate a segmentation mask for a single image.

    Args:
        model: loaded SAM2+UNet model
        image_rgb: (H, W, 3) uint8 RGB
        device: cuda/cpu
        threshold: confidence threshold for anatomy detection

    Returns:
        (H, W) uint8 mask in Endoscapes 7-class format
    """
    H_orig, W_orig = image_rgb.shape[:2]

    # Preprocess
    image_tensor = preprocess_image(image_rgb).to(device)

    # Forward pass
    output = model(image_tensor)
    logits = output["logits"]  # (1, 3, model_H, model_W)

    # Softmax probabilities
    probs = F.softmax(logits, dim=1).squeeze(0)  # (3, model_H, model_W)

    # Class predictions
    prediction = probs.argmax(dim=0).cpu().numpy()  # (model_H, model_W)

    # Max confidence per pixel
    max_prob = probs.max(dim=0)[0].cpu().numpy()  # (model_H, model_W)

    # Build Endoscapes-format mask
    mask = np.zeros_like(prediction, dtype=np.uint8)

    # Map confident predictions to Endoscapes classes
    for coarse_cls, endo_cls in COARSE_TO_ENDOSCAPES.items():
        mask[prediction == coarse_cls] = endo_cls

    # Uncertain regions -> anatomy
    uncertain = max_prob < threshold
    # Only mark as anatomy if currently labeled as background
    anatomy_mask = uncertain & (prediction == 0)
    mask[anatomy_mask] = ANATOMY_CLASS

    # Post-process and resize to original resolution
    mask = postprocess_mask(mask, (H_orig, W_orig))

    return mask


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
# Progress Tracking
# ============================================================================

PROGRESS_FILE = "sages_mask_progress.json"


def load_progress(output_dir: Path) -> dict:
    """Load progress from checkpoint file."""
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)
    return {"completed": [], "total": 0, "elapsed_seconds": 0.0}


def save_progress(output_dir: Path, progress: dict):
    """Save progress to checkpoint file."""
    progress_path = output_dir / PROGRESS_FILE
    with open(progress_path, "w") as f:
        json.dump(progress, f)


# ============================================================================
# Main Generation
# ============================================================================

def find_all_frames(frames_dir: Path) -> List[Path]:
    """Find all JPEG frames in the SAGES dataset."""
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(frames_dir.glob("*.jpeg"))
    if not frames:
        frames = sorted(frames_dir.glob("*.png"))
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic segmentation masks for SAGES dataset"
    )
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH),
                       help="Path to SAM2+UNet coarse v2 checkpoint")
    parser.add_argument("--sages_root", type=str, default=str(SAGES_ROOT),
                       help="Path to SAGES dataset root")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                       help="Output directory for masks")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Confidence threshold for anatomy detection")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--test", type=int, default=0,
                       help="Test on N frames only (0 = full run)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't resume from previous progress")
    parser.add_argument("--save_every", type=int, default=1000,
                       help="Save progress every N frames")
    args = parser.parse_args()

    sages_root = Path(args.sages_root)
    frames_dir = sages_root / "frames"
    output_dir = Path(args.output_dir)
    semantic_dir = output_dir / "semantic"

    print("=" * 70)
    print(" SAGES Mask Generation - SAM2 + UNet (Coarse V2)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Find frames
    # -----------------------------------------------------------------------
    print(f"\nScanning frames in: {frames_dir}")
    all_frames = find_all_frames(frames_dir)
    total_frames = len(all_frames)

    if total_frames == 0:
        print(f"ERROR: No frames found in {frames_dir}")
        return

    print(f"Found {total_frames:,} frames")

    # Limit for testing
    if args.test > 0:
        all_frames = all_frames[:args.test]
        print(f"TEST MODE: Processing only {len(all_frames)} frames")

    # -----------------------------------------------------------------------
    # Set up output directory
    # -----------------------------------------------------------------------
    semantic_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {semantic_dir}")

    # -----------------------------------------------------------------------
    # Resume from progress
    # -----------------------------------------------------------------------
    progress = load_progress(output_dir)
    completed_set = set(progress.get("completed", []))
    elapsed_prev = progress.get("elapsed_seconds", 0.0)

    if not args.no_resume and len(completed_set) > 0:
        print(f"Resuming: {len(completed_set):,} frames already processed")

    # Filter out already completed frames
    if not args.no_resume:
        frames_to_process = [f for f in all_frames if f.stem not in completed_set]
    else:
        frames_to_process = all_frames
        completed_set = set()
        elapsed_prev = 0.0

    print(f"Frames to process: {len(frames_to_process):,}")

    if len(frames_to_process) == 0:
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
    # Generate masks
    # -----------------------------------------------------------------------
    print(f"\nThreshold: {args.threshold}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save progress every: {args.save_every} frames")

    print("\n" + "=" * 70)
    print(" Starting mask generation")
    print("=" * 70)

    start_time = time.time()
    newly_completed = []
    errors = []
    batch_images = []
    batch_paths = []
    batch_sizes = []

    pbar = tqdm(frames_to_process, desc="Generating masks", unit="frame")

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
            batch_sizes.append((image_rgb.shape[0], image_rgb.shape[1]))

            # Process batch when full or last frame
            if len(batch_images) >= args.batch_size or frame_path == frames_to_process[-1]:
                # All images get resized to 512x512 in preprocessing,
                # so batching always works regardless of original sizes
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
                batch_sizes = []

            # Save progress periodically
            total_done = len(completed_set)
            if total_done % args.save_every == 0 and len(newly_completed) > 0:
                elapsed = time.time() - start_time + elapsed_prev
                progress = {
                    "completed": list(completed_set),
                    "total": total_frames,
                    "elapsed_seconds": elapsed,
                    "errors": len(errors),
                }
                save_progress(output_dir, progress)

            # Update progress bar
            total_done = len(completed_set)
            elapsed = time.time() - start_time
            if total_done > len(completed_set) - len(newly_completed):
                speed = len(newly_completed) / max(elapsed, 1e-6)
                remaining = len(frames_to_process) - len(newly_completed)
                eta = remaining / max(speed, 1e-6)
                pbar.set_postfix({
                    "done": f"{total_done:,}/{total_frames:,}",
                    "speed": f"{speed:.1f} fps",
                    "ETA": f"{eta/60:.1f}m",
                    "errs": len(errors),
                })

        except Exception as e:
            errors.append((str(frame_path), str(e)))
            # Clear the batch on error to avoid cascading failures
            batch_images = []
            batch_paths = []
            batch_sizes = []
            continue

    # -----------------------------------------------------------------------
    # Final progress save
    # -----------------------------------------------------------------------
    elapsed_total = time.time() - start_time + elapsed_prev
    progress = {
        "completed": list(completed_set),
        "total": total_frames,
        "elapsed_seconds": elapsed_total,
        "errors": len(errors),
    }
    save_progress(output_dir, progress)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" MASK GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total frames processed: {len(completed_set):,} / {total_frames:,}")
    print(f"Newly processed: {len(newly_completed):,}")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed_total/60:.1f} minutes")

    if len(newly_completed) > 0:
        speed = len(newly_completed) / (time.time() - start_time)
        print(f"Speed: {speed:.1f} frames/sec")

    print(f"Output: {semantic_dir}")
    print(f"Mask format: PNG, uint8, Endoscapes 7-class indices")
    print(f"  0=background, 1=anatomy(uncertain), 5=gallbladder, 6=tool")

    if errors:
        print(f"\nFirst 10 errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")

    # Verify a few masks
    if len(newly_completed) > 0:
        print(f"\nVerifying sample masks...")
        sample_masks = list(semantic_dir.glob("*.png"))[:5]
        for mask_path in sample_masks:
            mask = np.array(Image.open(mask_path))
            unique_vals = np.unique(mask)
            print(f"  {mask_path.name}: shape={mask.shape}, classes={unique_vals}")


if __name__ == "__main__":
    main()
