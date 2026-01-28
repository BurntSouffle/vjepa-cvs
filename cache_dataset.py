"""
Pre-cache dataset frames as tensors for faster training.

Converts JPEG images to pre-resized uint8 tensors stored as .pt files.
This speeds up data loading from ~35ms/frame (JPEG decode + resize) to ~2ms/frame (tensor load).

Usage:
    python cache_dataset.py --sages-root /path/to/sages --endoscapes-root /path/to/endoscapes --cache-dir /path/to/cache
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def process_frame(args: Tuple[Path, Path, int]) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single frame: load, resize, save as tensor.

    Args:
        args: Tuple of (input_path, output_path, resolution)

    Returns:
        Tuple of (frame_id, success, error_message)
    """
    input_path, output_path, resolution = args
    frame_id = output_path.stem

    try:
        # Skip if already cached
        if output_path.exists():
            return (frame_id, True, "skipped")

        # Load and resize
        img = Image.open(input_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.BILINEAR)

        # Convert to tensor (uint8 to save space)
        tensor = torch.from_numpy(np.array(img, dtype=np.uint8))

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)

        return (frame_id, True, None)
    except Exception as e:
        return (frame_id, False, str(e))


def get_sages_frames(sages_root: Path) -> List[Tuple[Path, str, int]]:
    """Get all SAGES frames with their video_id and frame_num."""
    frames_dir = sages_root / "sages_cvs_challenge_2025" / "frames"
    frames = []

    if not frames_dir.exists():
        print(f"Warning: SAGES frames directory not found: {frames_dir}")
        return frames

    for f in frames_dir.glob("*.jpg"):
        # Parse filename: {video_id}_{frame_num}.jpg
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            video_id, frame_num = parts[0], int(parts[1])
            frames.append((f, video_id, frame_num))

    return frames


def get_endoscapes_frames(endoscapes_root: Path, splits: List[str] = ["train", "val", "test"]) -> List[Tuple[Path, str, int]]:
    """Get all Endoscapes frames with their video_id and frame_num."""
    frames = []

    for split in splits:
        split_dir = endoscapes_root / split
        if not split_dir.exists():
            print(f"Warning: Endoscapes {split} directory not found: {split_dir}")
            continue

        for f in split_dir.glob("*.jpg"):
            # Parse filename: {video_id}_{frame_num}.jpg
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                video_id, frame_num = parts[0], int(parts[1])
                frames.append((f, video_id, frame_num))

    return frames


def cache_dataset(
    sages_root: Optional[Path],
    endoscapes_root: Optional[Path],
    cache_dir: Path,
    resolution: int = 256,
    num_workers: int = 8,
) -> Dict:
    """
    Cache all dataset frames as tensors.

    Args:
        sages_root: Path to SAGES dataset root
        endoscapes_root: Path to Endoscapes dataset root
        cache_dir: Directory to save cached tensors
        resolution: Target resolution (default 256)
        num_workers: Number of parallel workers

    Returns:
        Metadata dict with caching statistics
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "resolution": resolution,
        "format": "torch.uint8",
        "sages": {"frames": 0, "cached": 0, "errors": 0},
        "endoscapes": {"frames": 0, "cached": 0, "errors": 0},
        "total_size_mb": 0,
    }

    all_tasks = []

    # Collect SAGES frames
    if sages_root and Path(sages_root).exists():
        sages_cache = cache_dir / "sages"
        sages_frames = get_sages_frames(Path(sages_root))
        metadata["sages"]["frames"] = len(sages_frames)

        for input_path, video_id, frame_num in sages_frames:
            output_path = sages_cache / f"{video_id}_{frame_num}.pt"
            all_tasks.append((input_path, output_path, resolution, "sages"))

        print(f"SAGES: Found {len(sages_frames)} frames")

    # Collect Endoscapes frames
    if endoscapes_root and Path(endoscapes_root).exists():
        endo_cache = cache_dir / "endoscapes"
        endo_frames = get_endoscapes_frames(Path(endoscapes_root))
        metadata["endoscapes"]["frames"] = len(endo_frames)

        for input_path, video_id, frame_num in endo_frames:
            output_path = endo_cache / f"{video_id}_{frame_num}.pt"
            all_tasks.append((input_path, output_path, resolution, "endoscapes"))

        print(f"Endoscapes: Found {len(endo_frames)} frames")

    if not all_tasks:
        print("No frames found to cache!")
        return metadata

    # Estimate size
    frame_size_bytes = resolution * resolution * 3  # uint8
    total_size_gb = len(all_tasks) * frame_size_bytes / (1024**3)
    print(f"\nTotal frames: {len(all_tasks)}")
    print(f"Estimated cache size: {total_size_gb:.1f} GB")

    # Count already cached
    already_cached = sum(1 for _, output_path, _, _ in all_tasks if output_path.exists())
    print(f"Already cached: {already_cached} ({100*already_cached/len(all_tasks):.1f}%)")

    if already_cached == len(all_tasks):
        print("All frames already cached!")
        metadata["total_size_mb"] = total_size_gb * 1024
        return metadata

    # Process frames
    print(f"\nCaching frames with {num_workers} workers...")
    start_time = time.time()

    # Prepare tasks (without dataset label for process_frame)
    process_tasks = [(input_path, output_path, resolution) for input_path, output_path, resolution, _ in all_tasks]
    dataset_labels = [dataset for _, _, _, dataset in all_tasks]

    errors = []
    cached_count = {"sages": 0, "endoscapes": 0}
    skipped_count = {"sages": 0, "endoscapes": 0}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_frame, task): i for i, task in enumerate(process_tasks)}

        with tqdm(total=len(all_tasks), desc="Caching frames") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                dataset = dataset_labels[idx]

                try:
                    frame_id, success, error = future.result()
                    if success:
                        if error == "skipped":
                            skipped_count[dataset] += 1
                        else:
                            cached_count[dataset] += 1
                    else:
                        errors.append((frame_id, error))
                        metadata[dataset]["errors"] += 1
                except Exception as e:
                    errors.append((str(idx), str(e)))

                pbar.update(1)

    elapsed = time.time() - start_time

    # Update metadata
    metadata["sages"]["cached"] = cached_count["sages"] + skipped_count["sages"]
    metadata["endoscapes"]["cached"] = cached_count["endoscapes"] + skipped_count["endoscapes"]

    # Calculate actual size
    total_size = 0
    for subdir in ["sages", "endoscapes"]:
        subdir_path = cache_dir / subdir
        if subdir_path.exists():
            for f in subdir_path.glob("*.pt"):
                total_size += f.stat().st_size
    metadata["total_size_mb"] = total_size / (1024**2)

    # Save metadata
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("Caching Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/len(all_tasks)*1000:.1f} ms/frame)")
    print(f"SAGES: {metadata['sages']['cached']}/{metadata['sages']['frames']} cached")
    print(f"Endoscapes: {metadata['endoscapes']['cached']}/{metadata['endoscapes']['frames']} cached")
    print(f"Total size: {metadata['total_size_mb']/1024:.2f} GB")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nFirst 10 errors:")
        for frame_id, error in errors[:10]:
            print(f"  {frame_id}: {error}")

    print(f"\nMetadata saved to: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Pre-cache dataset frames as tensors")
    parser.add_argument("--sages-root", type=str, help="Path to SAGES dataset root")
    parser.add_argument("--endoscapes-root", type=str, help="Path to Endoscapes dataset root")
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory to save cached tensors")
    parser.add_argument("--resolution", type=int, default=256, help="Target resolution (default: 256)")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    if not args.sages_root and not args.endoscapes_root:
        parser.error("At least one of --sages-root or --endoscapes-root must be provided")

    cache_dataset(
        sages_root=args.sages_root,
        endoscapes_root=args.endoscapes_root,
        cache_dir=args.cache_dir,
        resolution=args.resolution,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
