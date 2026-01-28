"""
Cached CVS Dataset for fast training.

Loads pre-cached tensor files instead of JPEG images.
Falls back to raw loading if cache miss.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CachedCVSDataset(Dataset):
    """
    Combined SAGES + Endoscapes dataset with tensor caching support.

    Loads frames from pre-cached .pt files for ~10x faster data loading.
    Falls back to raw JPEG loading on cache miss.
    """

    def __init__(
        self,
        cache_dir: str,
        sages_root: str,
        endoscapes_root: str,
        split: str = "train",
        num_frames: int = 16,
        resolution: int = 256,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        sages_val_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            cache_dir: Directory containing cached .pt files
            sages_root: Path to SAGES dataset (for metadata and fallback)
            endoscapes_root: Path to Endoscapes dataset (for metadata and fallback)
            split: One of "train", "val", "test"
            num_frames: Number of frames to load per sample
            resolution: Target resolution (should match cache)
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
            sages_val_ratio: Ratio of SAGES videos for validation
            seed: Random seed for reproducibility
        """
        self.cache_dir = Path(cache_dir)
        self.sages_root = Path(sages_root)
        self.endoscapes_root = Path(endoscapes_root)
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob

        # Cache paths
        self.sages_cache = self.cache_dir / "sages"
        self.endo_cache = self.cache_dir / "endoscapes"

        # Load cache metadata
        self._load_cache_metadata()

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Build sample list
        self.samples = []
        self.video_frames = {}

        # Load SAGES samples
        if self.sages_root.exists():
            self._load_sages_samples(sages_val_ratio, seed)

        # Load Endoscapes samples
        if self.endoscapes_root.exists():
            self._load_endoscapes_samples()

        print(f"[Cached {split}] Total: {len(self.samples)} samples")
        print(f"  - SAGES: {sum(1 for s in self.samples if s['dataset'] == 'sages')} samples")
        print(f"  - Endoscapes: {sum(1 for s in self.samples if s['dataset'] == 'endoscapes')} samples")

    def _load_cache_metadata(self):
        """Load cache metadata if available."""
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.cache_metadata = json.load(f)
            print(f"[Cache] Loaded metadata: {self.cache_metadata.get('total_size_mb', 0)/1024:.1f} GB cached")
        else:
            self.cache_metadata = {}
            print(f"[Cache] No metadata found, will use fallback loading")

    def _load_sages_samples(self, val_ratio: float, seed: int):
        """Load SAGES samples with train/val split."""
        sages_data_dir = self.sages_root / "sages_cvs_challenge_2025"
        labels_dir = sages_data_dir / "labels"
        frames_dir = sages_data_dir / "frames"

        if not labels_dir.exists():
            print(f"[Cached] SAGES labels not found: {labels_dir}")
            return

        # Get all video IDs with labels
        video_ids = []
        for item in labels_dir.iterdir():
            if item.is_dir() and (item / "frame.csv").exists():
                video_ids.append(item.name)
        video_ids = sorted(video_ids)

        # Create train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(video_ids))
        n_val = int(len(video_ids) * val_ratio)

        if self.split == "val":
            selected_indices = indices[:n_val]
        else:  # train
            selected_indices = indices[n_val:]

        selected_videos = [video_ids[i] for i in selected_indices]

        # Build frame index and samples
        for video_id in selected_videos:
            # Load labels
            label_path = labels_dir / video_id / "frame.csv"
            labels_df = pd.read_csv(label_path)

            # Get available frames
            available_frames = []
            for f in frames_dir.glob(f"{video_id}_*.jpg"):
                frame_num = int(f.stem.split("_")[-1])
                available_frames.append(frame_num)
            available_frames = sorted(available_frames)
            self.video_frames[f"sages_{video_id}"] = available_frames

            # Create samples
            for _, row in labels_df.iterrows():
                frame_id = int(row["frame_id"])

                # Majority vote for labels
                c1 = 1.0 if sum([row["c1_rater1"], row["c1_rater2"], row["c1_rater3"]]) >= 2 else 0.0
                c2 = 1.0 if sum([row["c2_rater1"], row["c2_rater2"], row["c2_rater3"]]) >= 2 else 0.0
                c3 = 1.0 if sum([row["c3_rater1"], row["c3_rater2"], row["c3_rater3"]]) >= 2 else 0.0

                # Check if frame exists (in cache or raw)
                cache_path = self.sages_cache / f"{video_id}_{frame_id}.pt"
                raw_path = frames_dir / f"{video_id}_{frame_id}.jpg"

                if cache_path.exists() or raw_path.exists():
                    self.samples.append({
                        "dataset": "sages",
                        "video_id": video_id,
                        "center_frame": frame_id,
                        "labels": np.array([c1, c2, c3], dtype=np.float32),
                    })

    def _load_endoscapes_samples(self):
        """Load Endoscapes samples."""
        # Determine frame directory based on split
        if self.split == "test":
            frame_dir = self.endoscapes_root / "test"
        else:
            frame_dir = self.endoscapes_root / self.split

        # Load metadata
        metadata_path = self.endoscapes_root / "all_metadata.csv"
        if not metadata_path.exists():
            print(f"[Cached] Endoscapes metadata not found: {metadata_path}")
            return

        metadata = pd.read_csv(metadata_path)

        # Load video IDs for this split
        split_file = self.endoscapes_root / f"{self.split}_vids.txt"
        if not split_file.exists():
            print(f"[Cached] Endoscapes split file not found: {split_file}")
            return

        with open(split_file) as f:
            video_ids = [int(float(line.strip())) for line in f.readlines()]

        # Filter metadata
        metadata = metadata[metadata["vid"].isin(video_ids)]

        # Build frame index
        for vid in video_ids:
            vid_meta = metadata[metadata["vid"] == vid]
            frames = sorted(vid_meta["frame"].unique())
            self.video_frames[f"endoscapes_{vid}"] = frames

        # Create samples
        for _, row in metadata.iterrows():
            vid = int(row["vid"])
            frame = int(row["frame"])

            # Binarize labels
            labels = np.array([
                1.0 if row["C1"] >= 0.5 else 0.0,
                1.0 if row["C2"] >= 0.5 else 0.0,
                1.0 if row["C3"] >= 0.5 else 0.0,
            ], dtype=np.float32)

            # Check if frame exists
            cache_path = self.endo_cache / f"{vid}_{frame}.pt"
            raw_path = frame_dir / f"{vid}_{frame}.jpg"

            if cache_path.exists() or raw_path.exists():
                self.samples.append({
                    "dataset": "endoscapes",
                    "video_id": vid,
                    "center_frame": frame,
                    "labels": labels,
                })

    def _get_frame_sequence(self, dataset: str, video_id, center_frame: int) -> List[int]:
        """Get sequence of frame numbers centered on center_frame."""
        key = f"{dataset}_{video_id}"
        available_frames = self.video_frames.get(key, [])

        if not available_frames:
            return [center_frame] * self.num_frames

        # Find center index
        try:
            center_idx = available_frames.index(center_frame)
        except ValueError:
            center_idx = min(range(len(available_frames)),
                           key=lambda i: abs(available_frames[i] - center_frame))

        # Get sequence with padding
        half = self.num_frames // 2
        start_idx = center_idx - half

        sequence = []
        for i in range(start_idx, start_idx + self.num_frames):
            if i < 0:
                sequence.append(available_frames[0])
            elif i >= len(available_frames):
                sequence.append(available_frames[-1])
            else:
                sequence.append(available_frames[i])

        return sequence

    def _load_frame_cached(self, dataset: str, video_id, frame_num: int) -> np.ndarray:
        """Load frame from cache, fallback to raw if not cached."""
        # Determine cache path
        if dataset == "sages":
            cache_path = self.sages_cache / f"{video_id}_{frame_num}.pt"
            raw_dir = self.sages_root / "sages_cvs_challenge_2025" / "frames"
        else:
            cache_path = self.endo_cache / f"{video_id}_{frame_num}.pt"
            raw_dir = self.endoscapes_root / self.split

        # Try cache first
        if cache_path.exists():
            self.cache_hits += 1
            tensor = torch.load(cache_path)
            return tensor.numpy()

        # Fallback to raw
        self.cache_misses += 1
        raw_path = raw_dir / f"{video_id}_{frame_num}.jpg"

        if raw_path.exists():
            img = Image.open(raw_path).convert("RGB")
            img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
            return np.array(img, dtype=np.uint8)
        else:
            # Return blank frame
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample."""
        sample = self.samples[idx]
        dataset = sample["dataset"]
        video_id = sample["video_id"]
        center_frame = sample["center_frame"]
        labels = sample["labels"]

        # Get frame sequence
        frame_sequence = self._get_frame_sequence(dataset, video_id, center_frame)

        # Load frames
        frames = []
        for frame_num in frame_sequence:
            frame = self._load_frame_cached(dataset, video_id, frame_num)
            frames.append(frame)

        # Stack into video array (T, H, W, C)
        video = np.stack(frames, axis=0)

        # Apply augmentation
        if self.augment:
            if np.random.random() < self.horizontal_flip_prob:
                video = video[:, :, ::-1, :].copy()

        return {
            "video": video,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "meta": {
                "video_id": video_id,
                "center_frame": center_frame,
                "dataset": dataset,
            }
        }

    def get_cache_stats(self) -> Dict:
        """Get cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    videos = [item["video"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    metas = [item["meta"] for item in batch]

    return {
        "videos": videos,
        "labels": labels,
        "metas": metas,
    }


if __name__ == "__main__":
    # Test the cached dataset
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--sages-root", type=str, required=True)
    parser.add_argument("--endoscapes-root", type=str, required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Testing CachedCVSDataset")
    print("=" * 60)

    # Create dataset
    dataset = CachedCVSDataset(
        cache_dir=args.cache_dir,
        sages_root=args.sages_root,
        endoscapes_root=args.endoscapes_root,
        split="train",
        num_frames=16,
        resolution=256,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading
    print("\nLoading 10 samples...")
    start = time.time()
    for i in range(10):
        sample = dataset[i]
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s ({elapsed/10*1000:.1f}ms/sample)")

    # Show sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Labels: {sample['labels']}")
    print(f"  Meta: {sample['meta']}")

    # Cache stats
    stats = dataset.get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
