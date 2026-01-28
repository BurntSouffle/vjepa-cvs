"""
Combined CVS Dataset merging SAGES + Endoscapes for V-JEPA 2.

Handles different resolutions and frame rates.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from dataset import EndoscapesCVSDataset
from dataset_sages import SAGESCVSDataset


class CombinedCVSDataset(Dataset):
    """
    Combined dataset merging SAGES + Endoscapes for CVS classification.

    Features:
        - Merges samples from both datasets
        - Handles different resolutions (resizes to target)
        - Tracks source dataset in metadata
        - Maintains separate train/val splits for each source
    """

    def __init__(
        self,
        sages_root: str,
        endoscapes_root: str,
        split: str = "train",
        num_frames: int = 16,
        resolution: int = 256,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        sages_val_ratio: float = 0.2,
        seed: int = 42,
        include_sages: bool = True,
        include_endoscapes: bool = True,
    ):
        """
        Args:
            sages_root: Path to sages_cvs_challenge_2025_r1 directory
            endoscapes_root: Path to endoscapes directory
            split: One of "train", "val"
            num_frames: Number of frames to load per sample
            resolution: Target resolution for frames (all resized to this)
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
            sages_val_ratio: Ratio of SAGES videos for validation
            seed: Random seed for reproducibility
            include_sages: Whether to include SAGES dataset
            include_endoscapes: Whether to include Endoscapes dataset
        """
        self.sages_root = Path(sages_root)
        self.endoscapes_root = Path(endoscapes_root)
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob

        self.samples = []
        self.datasets = {}

        # Load SAGES dataset
        if include_sages and self.sages_root.exists():
            print(f"Loading SAGES dataset from {self.sages_root}")
            self.datasets["sages"] = SAGESCVSDataset(
                root_dir=str(self.sages_root),
                split=split,
                num_frames=num_frames,
                resolution=resolution,
                augment=augment,
                horizontal_flip_prob=horizontal_flip_prob,
                val_ratio=sages_val_ratio,
                seed=seed,
            )
            # Add samples with dataset reference
            for i, sample in enumerate(self.datasets["sages"].samples):
                self.samples.append({
                    "dataset": "sages",
                    "index": i,
                    **sample,
                })

        # Load Endoscapes dataset
        if include_endoscapes and self.endoscapes_root.exists():
            print(f"Loading Endoscapes dataset from {self.endoscapes_root}")

            # Check if Endoscapes uses the same frame_step
            frame_step = 25  # Endoscapes default

            self.datasets["endoscapes"] = EndoscapesCVSDataset(
                root_dir=str(self.endoscapes_root),
                split=split,
                num_frames=num_frames,
                frame_step=frame_step,
                resolution=resolution,
                augment=augment,
                horizontal_flip_prob=horizontal_flip_prob,
            )
            # Add samples with dataset reference
            for i, sample in enumerate(self.datasets["endoscapes"].samples):
                self.samples.append({
                    "dataset": "endoscapes",
                    "index": i,
                    **sample,
                })

        print(f"[Combined {split}] Total: {len(self.samples)} samples")
        print(f"  - SAGES: {len(self.datasets.get('sages', []))} samples")
        print(f"  - Endoscapes: {len(self.datasets.get('endoscapes', []))} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Dictionary with:
                - video: numpy array (T, H, W, C) uint8
                - labels: tensor (3,) float32
                - meta: dict with video_id, center_frame, dataset
        """
        sample_info = self.samples[idx]
        dataset_name = sample_info["dataset"]
        dataset_idx = sample_info["index"]

        # Get sample from appropriate dataset
        item = self.datasets[dataset_name][dataset_idx]

        # Ensure meta includes dataset source
        item["meta"]["dataset"] = dataset_name

        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    The V-JEPA processor will handle the video processing, so we keep videos as numpy.
    """
    videos = [item["video"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    metas = [item["meta"] for item in batch]

    return {
        "videos": videos,
        "labels": labels,
        "metas": metas,
    }


def get_class_weights(dataset: CombinedCVSDataset) -> torch.Tensor:
    """
    Compute positive class weights for imbalanced data.

    weight = num_negative / num_positive for each class
    """
    all_labels = np.array([s["labels"] for s in dataset.samples])

    weights = []
    for i in range(3):
        pos = (all_labels[:, i] > 0).sum()
        neg = (all_labels[:, i] == 0).sum()
        w = neg / (pos + 1e-6)
        weights.append(min(w, 10.0))

    return torch.tensor(weights, dtype=torch.float32)


def get_dataset_statistics(dataset: CombinedCVSDataset) -> Dict:
    """Get comprehensive statistics about the combined dataset."""
    stats = {
        "total_samples": len(dataset.samples),
        "datasets": {},
    }

    # Per-dataset stats
    for name, ds in dataset.datasets.items():
        all_labels = np.array([s["labels"] for s in ds.samples])

        ds_stats = {
            "samples": len(ds.samples),
            "classes": {},
        }

        for i, class_name in enumerate(["C1", "C2", "C3"]):
            pos = int((all_labels[:, i] > 0).sum())
            neg = int((all_labels[:, i] == 0).sum())
            ds_stats["classes"][class_name] = {
                "positive": pos,
                "negative": neg,
                "ratio": pos / (pos + neg) if (pos + neg) > 0 else 0,
            }

        stats["datasets"][name] = ds_stats

    # Overall class distribution
    all_labels = np.array([s["labels"] for s in dataset.samples])
    stats["overall_classes"] = {}

    for i, class_name in enumerate(["C1", "C2", "C3"]):
        pos = int((all_labels[:, i] > 0).sum())
        neg = int((all_labels[:, i] == 0).sum())
        stats["overall_classes"][class_name] = {
            "positive": pos,
            "negative": neg,
            "ratio": pos / (pos + neg) if (pos + neg) > 0 else 0,
        }

    return stats


if __name__ == "__main__":
    # Test the combined dataset
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Test combined dataset")
    parser.add_argument("--sages", type=str, required=True,
                        help="Path to sages_cvs_challenge_2025_r1 directory")
    parser.add_argument("--endoscapes", type=str, required=True,
                        help="Path to endoscapes directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Combined Dataset")
    print("=" * 60)

    # Create train dataset
    print("\n--- Train Split ---")
    train_dataset = CombinedCVSDataset(
        sages_root=args.sages,
        endoscapes_root=args.endoscapes,
        split="train",
        num_frames=16,
        resolution=256,
        augment=True,
    )

    # Create val dataset
    print("\n--- Val Split ---")
    val_dataset = CombinedCVSDataset(
        sages_root=args.sages,
        endoscapes_root=args.endoscapes,
        split="val",
        num_frames=16,
        resolution=256,
        augment=False,
    )

    # Test loading samples
    print("\n--- Sample Tests ---")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i}: video shape={sample['video'].shape}, "
              f"labels={sample['labels'].tolist()}, "
              f"dataset={sample['meta']['dataset']}")

    # Statistics
    print("\n--- Dataset Statistics ---")
    stats = get_dataset_statistics(train_dataset)
    print(f"Total train samples: {stats['total_samples']}")

    for ds_name, ds_stats in stats["datasets"].items():
        print(f"\n{ds_name.upper()}:")
        print(f"  Samples: {ds_stats['samples']}")
        for class_name, class_stats in ds_stats["classes"].items():
            print(f"  {class_name}: {class_stats['positive']} pos / "
                  f"{class_stats['negative']} neg "
                  f"({class_stats['ratio']*100:.1f}%)")

    print("\nOverall class distribution:")
    for class_name, class_stats in stats["overall_classes"].items():
        print(f"  {class_name}: {class_stats['positive']} pos / "
              f"{class_stats['negative']} neg "
              f"({class_stats['ratio']*100:.1f}%)")

    # Class weights
    print("\n--- Class Weights ---")
    weights = get_class_weights(train_dataset)
    print(f"Weights: {weights.tolist()}")

    # Test DataLoader
    print("\n--- DataLoader Test ---")
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"Batch videos: {len(batch['videos'])} items")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch datasets: {[m['dataset'] for m in batch['metas']]}")

    print("\n" + "=" * 60)
    print("Combined Dataset test complete!")
    print("=" * 60)
