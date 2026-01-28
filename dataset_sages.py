"""
SAGES CVS Challenge 2025 Dataset for V-JEPA 2.

Handles 3-rater annotations with majority voting.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SAGESCVSDataset(Dataset):
    """
    Dataset for CVS classification from SAGES CVS Challenge 2025.

    Features:
        - Loads frames from frames/<video_id>_<frame_number>.jpg
        - Loads labels from labels/<video_id>/frame.csv
        - Handles 3-rater annotations with majority voting
        - Creates train/val split by video (80/20)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        resolution: int = 256,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Path to sages_cvs_challenge_2025_r1 directory
            split: One of "train", "val"
            num_frames: Number of frames to load per sample
            resolution: Target resolution for frames
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
            val_ratio: Ratio of videos to use for validation
            seed: Random seed for train/val split
        """
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "sages_cvs_challenge_2025"
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob

        # Directories
        self.frames_dir = self.data_dir / "frames"
        self.labels_dir = self.data_dir / "labels"

        # Get all video IDs with labels
        self.all_video_ids = self._get_video_ids()

        # Create train/val split by video
        self._create_split(val_ratio, seed)

        # Build frame index and samples
        self._build_samples()

    def _get_video_ids(self) -> List[str]:
        """Get all video IDs that have labels."""
        video_ids = []
        if self.labels_dir.exists():
            for item in self.labels_dir.iterdir():
                if item.is_dir() and (item / "frame.csv").exists():
                    video_ids.append(item.name)
        return sorted(video_ids)

    def _create_split(self, val_ratio: float, seed: int):
        """Create train/val split by video."""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.all_video_ids))
        n_val = int(len(self.all_video_ids) * val_ratio)

        if self.split == "val":
            selected_indices = indices[:n_val]
        else:  # train
            selected_indices = indices[n_val:]

        self.video_ids = [self.all_video_ids[i] for i in selected_indices]
        print(f"[SAGES {self.split}] Selected {len(self.video_ids)} videos "
              f"from {len(self.all_video_ids)} total")

    def _load_frame_labels(self, video_id: str) -> pd.DataFrame:
        """Load frame-level labels for a video."""
        label_path = self.labels_dir / video_id / "frame.csv"
        df = pd.read_csv(label_path)
        return df

    def _majority_vote(self, row: pd.Series, criterion: str) -> float:
        """
        Compute majority vote for a criterion across 3 raters.

        Returns 1.0 if >= 2 raters agree the criterion is met, else 0.0.
        """
        votes = [
            row[f"{criterion}_rater1"],
            row[f"{criterion}_rater2"],
            row[f"{criterion}_rater3"],
        ]
        return 1.0 if sum(votes) >= 2 else 0.0

    def _build_samples(self):
        """Build list of samples with majority-voted labels."""
        self.samples = []
        self.video_frames = {}

        for video_id in self.video_ids:
            # Load labels for this video
            labels_df = self._load_frame_labels(video_id)

            # Get available frames for this video
            available_frames = self._get_available_frames(video_id)
            self.video_frames[video_id] = available_frames

            # Create sample for each labeled frame
            for _, row in labels_df.iterrows():
                frame_id = int(row["frame_id"])

                # Majority vote for each criterion
                c1 = self._majority_vote(row, "c1")
                c2 = self._majority_vote(row, "c2")
                c3 = self._majority_vote(row, "c3")

                labels = np.array([c1, c2, c3], dtype=np.float32)

                # Check if frame exists
                frame_path = self.frames_dir / f"{video_id}_{frame_id}.jpg"
                if frame_path.exists():
                    self.samples.append({
                        "video_id": video_id,
                        "center_frame": frame_id,
                        "labels": labels,
                    })

        print(f"[SAGES {self.split}] Loaded {len(self.samples)} samples")

    def _get_available_frames(self, video_id: str) -> List[int]:
        """Get sorted list of available frame numbers for a video."""
        frames = []
        for f in self.frames_dir.glob(f"{video_id}_*.jpg"):
            # Extract frame number from filename
            frame_num = int(f.stem.split("_")[-1])
            frames.append(frame_num)
        return sorted(frames)

    def _get_frame_sequence(self, video_id: str, center_frame: int) -> List[int]:
        """
        Get sequence of frame numbers centered on center_frame.

        SAGES has 1fps frames (every 30 raw frames).
        We sample frames around the center with 1-frame step.
        """
        available_frames = self.video_frames.get(video_id, [])
        if not available_frames:
            return [center_frame] * self.num_frames

        # Find index of center frame
        try:
            center_idx = available_frames.index(center_frame)
        except ValueError:
            # Find closest frame
            center_idx = min(range(len(available_frames)),
                           key=lambda i: abs(available_frames[i] - center_frame))

        # Calculate start and end indices
        half = self.num_frames // 2
        start_idx = center_idx - half
        end_idx = start_idx + self.num_frames

        # Get frame sequence with padding
        sequence = []
        for i in range(start_idx, end_idx):
            if i < 0:
                sequence.append(available_frames[0])
            elif i >= len(available_frames):
                sequence.append(available_frames[-1])
            else:
                sequence.append(available_frames[i])

        return sequence

    def _load_frame(self, video_id: str, frame_num: int) -> np.ndarray:
        """Load a single frame as numpy array."""
        frame_path = self.frames_dir / f"{video_id}_{frame_num}.jpg"

        if not frame_path.exists():
            # Return blank frame if not found
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        # Load and resize
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

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
        sample = self.samples[idx]
        video_id = sample["video_id"]
        center_frame = sample["center_frame"]
        labels = sample["labels"]

        # Get frame sequence
        frame_sequence = self._get_frame_sequence(video_id, center_frame)

        # Load frames
        frames = []
        for frame_num in frame_sequence:
            frame = self._load_frame(video_id, frame_num)
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
                "dataset": "sages",
            }
        }


def get_class_weights(dataset: SAGESCVSDataset) -> torch.Tensor:
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


def get_class_distribution(dataset: SAGESCVSDataset) -> Dict[str, Dict[str, int]]:
    """Get class distribution statistics."""
    all_labels = np.array([s["labels"] for s in dataset.samples])

    distribution = {}
    for i, name in enumerate(["C1", "C2", "C3"]):
        pos = int((all_labels[:, i] > 0).sum())
        neg = int((all_labels[:, i] == 0).sum())
        distribution[name] = {"positive": pos, "negative": neg, "ratio": pos / (pos + neg)}

    return distribution


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser(description="Test SAGES dataset")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to sages_cvs_challenge_2025_r1 directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing SAGES Dataset")
    print("=" * 60)

    # Create train dataset
    print("\n--- Train Split ---")
    train_dataset = SAGESCVSDataset(
        root_dir=args.root,
        split="train",
        num_frames=16,
        resolution=256,
        augment=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Train videos: {len(train_dataset.video_ids)}")

    # Create val dataset
    print("\n--- Val Split ---")
    val_dataset = SAGESCVSDataset(
        root_dir=args.root,
        split="val",
        num_frames=16,
        resolution=256,
        augment=False,
    )

    print(f"Val samples: {len(val_dataset)}")
    print(f"Val videos: {len(val_dataset.video_ids)}")

    # Test loading a sample
    print("\n--- Sample Test ---")
    sample = train_dataset[0]
    print(f"Video shape: {sample['video'].shape}")
    print(f"Labels: {sample['labels']}")
    print(f"Meta: {sample['meta']}")

    # Class distribution
    print("\n--- Class Distribution (Train) ---")
    dist = get_class_distribution(train_dataset)
    for name, stats in dist.items():
        print(f"  {name}: {stats['positive']} pos / {stats['negative']} neg "
              f"({stats['ratio']*100:.1f}% positive)")

    # Class weights
    print("\n--- Class Weights ---")
    weights = get_class_weights(train_dataset)
    print(f"Weights: {weights.tolist()}")

    print("\n" + "=" * 60)
    print("SAGES Dataset test complete!")
    print("=" * 60)
