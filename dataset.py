"""
Endoscapes CVS Dataset for V-JEPA 2.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EndoscapesCVSDataset(Dataset):
    """
    Dataset for CVS (Critical View of Safety) classification from Endoscapes.

    Loads sequences of consecutive frames centered on labeled frames.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        frame_step: int = 25,
        resolution: int = 256,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
    ):
        """
        Args:
            root_dir: Path to Endoscapes dataset root
            split: One of "train", "val", "test"
            num_frames: Number of frames to load per sample
            frame_step: Step between consecutive frames in Endoscapes (25)
            resolution: Target resolution for frames
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resolution = resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob

        # Load metadata
        metadata_path = self.root_dir / "all_metadata.csv"
        self.metadata = pd.read_csv(metadata_path)

        # Load video IDs for this split
        split_file = self.root_dir / f"{split}_vids.txt"
        with open(split_file, "r") as f:
            self.video_ids = [int(float(line.strip())) for line in f.readlines()]

        # Filter metadata to this split
        self.metadata = self.metadata[self.metadata["vid"].isin(self.video_ids)].reset_index(drop=True)

        # Determine frame directory based on split
        self.frame_dir = self.root_dir / split

        # Build index of available frames per video
        self._build_frame_index()

        # Create sample list (each labeled frame is a sample)
        self._create_samples()

    def _build_frame_index(self):
        """Build index of available frames for each video."""
        self.video_frames = {}

        for vid in self.video_ids:
            # Get all frames for this video from metadata
            vid_meta = self.metadata[self.metadata["vid"] == vid]
            frames = sorted(vid_meta["frame"].unique())
            self.video_frames[vid] = frames

    def _create_samples(self):
        """Create list of samples (video_id, center_frame, labels)."""
        self.samples = []

        for idx, row in self.metadata.iterrows():
            vid = int(row["vid"])
            frame = int(row["frame"])
            # Binarize labels at threshold 0.5 (convert annotator agreement to binary)
            # Labels >0.5 mean majority of annotators agreed the criterion is achieved
            labels = np.array([
                1.0 if row["C1"] >= 0.5 else 0.0,
                1.0 if row["C2"] >= 0.5 else 0.0,
                1.0 if row["C3"] >= 0.5 else 0.0,
            ], dtype=np.float32)

            # Check if frame file exists
            frame_file = self.frame_dir / f"{vid}_{frame}.jpg"
            if frame_file.exists():
                self.samples.append({
                    "video_id": vid,
                    "center_frame": frame,
                    "labels": labels,
                })

        print(f"[{self.split}] Loaded {len(self.samples)} samples from {len(self.video_ids)} videos")

    def _get_frame_sequence(self, video_id: int, center_frame: int) -> List[int]:
        """
        Get sequence of frame numbers centered on center_frame.

        If near video boundary, pad by repeating edge frames.
        """
        available_frames = self.video_frames.get(video_id, [])
        if not available_frames:
            return [center_frame] * self.num_frames

        # Find index of center frame
        try:
            center_idx = available_frames.index(center_frame)
        except ValueError:
            # Frame not in list, find closest
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
                # Pad with first frame
                sequence.append(available_frames[0])
            elif i >= len(available_frames):
                # Pad with last frame
                sequence.append(available_frames[-1])
            else:
                sequence.append(available_frames[i])

        return sequence

    def _load_frame(self, video_id: int, frame_num: int) -> np.ndarray:
        """Load a single frame as numpy array."""
        frame_path = self.frame_dir / f"{video_id}_{frame_num}.jpg"

        if not frame_path.exists():
            # Try alternate extensions
            for ext in [".png", ".jpeg"]:
                alt_path = frame_path.with_suffix(ext)
                if alt_path.exists():
                    frame_path = alt_path
                    break
            else:
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
                - meta: dict with video_id, center_frame
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
                video = video[:, :, ::-1, :].copy()  # Horizontal flip

        return {
            "video": video,  # (T, H, W, C) uint8 numpy - processor expects this
            "labels": torch.tensor(labels, dtype=torch.float32),
            "meta": {
                "video_id": video_id,
                "center_frame": center_frame,
            }
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    The V-JEPA processor will handle the video processing, so we keep videos as numpy.
    """
    videos = [item["video"] for item in batch]  # List of (T, H, W, C) numpy arrays
    labels = torch.stack([item["labels"] for item in batch])  # (B, 3)
    metas = [item["meta"] for item in batch]

    return {
        "videos": videos,  # List of numpy arrays
        "labels": labels,
        "metas": metas,
    }


def get_class_weights(dataset: EndoscapesCVSDataset) -> torch.Tensor:
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
        weights.append(min(w, 10.0))  # Cap at 10 to avoid extreme weights

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test the dataset
    from utils import load_config

    config = load_config("config.yaml")

    # Create dataset
    train_dataset = EndoscapesCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="train",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"]["frame_step"],
        resolution=config["dataset"]["resolution"],
        augment=config["dataset"]["augment_train"],
    )

    print(f"\nDataset size: {len(train_dataset)}")

    # Test loading a sample
    sample = train_dataset[0]
    print(f"Video shape: {sample['video'].shape}")
    print(f"Labels: {sample['labels']}")
    print(f"Meta: {sample['meta']}")

    # Test class weights
    weights = get_class_weights(train_dataset)
    print(f"Class weights: {weights}")
