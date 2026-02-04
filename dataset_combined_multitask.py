"""
Combined Multi-task Dataset merging SAGES + Endoscapes with segmentation masks.

Supports:
- CVS classification labels (C1, C2, C3)
- Segmentation masks (GT or SAM2-generated synthetic)
- Data augmentation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_multitask import MultiTaskCVSDataset


class MultiTaskSAGESDataset(Dataset):
    """
    SAGES dataset with CVS labels AND segmentation masks.

    Extends SAGESCVSDataset to also load synthetic segmentation masks.

    Segmentation classes (matching Endoscapes for consistency):
        0: Background
        1: Gallbladder (maps from SAGES class 1)
        5: Anatomy/Tissue (maps from SAGES class 5)
        6: Tool (maps from SAGES class 6)

    For training, we remap to the 5-class system used by Endoscapes:
        0: Background
        1: Cystic Plate / Anatomy
        2: Calot Triangle
        3: Cystic Artery
        4: Cystic Duct
    """

    # SAGES synthetic masks have classes: 0=bg, 1=gallbladder, 5=anatomy, 6=tool
    # Map to Endoscapes-style classes for consistency
    SAGES_CLASS_MAP = {
        0: 0,    # Background -> Background
        1: 0,    # Gallbladder -> Background (or keep if include_gallbladder)
        5: 1,    # Anatomy -> Cystic Plate (generic anatomy class)
        6: 0,    # Tool -> Background
    }

    NUM_SEG_CLASSES = 5

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        resolution: int = 256,
        mask_resolution: int = 64,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        val_ratio: float = 0.2,
        seed: int = 42,
        masks_dir: Optional[str] = None,
        include_gallbladder: bool = False,
    ):
        """
        Args:
            root_dir: Path to sages_cvs_challenge_2025_r1 directory
            split: One of "train", "val"
            num_frames: Number of frames to load per sample
            resolution: Target resolution for frames
            mask_resolution: Resolution for segmentation masks
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
            val_ratio: Ratio of videos for validation
            seed: Random seed
            masks_dir: Path to synthetic masks directory
            include_gallbladder: Keep gallbladder as separate class
        """
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "sages_cvs_challenge_2025"
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.mask_resolution = mask_resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.include_gallbladder = include_gallbladder

        # Update class map if including gallbladder
        if include_gallbladder:
            self.SAGES_CLASS_MAP[1] = 5  # Gallbladder -> class 5
            self.NUM_SEG_CLASSES = 6

        # Directories
        self.frames_dir = self.data_dir / "frames"
        self.labels_dir = self.data_dir / "labels"

        # Get all video IDs with labels
        self.all_video_ids = self._get_video_ids()

        # Create train/val split by video
        self._create_split(val_ratio, seed)

        # Build samples
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
        else:
            selected_indices = indices[n_val:]

        self.video_ids = [self.all_video_ids[i] for i in selected_indices]

    def _load_frame_labels(self, video_id: str) -> pd.DataFrame:
        """Load frame-level labels for a video."""
        label_path = self.labels_dir / video_id / "frame.csv"
        return pd.read_csv(label_path)

    def _majority_vote(self, row: pd.Series, criterion: str) -> float:
        """Compute majority vote for a criterion across 3 raters."""
        votes = [
            row[f"{criterion}_rater1"],
            row[f"{criterion}_rater2"],
            row[f"{criterion}_rater3"],
        ]
        return 1.0 if sum(votes) >= 2 else 0.0

    def _build_samples(self):
        """Build list of samples with labels."""
        self.samples = []
        self.video_frames = {}

        for video_id in self.video_ids:
            labels_df = self._load_frame_labels(video_id)
            available_frames = self._get_available_frames(video_id)
            self.video_frames[video_id] = available_frames

            for _, row in labels_df.iterrows():
                frame_id = int(row["frame_id"])

                c1 = self._majority_vote(row, "c1")
                c2 = self._majority_vote(row, "c2")
                c3 = self._majority_vote(row, "c3")

                labels = np.array([c1, c2, c3], dtype=np.float32)

                frame_path = self.frames_dir / f"{video_id}_{frame_id}.jpg"
                if frame_path.exists():
                    self.samples.append({
                        "video_id": video_id,
                        "center_frame": frame_id,
                        "labels": labels,
                    })

        print(f"[SAGES MultiTask {self.split}] Loaded {len(self.samples)} samples")

    def _get_available_frames(self, video_id: str) -> List[int]:
        """Get sorted list of available frame numbers."""
        frames = []
        for f in self.frames_dir.glob(f"{video_id}_*.jpg"):
            frame_num = int(f.stem.split("_")[-1])
            frames.append(frame_num)
        return sorted(frames)

    def _get_frame_sequence(self, video_id: str, center_frame: int) -> List[int]:
        """Get sequence of frame numbers centered on center_frame."""
        available_frames = self.video_frames.get(video_id, [])
        if not available_frames:
            return [center_frame] * self.num_frames

        try:
            center_idx = available_frames.index(center_frame)
        except ValueError:
            center_idx = min(range(len(available_frames)),
                           key=lambda i: abs(available_frames[i] - center_frame))

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

    def _load_frame(self, video_id: str, frame_num: int) -> np.ndarray:
        """Load a single frame."""
        frame_path = self.frames_dir / f"{video_id}_{frame_num}.jpg"

        if not frame_path.exists():
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        img = Image.open(frame_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def _load_mask(self, video_id: str, frame_num: int) -> Optional[np.ndarray]:
        """Load segmentation mask for a frame."""
        if self.masks_dir is None:
            return None

        mask_path = self.masks_dir / f"{video_id}_{frame_num}.png"

        if not mask_path.exists():
            return None

        mask = Image.open(mask_path)
        mask = mask.resize((self.mask_resolution, self.mask_resolution), Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        # Remap classes
        remapped = np.zeros_like(mask)
        for src, dst in self.SAGES_CLASS_MAP.items():
            remapped[mask == src] = dst

        return remapped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample with video, labels, and masks (matching Endoscapes interface)."""
        sample = self.samples[idx]
        video_id = sample["video_id"]
        center_frame = sample["center_frame"]
        labels = sample["labels"]

        frame_sequence = self._get_frame_sequence(video_id, center_frame)

        # Load frames
        frames = []
        for frame_num in frame_sequence:
            frame = self._load_frame(video_id, frame_num)
            frames.append(frame)
        video = np.stack(frames, axis=0)

        # Load masks for frames in sequence
        masks = []
        mask_indices = []
        for frame_idx, frame_num in enumerate(frame_sequence):
            mask = self._load_mask(video_id, frame_num)
            if mask is not None:
                masks.append(mask)
                mask_indices.append(frame_idx)

        # Apply augmentation
        if self.augment and np.random.random() < self.horizontal_flip_prob:
            video = video[:, :, ::-1, :].copy()
            masks = [m[:, ::-1].copy() for m in masks]

        # Convert masks to tensor (matching Endoscapes interface)
        if masks:
            masks_tensor = torch.tensor(np.stack(masks), dtype=torch.long)
            indices_tensor = torch.tensor(mask_indices, dtype=torch.long)
        else:
            masks_tensor = torch.zeros((0, self.mask_resolution, self.mask_resolution), dtype=torch.long)
            indices_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            "video": video,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "masks": masks_tensor,
            "mask_indices": indices_tensor,
            "has_masks": len(masks) > 0,
            "meta": {
                "video_id": video_id,
                "center_frame": center_frame,
                "dataset": "sages",
                "num_masks": len(masks),
            },
        }


class CombinedMultiTaskDataset(Dataset):
    """
    Combined dataset merging SAGES + Endoscapes with multi-task support.

    Provides:
    - CVS classification labels
    - Segmentation masks
    - Consistent interface for both datasets
    """

    def __init__(
        self,
        endoscapes_root: str,
        sages_root: str,
        split: str = "train",
        num_frames: int = 16,
        resolution: int = 256,
        mask_resolution: int = 64,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        use_synthetic_masks: bool = True,
        endoscapes_gt_masks_dir: Optional[str] = None,
        endoscapes_synthetic_masks_dir: Optional[str] = None,
        sages_masks_dir: Optional[str] = None,
        sages_val_ratio: float = 0.2,
        seed: int = 42,
        include_endoscapes: bool = True,
        include_sages: bool = True,
        # Augmentation params for AugmentedMultiTaskDataset compatibility
        rotation_degrees: float = 0.0,
        color_jitter: Optional[Dict] = None,
        random_erasing_prob: float = 0.0,
        gaussian_blur_prob: float = 0.0,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        """
        Args:
            endoscapes_root: Path to Endoscapes dataset
            sages_root: Path to SAGES dataset (sages_cvs_challenge_2025_r1)
            split: "train" or "val"
            num_frames: Frames per clip
            resolution: Frame resolution
            mask_resolution: Mask resolution
            augment: Enable augmentation
            horizontal_flip_prob: Flip probability
            use_synthetic_masks: Use synthetic masks when GT unavailable
            endoscapes_gt_masks_dir: Override GT masks path
            endoscapes_synthetic_masks_dir: Override synthetic masks path
            sages_masks_dir: Path to SAGES synthetic masks
            sages_val_ratio: Validation ratio for SAGES
            seed: Random seed
            include_endoscapes: Include Endoscapes dataset
            include_sages: Include SAGES dataset
        """
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.mask_resolution = mask_resolution
        self.augment = augment and (split == "train")

        self.samples = []
        self.datasets = {}

        # Load Endoscapes
        if include_endoscapes and Path(endoscapes_root).exists():
            print(f"Loading Endoscapes MultiTask dataset from {endoscapes_root}")
            self.datasets["endoscapes"] = MultiTaskCVSDataset(
                root_dir=endoscapes_root,
                split=split,
                num_frames=num_frames,
                resolution=resolution,
                mask_resolution=mask_resolution,
                augment=augment,
                horizontal_flip_prob=horizontal_flip_prob,
                use_synthetic_masks=use_synthetic_masks,
                gt_masks_dir=endoscapes_gt_masks_dir,
                synthetic_masks_dir=endoscapes_synthetic_masks_dir,
            )

            for i in range(len(self.datasets["endoscapes"])):
                self.samples.append({
                    "dataset": "endoscapes",
                    "index": i,
                })

        # Load SAGES
        if include_sages and Path(sages_root).exists():
            print(f"Loading SAGES MultiTask dataset from {sages_root}")
            self.datasets["sages"] = MultiTaskSAGESDataset(
                root_dir=sages_root,
                split=split,
                num_frames=num_frames,
                resolution=resolution,
                mask_resolution=mask_resolution,
                augment=augment,
                horizontal_flip_prob=horizontal_flip_prob,
                val_ratio=sages_val_ratio,
                seed=seed,
                masks_dir=sages_masks_dir,
            )

            for i in range(len(self.datasets["sages"])):
                self.samples.append({
                    "dataset": "sages",
                    "index": i,
                })

        endoscapes_count = len(self.datasets.get("endoscapes", []))
        sages_count = len(self.datasets.get("sages", []))
        print(f"[Combined {split}] Total: {len(self.samples)} samples")
        print(f"  - Endoscapes: {endoscapes_count}")
        print(f"  - SAGES: {sages_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the appropriate dataset."""
        sample_info = self.samples[idx]
        dataset_name = sample_info["dataset"]
        dataset_idx = sample_info["index"]

        item = self.datasets[dataset_name][dataset_idx]
        item["meta"]["dataset"] = dataset_name

        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for combined dataset.

    Returns dict with:
    - videos: list of numpy arrays (for V-JEPA processor)
    - labels: tensor (B, 3)
    - masks: list of tensors (variable length per sample)
    - mask_indices: list of tensors
    - has_masks: list of bools
    - metas: list of metadata dicts
    """
    videos = [item["video"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    masks = [item["masks"] for item in batch]
    mask_indices = [item["mask_indices"] for item in batch]
    has_masks = [item["has_masks"] for item in batch]
    metas = [item["meta"] for item in batch]

    return {
        "videos": videos,
        "labels": labels,
        "masks": masks,
        "mask_indices": mask_indices,
        "has_masks": has_masks,
        "metas": metas,
    }


if __name__ == "__main__":
    # Test the combined dataset
    print("=" * 60)
    print("Testing Combined MultiTask Dataset")
    print("=" * 60)

    # Local paths
    endoscapes_root = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
    sages_root = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sages_cvs_challenge_2025_r1"
    endoscapes_masks = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\Masks\synthetic_masks_sam2"
    sages_masks = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\Masks\synthetic_masks\semantic"

    # Create dataset
    train_dataset = CombinedMultiTaskDataset(
        endoscapes_root=endoscapes_root,
        sages_root=sages_root,
        split="train",
        num_frames=16,
        resolution=256,
        mask_resolution=64,
        augment=False,
        endoscapes_synthetic_masks_dir=endoscapes_masks,
        sages_masks_dir=sages_masks,
    )

    # Test a few samples from each dataset
    print("\nSample tests (Endoscapes):")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"  {i}: video={sample['video'].shape}, "
              f"labels={sample['labels'].tolist()}, "
              f"masks={sample['masks'].shape}, "
              f"has_masks={sample['has_masks']}, "
              f"dataset={sample['meta']['dataset']}")

    # Test SAGES samples (at the end)
    print("\nSample tests (SAGES):")
    endoscapes_count = len(train_dataset.datasets.get("endoscapes", []))
    for i in range(endoscapes_count, min(endoscapes_count + 3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"  {i}: video={sample['video'].shape}, "
              f"labels={sample['labels'].tolist()}, "
              f"masks={sample['masks'].shape}, "
              f"has_masks={sample['has_masks']}, "
              f"dataset={sample['meta']['dataset']}")

    print("\nDataset test complete!")
