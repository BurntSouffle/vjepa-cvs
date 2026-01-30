"""
Multi-task Dataset for V-JEPA CVS Classification + Segmentation.

Loads video clips with CVS labels and segmentation masks (where available).
Uses GT masks when available, falls back to SAM2-generated synthetic masks.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiTaskCVSDataset(Dataset):
    """
    Dataset for multi-task learning: CVS classification + anatomical segmentation.

    For each clip:
    - Loads 16 frames centered on the labeled frame
    - Loads CVS labels (C1, C2, C3)
    - Loads segmentation masks for frames that have them (GT preferred, SAM2 fallback)

    Segmentation classes:
        0: Background
        1: Cystic Plate (C2 criterion)
        2: Calot Triangle (C1 criterion)
        3: Cystic Artery (C3 criterion)
        4: Cystic Duct (C3 criterion)
        5: Gallbladder (context)
        6: Tool (instruments)
        255: Ignore
    """

    # Map from original class IDs to our 5-class setup (excluding tools)
    # We focus on anatomically relevant classes for CVS
    CLASS_MAP = {
        0: 0,    # Background -> Background
        1: 1,    # Cystic Plate -> Cystic Plate
        2: 2,    # Calot Triangle -> Calot Triangle
        3: 3,    # Cystic Artery -> Cystic Artery
        4: 4,    # Cystic Duct -> Cystic Duct
        5: 0,    # Gallbladder -> Background (or keep as 5 if needed)
        6: 0,    # Tool -> Background (ignore tools)
        255: 255,  # Ignore -> Ignore
    }

    NUM_SEG_CLASSES = 5  # Background + 4 anatomical structures

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        frame_step: int = 25,
        resolution: int = 256,
        mask_resolution: int = 64,  # Downsampled mask resolution for efficiency
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        use_synthetic_masks: bool = True,
        include_gallbladder: bool = False,  # Whether to include gallbladder as class 5
    ):
        """
        Args:
            root_dir: Path to Endoscapes dataset root
            split: One of "train", "val", "test"
            num_frames: Number of frames to load per sample
            frame_step: Step between frames in Endoscapes (25)
            resolution: Target resolution for frames
            mask_resolution: Resolution for segmentation masks (smaller for efficiency)
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
            use_synthetic_masks: Whether to use SAM2 synthetic masks as fallback
            include_gallbladder: Whether to keep gallbladder as separate class
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resolution = resolution
        self.mask_resolution = mask_resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob
        self.use_synthetic_masks = use_synthetic_masks
        self.include_gallbladder = include_gallbladder

        # Update class map if including gallbladder
        if include_gallbladder:
            self.CLASS_MAP[5] = 5
            self.NUM_SEG_CLASSES = 6

        # Paths
        self.frame_dir = self.root_dir / split
        self.semseg_dir = self.root_dir / "semseg"
        self.synthetic_dir = self.root_dir / "synthetic_masks"

        # Load metadata
        metadata_path = self.root_dir / "all_metadata.csv"
        self.metadata = pd.read_csv(metadata_path)

        # Load video IDs for this split
        split_file = self.root_dir / f"{split}_vids.txt"
        with open(split_file, "r") as f:
            self.video_ids = [int(float(line.strip())) for line in f.readlines()]

        # Filter metadata to this split
        self.metadata = self.metadata[self.metadata["vid"].isin(self.video_ids)].reset_index(drop=True)

        # Build frame index and mask index
        self._build_indices()

        # Create sample list
        self._create_samples()

        # Statistics
        self.clips_with_masks = sum(1 for s in self.samples if s["has_any_mask"])
        print(f"[MultiTask {split}] {len(self.samples)} clips, {self.clips_with_masks} with mask supervision ({100*self.clips_with_masks/len(self.samples):.1f}%)")

    def _build_indices(self):
        """Build indices for frames and masks."""
        # Frame index per video
        self.video_frames = {}
        for vid in self.video_ids:
            vid_meta = self.metadata[self.metadata["vid"] == vid]
            frames = sorted(vid_meta["frame"].unique())
            self.video_frames[vid] = frames

        # GT mask index
        self.gt_masks = set()
        if self.semseg_dir.exists():
            for f in self.semseg_dir.glob("*.png"):
                self.gt_masks.add(f.stem)

        # Synthetic mask index
        self.synthetic_masks = set()
        if self.use_synthetic_masks and self.synthetic_dir.exists():
            for split_dir in ["train", "val"]:
                sem_dir = self.synthetic_dir / split_dir / "semantic"
                if sem_dir.exists():
                    for f in sem_dir.glob("*.png"):
                        self.synthetic_masks.add(f.stem)

        print(f"[MultiTask] GT masks: {len(self.gt_masks)}, Synthetic masks: {len(self.synthetic_masks)}")

    def _create_samples(self):
        """Create list of samples with mask availability info."""
        self.samples = []

        for idx, row in self.metadata.iterrows():
            vid = int(row["vid"])
            frame = int(row["frame"])

            # Binarize CVS labels
            labels = np.array([
                1.0 if row["C1"] >= 0.5 else 0.0,
                1.0 if row["C2"] >= 0.5 else 0.0,
                1.0 if row["C3"] >= 0.5 else 0.0,
            ], dtype=np.float32)

            # Check frame file exists
            frame_file = self.frame_dir / f"{vid}_{frame}.jpg"
            if not frame_file.exists():
                continue

            # Get frame sequence for this clip
            frame_sequence = self._get_frame_sequence(vid, frame)

            # Check which frames have masks
            mask_info = []
            for i, f in enumerate(frame_sequence):
                mask_name = f"{vid}_{f}"
                mask_type = None
                if mask_name in self.gt_masks:
                    mask_type = "gt"
                elif mask_name in self.synthetic_masks:
                    mask_type = "synthetic"

                if mask_type:
                    mask_info.append({"frame_idx": i, "frame_num": f, "type": mask_type})

            self.samples.append({
                "video_id": vid,
                "center_frame": frame,
                "frame_sequence": frame_sequence,
                "labels": labels,
                "mask_info": mask_info,
                "has_any_mask": len(mask_info) > 0,
            })

    def _get_frame_sequence(self, video_id: int, center_frame: int) -> List[int]:
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

    def _load_frame(self, video_id: int, frame_num: int) -> np.ndarray:
        """Load a single frame as numpy array."""
        frame_path = self.frame_dir / f"{video_id}_{frame_num}.jpg"

        if not frame_path.exists():
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        img = Image.open(frame_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def _load_mask(self, video_id: int, frame_num: int, mask_type: str) -> Optional[np.ndarray]:
        """Load segmentation mask."""
        mask_name = f"{video_id}_{frame_num}.png"

        if mask_type == "gt":
            mask_path = self.semseg_dir / mask_name
        else:
            # Try both train and val synthetic dirs
            mask_path = None
            for split_dir in ["train", "val"]:
                candidate = self.synthetic_dir / split_dir / "semantic" / mask_name
                if candidate.exists():
                    mask_path = candidate
                    break

        if mask_path is None or not mask_path.exists():
            return None

        # Load mask
        mask = Image.open(mask_path)

        # Resize to mask resolution (use NEAREST to preserve class IDs)
        mask = mask.resize((self.mask_resolution, self.mask_resolution), Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        # Remap classes
        remapped = np.zeros_like(mask)
        for src, dst in self.CLASS_MAP.items():
            remapped[mask == src] = dst

        return remapped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample.

        Returns:
            Dictionary with:
                - video: numpy array (T, H, W, C) uint8
                - labels: tensor (3,) float32 - CVS labels
                - masks: tensor (N, H, W) int64 - segmentation masks for N frames
                - mask_indices: tensor (N,) int64 - which frame indices have masks
                - has_masks: bool - whether any masks are available
                - meta: dict with video_id, center_frame
        """
        sample = self.samples[idx]
        video_id = sample["video_id"]
        center_frame = sample["center_frame"]
        frame_sequence = sample["frame_sequence"]
        labels = sample["labels"]
        mask_info = sample["mask_info"]

        # Load frames
        frames = []
        for frame_num in frame_sequence:
            frame = self._load_frame(video_id, frame_num)
            frames.append(frame)

        video = np.stack(frames, axis=0)  # (T, H, W, C)

        # Load masks for frames that have them
        masks = []
        mask_indices = []
        for info in mask_info:
            mask = self._load_mask(video_id, info["frame_num"], info["type"])
            if mask is not None:
                masks.append(mask)
                mask_indices.append(info["frame_idx"])

        # Apply augmentation (same transform to video and masks)
        if self.augment:
            if np.random.random() < self.horizontal_flip_prob:
                video = video[:, :, ::-1, :].copy()
                masks = [m[:, ::-1].copy() for m in masks]

        # Convert masks to tensor
        if masks:
            masks_tensor = torch.tensor(np.stack(masks), dtype=torch.long)
            indices_tensor = torch.tensor(mask_indices, dtype=torch.long)
        else:
            # Empty tensors if no masks
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
                "num_masks": len(masks),
            }
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for multi-task DataLoader.

    Handles variable number of masks per sample.
    """
    videos = [item["video"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    has_masks = [item["has_masks"] for item in batch]
    metas = [item["meta"] for item in batch]

    # Collect masks with batch indices
    all_masks = []
    all_mask_indices = []
    all_batch_indices = []

    for batch_idx, item in enumerate(batch):
        if item["has_masks"]:
            all_masks.append(item["masks"])
            all_mask_indices.append(item["mask_indices"])
            all_batch_indices.extend([batch_idx] * len(item["mask_indices"]))

    # Stack if we have any masks
    if all_masks:
        masks = torch.cat(all_masks, dim=0)  # (total_masks, H, W)
        mask_frame_indices = torch.cat(all_mask_indices, dim=0)  # (total_masks,)
        mask_batch_indices = torch.tensor(all_batch_indices, dtype=torch.long)  # (total_masks,)
    else:
        masks = torch.zeros((0, batch[0]["masks"].shape[1], batch[0]["masks"].shape[2]), dtype=torch.long)
        mask_frame_indices = torch.zeros((0,), dtype=torch.long)
        mask_batch_indices = torch.zeros((0,), dtype=torch.long)

    return {
        "videos": videos,
        "labels": labels,
        "masks": masks,
        "mask_frame_indices": mask_frame_indices,
        "mask_batch_indices": mask_batch_indices,
        "has_masks": has_masks,
        "metas": metas,
    }


if __name__ == "__main__":
    # Test the dataset
    print("=" * 60)
    print("Testing MultiTaskCVSDataset")
    print("=" * 60)

    dataset = MultiTaskCVSDataset(
        root_dir=r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes",
        split="train",
        num_frames=16,
        resolution=256,
        mask_resolution=64,
        use_synthetic_masks=True,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Clips with masks: {dataset.clips_with_masks}")

    # Test a few samples
    print("\nSample 0:")
    sample = dataset[0]
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Labels: {sample['labels']}")
    print(f"  Has masks: {sample['has_masks']}")
    print(f"  Masks shape: {sample['masks'].shape}")
    print(f"  Mask indices: {sample['mask_indices']}")

    # Find a sample with masks
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["has_masks"]:
            print(f"\nSample {i} (with masks):")
            print(f"  Video shape: {sample['video'].shape}")
            print(f"  Labels: {sample['labels']}")
            print(f"  Masks shape: {sample['masks'].shape}")
            print(f"  Mask indices: {sample['mask_indices']}")
            print(f"  Unique mask values: {torch.unique(sample['masks'])}")
            break

    # Test collate
    print("\nTesting collate_fn:")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))
    print(f"  Videos: {len(batch['videos'])} x {batch['videos'][0].shape}")
    print(f"  Labels: {batch['labels'].shape}")
    print(f"  Masks: {batch['masks'].shape}")
    print(f"  Mask frame indices: {batch['mask_frame_indices']}")
    print(f"  Mask batch indices: {batch['mask_batch_indices']}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
