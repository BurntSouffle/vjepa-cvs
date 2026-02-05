# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Surgical video dataset for anatomy-guided V-JEPA pretraining

import os
import random
import warnings
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

logger = getLogger()


def make_surgical_videodataset(
    endoscapes_config: dict,
    sages_config: dict,
    batch_size: int,
    frames_per_clip: int = 16,
    crop_size: int = 256,
    centre_crop: int = 480,
    transform=None,
    rank: int = 0,
    world_size: int = 1,
    collator=None,
    drop_last: bool = True,
    num_workers: int = 8,
    pin_mem: bool = True,
):
    """
    Create surgical video dataset and dataloader.

    Args:
        endoscapes_config: Dict with frames_dir, masks_dir
        sages_config: Dict with frames_dir, masks_dir
        batch_size: Batch size per GPU
        frames_per_clip: Number of frames per clip
        crop_size: Final crop size after resize
        centre_crop: Centre crop size before resize
        transform: Video transforms
        rank: Process rank for distributed training
        world_size: Total number of processes
        collator: Custom collate function (anatomy-guided mask collator)
        drop_last: Drop last incomplete batch
        num_workers: DataLoader workers
        pin_mem: Pin memory for faster GPU transfer

    Returns:
        dataset, dataloader, sampler
    """
    dataset = SurgicalVideoDataset(
        endoscapes_frames_dir=endoscapes_config.get('frames_dir'),
        endoscapes_masks_dir=endoscapes_config.get('masks_dir'),
        sages_frames_dir=sages_config.get('frames_dir'),
        sages_masks_dir=sages_config.get('masks_dir'),
        frames_per_clip=frames_per_clip,
        crop_size=crop_size,
        centre_crop=centre_crop,
        transform=transform,
    )

    logger.info(f'SurgicalVideoDataset created with {len(dataset)} samples')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    logger.info('SurgicalVideoDataset data loader created')
    return dataset, data_loader, dist_sampler


class SurgicalVideoDataset(Dataset):
    """
    Dataset for surgical video pretraining with anatomy masks.

    Combines Endoscapes and SAGES datasets. Each sample returns:
    - Video clip (frames_per_clip consecutive frames)
    - Anatomy probability map (for anatomy-guided masking)

    Since surgical videos are extracted as individual frames (not video files),
    we load consecutive frames to create clips.
    """

    # Default class weights for anatomy probability map
    DEFAULT_CLASS_WEIGHTS = {
        0: 0.1,    # background - low priority
        1: 1.0,    # cystic_plate / anatomy - HIGH
        2: 1.0,    # calot_triangle - HIGH
        3: 1.0,    # cystic_artery - HIGH
        4: 1.0,    # cystic_duct - HIGH
        5: 0.8,    # gallbladder - high
        6: 0.3,    # tool - medium
        255: 0.0,  # ignore
    }

    def __init__(
        self,
        endoscapes_frames_dir: str,
        endoscapes_masks_dir: str,
        sages_frames_dir: str,
        sages_masks_dir: str,
        frames_per_clip: int = 16,
        crop_size: int = 256,
        centre_crop: int = 480,
        transform=None,
        class_weights: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize surgical video dataset.

        Args:
            endoscapes_frames_dir: Root dir containing train/, val/, test/ frame subdirs
            endoscapes_masks_dir: Root dir containing train/, val/, test/ mask subdirs
            sages_frames_dir: Directory containing SAGES frames
            sages_masks_dir: Directory containing SAGES masks
            frames_per_clip: Number of frames per video clip
            crop_size: Final crop/resize size
            centre_crop: Centre crop size (removes black borders)
            transform: Video augmentation transforms
            class_weights: Optional custom weights for anatomy map
        """
        self.frames_per_clip = frames_per_clip
        self.crop_size = crop_size
        self.centre_crop = centre_crop
        self.transform = transform
        self.class_weights = class_weights or self.DEFAULT_CLASS_WEIGHTS

        # Collect all video clips
        self.clips = []

        # Load Endoscapes
        if endoscapes_frames_dir and os.path.exists(endoscapes_frames_dir):
            self._load_endoscapes(endoscapes_frames_dir, endoscapes_masks_dir)

        # Load SAGES
        if sages_frames_dir and os.path.exists(sages_frames_dir):
            self._load_sages(sages_frames_dir, sages_masks_dir)

        logger.info(f"Loaded {len(self.clips)} video clips total")
        logger.info(f"  Endoscapes: {sum(1 for c in self.clips if c['source'] == 'endoscapes')}")
        logger.info(f"  SAGES: {sum(1 for c in self.clips if c['source'] == 'sages')}")

    def _load_endoscapes(self, frames_dir: str, masks_dir: str):
        """Load Endoscapes clips from frame directories."""
        frames_dir = Path(frames_dir)
        masks_dir = Path(masks_dir)

        # Group frames by video ID
        video_frames = {}  # video_id -> [(frame_num, frame_path, mask_path), ...]

        for split in ['train', 'val']:
            split_frames_dir = frames_dir / split
            split_masks_dir = masks_dir / split / 'semantic'

            if not split_frames_dir.exists():
                continue

            for frame_path in split_frames_dir.glob('*.jpg'):
                # Parse filename: {video_id}_{frame_num}.jpg
                name = frame_path.stem
                parts = name.rsplit('_', 1)
                if len(parts) != 2:
                    continue

                video_id = parts[0]
                try:
                    frame_num = int(parts[1])
                except ValueError:
                    continue

                # Find corresponding mask
                mask_path = split_masks_dir / f'{name}.png'
                if not mask_path.exists():
                    mask_path = None

                if video_id not in video_frames:
                    video_frames[video_id] = []
                video_frames[video_id].append((frame_num, str(frame_path), str(mask_path) if mask_path else None))

        # Create clips from consecutive frames
        for video_id, frames in video_frames.items():
            # Sort by frame number
            frames.sort(key=lambda x: x[0])

            # Create clips
            if len(frames) >= self.frames_per_clip:
                for i in range(len(frames) - self.frames_per_clip + 1):
                    clip_frames = frames[i:i + self.frames_per_clip]
                    self.clips.append({
                        'frames': [f[1] for f in clip_frames],
                        'masks': [f[2] for f in clip_frames],
                        'video_id': video_id,
                        'source': 'endoscapes',
                    })

        logger.info(f"Loaded {len(video_frames)} Endoscapes videos")

    def _load_sages(self, frames_dir: str, masks_dir: str):
        """Load SAGES clips from frame directory."""
        frames_dir = Path(frames_dir)
        masks_dir = Path(masks_dir)

        # Group frames by video ID (UUID)
        video_frames = {}

        for frame_path in frames_dir.glob('*.jpg'):
            # Parse filename: {uuid}_{frame_num}.jpg
            name = frame_path.stem
            parts = name.rsplit('_', 1)
            if len(parts) != 2:
                continue

            video_id = parts[0]
            try:
                frame_num = int(parts[1])
            except ValueError:
                continue

            # Find corresponding mask
            mask_path = masks_dir / f'{name}.png'
            if not mask_path.exists():
                mask_path = None

            if video_id not in video_frames:
                video_frames[video_id] = []
            video_frames[video_id].append((frame_num, str(frame_path), str(mask_path) if mask_path else None))

        # Create clips from consecutive frames
        for video_id, frames in video_frames.items():
            # Sort by frame number
            frames.sort(key=lambda x: x[0])

            # Create clips (with stride to reduce redundancy)
            if len(frames) >= self.frames_per_clip:
                stride = max(1, self.frames_per_clip // 2)  # 50% overlap
                for i in range(0, len(frames) - self.frames_per_clip + 1, stride):
                    clip_frames = frames[i:i + self.frames_per_clip]
                    self.clips.append({
                        'frames': [f[1] for f in clip_frames],
                        'masks': [f[2] for f in clip_frames],
                        'video_id': video_id,
                        'source': 'sages',
                    })

        logger.info(f"Loaded {len(video_frames)} SAGES videos")

    def _centre_crop_pil(self, img: Image.Image) -> Image.Image:
        """Apply centre crop to PIL image."""
        w, h = img.size
        crop_size = min(self.centre_crop, w, h)

        left = (w - crop_size) // 2
        top = (h - crop_size) // 2

        return img.crop((left, top, left + crop_size, top + crop_size))

    def _load_frame(self, frame_path: str) -> np.ndarray:
        """Load and preprocess a single frame.

        Returns:
            frame: numpy array [H, W, C] uint8 where C=3 (RGB)
                   This is the format expected by JEPA video transforms.
        """
        img = Image.open(frame_path).convert('RGB')

        # Centre crop
        if self.centre_crop:
            img = self._centre_crop_pil(img)

        # Resize to crop_size x crop_size
        img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)

        # Convert to numpy array [H, W, C] with dtype uint8
        # PIL Image.convert('RGB') ensures 3 channels
        # np.array on PIL image gives [H, W, C] format
        frame = np.array(img, dtype=np.uint8)

        # Verify shape is [H, W, 3]
        assert frame.shape == (self.crop_size, self.crop_size, 3), \
            f"Expected frame shape ({self.crop_size}, {self.crop_size}, 3), got {frame.shape}"

        return frame

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess a segmentation mask."""
        if mask_path is None or not os.path.exists(mask_path):
            # Return None if no mask available
            return None

        mask = Image.open(mask_path)

        # Centre crop (same as frame)
        if self.centre_crop:
            mask = self._centre_crop_pil(mask)

        # Resize with nearest neighbor to preserve class indices
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)

        return torch.tensor(np.array(mask), dtype=torch.long)

    def _mask_to_anatomy_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert segmentation mask to anatomy probability map."""
        if mask is None:
            return None

        anatomy_map = torch.zeros_like(mask, dtype=torch.float32)

        for class_id, weight in self.class_weights.items():
            anatomy_map[mask == class_id] = weight

        return anatomy_map

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load a video clip and its anatomy map.

        Returns:
            clip: (C, T, H, W) video tensor after transform, or (T, H, W, C) numpy if no transform
            anatomy_map: (H, W) anatomy probability map (from middle frame mask)
            clip_info: String with clip metadata
        """
        clip_data = self.clips[idx]

        # Load frames as numpy arrays [H, W, C] uint8
        frames = []
        for frame_path in clip_data['frames']:
            frame = self._load_frame(frame_path)
            # Ensure frame is [H, W, C] format
            if frame.ndim == 3 and frame.shape[2] == 3:
                frames.append(frame)
            elif frame.ndim == 3 and frame.shape[0] == 3:
                # Wrong format [C, H, W], transpose to [H, W, C]
                frames.append(frame.transpose(1, 2, 0))
            else:
                raise ValueError(f"Unexpected frame shape: {frame.shape}")

        # Stack to [T, H, W, C] - format expected by JEPA transforms
        clip = np.stack(frames, axis=0)

        # Verify shape is [T, H, W, C] where C=3
        if clip.shape[-1] != 3:
            raise ValueError(f"Expected clip shape [T, H, W, 3], got {clip.shape}")

        # Load mask for middle frame (use for anatomy guidance)
        middle_idx = len(clip_data['masks']) // 2
        mask_path = clip_data['masks'][middle_idx]
        mask = self._load_mask(mask_path)
        anatomy_map = self._mask_to_anatomy_map(mask)

        # Apply transforms if available
        # Transform expects [T, H, W, C] numpy and returns [C, T, H, W] tensor
        if self.transform is not None:
            clip = self.transform(clip)

        clip_info = f"{clip_data['source']}_{clip_data['video_id']}"

        return clip, anatomy_map, clip_info


class SurgicalVideoCollator:
    """
    Custom collator that separates video clips and anatomy maps.

    This wraps the anatomy-guided mask collator to handle the surgical dataset format.
    """

    def __init__(self, mask_collator):
        """
        Args:
            mask_collator: AnatomyGuidedMaskCollator instance
        """
        self.mask_collator = mask_collator

    def __call__(self, batch):
        """
        Collate batch and generate masks.

        Args:
            batch: List of (clip, anatomy_map, clip_info) tuples

        Returns:
            videos: (B, T, C, H, W) video tensor
            masks_enc: List of encoder masks
            masks_pred: List of predictor masks
            clip_infos: List of clip info strings
        """
        clips, anatomy_maps, clip_infos = zip(*batch)

        # Stack videos
        videos = torch.stack(clips, dim=0)  # (B, T, C, H, W)

        # Filter out None anatomy maps
        valid_anatomy_maps = [am for am in anatomy_maps if am is not None]
        if len(valid_anatomy_maps) == len(anatomy_maps):
            anatomy_maps_tensor = torch.stack(valid_anatomy_maps, dim=0)
        else:
            # Some masks missing, pass None to use random masking
            anatomy_maps_tensor = None

        # Generate masks using anatomy-guided collator
        # Note: We need to adapt the collator call
        if self.mask_collator is not None:
            _, masks_enc, masks_pred = self.mask_collator(
                clips,
                anatomy_maps=valid_anatomy_maps if valid_anatomy_maps else None
            )
        else:
            masks_enc, masks_pred = None, None

        return videos, masks_enc, masks_pred, list(clip_infos)


def collate_surgical_videos(batch):
    """
    Simple collate function for surgical video dataset.

    Args:
        batch: List of (clip, anatomy_map, clip_info) tuples

    Returns:
        videos: (B, T, C, H, W) video tensor
        anatomy_maps: List of anatomy maps (may contain None)
        clip_infos: List of clip info strings
    """
    clips, anatomy_maps, clip_infos = zip(*batch)

    videos = torch.stack(clips, dim=0)

    return videos, list(anatomy_maps), list(clip_infos)
