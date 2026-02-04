"""
Improved Multi-task Dataset for V-JEPA CVS Classification + Segmentation.

Key improvements over dataset_multitask.py:
1. Centre crop to remove endoscopic black borders before resizing
2. Label consistency checking across clip frames (strict/soft/majority)
3. Configurable temporal stride (frame_step) for clip construction
4. Clip subsampling to reduce inter-clip overlap
5. Detailed statistics logging

Drop-in compatible: same __getitem__ return format and collate_fn interface.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImprovedMultiTaskCVSDataset(Dataset):
    """
    Improved dataset for multi-task learning: CVS classification + segmentation.

    Compared to MultiTaskCVSDataset:
    - Centre crops frames/masks before resizing (removes endoscopic borders)
    - Checks label consistency across all frames in a clip
    - Uses configurable stride through available_frames (frame_step)
    - Subsamples clips per video to reduce redundancy

    Label strategies:
        'center':   Use center frame label only (original behaviour)
        'strict':   Skip clips where any frame disagrees with center
        'soft':     Average raw labels across all clip frames (continuous output)
        'majority': Binarize each frame, then majority vote per criterion

    Frame step:
        Controls stride through available_frames when building a clip.
        frame_step=1: consecutive available frames (densest, original behaviour)
        frame_step=N: every Nth available frame (wider temporal context)

        In Endoscapes, available frames are 25 raw frames apart.
        So frame_step=1 -> clip spans (num_frames-1)*25 raw frames
           frame_step=5 -> clip spans (num_frames-1)*5*25 raw frames
    """

    CLASS_MAP = {
        0: 0,    # Background
        1: 1,    # Cystic Plate
        2: 2,    # Calot Triangle
        3: 3,    # Cystic Artery
        4: 4,    # Cystic Duct
        5: 0,    # Gallbladder -> Background
        6: 0,    # Tool -> Background
        255: 255,
    }
    NUM_SEG_CLASSES = 5

    VALID_STRATEGIES = ('center', 'strict', 'soft', 'majority')

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        frame_step: int = 1,
        resolution: int = 256,
        centre_crop: Optional[int] = 480,
        mask_resolution: int = 64,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        use_synthetic_masks: bool = True,
        include_gallbladder: bool = False,
        gt_masks_dir: Optional[str] = None,
        synthetic_masks_dir: Optional[str] = None,
        # --- New parameters ---
        label_strategy: str = "soft",
        clip_subsample: int = 1,
        binarize_threshold: float = 0.5,
    ):
        """
        Args:
            root_dir: Path to Endoscapes dataset root.
            split: One of "train", "val", "test".
            num_frames: Number of frames per clip (e.g. 5 or 16).
            frame_step: Stride through available_frames list when building clip.
            resolution: Target spatial resolution after crop+resize.
            centre_crop: Crop centre square of this size before resizing.
                         Set to None to disable (direct resize like original).
            mask_resolution: Resolution for segmentation masks.
            augment: Whether to apply data augmentation (train split only).
            horizontal_flip_prob: Probability of horizontal flip augmentation.
            use_synthetic_masks: Whether to use SAM2 synthetic masks as fallback.
            include_gallbladder: Whether to keep gallbladder as separate class.
            gt_masks_dir: Override GT masks directory.
            synthetic_masks_dir: Override synthetic masks directory.
            label_strategy: How to handle labels across clip frames.
                'center': center frame only (original).
                'strict': skip clips with any disagreement.
                'soft': mean of raw labels across clip.
                'majority': majority vote of binarized labels.
            clip_subsample: Keep every Nth clip per video (1=all, 4=every 4th).
            binarize_threshold: Threshold for binarizing labels (default 0.5).
        """
        assert label_strategy in self.VALID_STRATEGIES, \
            f"label_strategy must be one of {self.VALID_STRATEGIES}, got '{label_strategy}'"
        assert clip_subsample >= 1, f"clip_subsample must be >= 1, got {clip_subsample}"
        assert frame_step >= 1, f"frame_step must be >= 1, got {frame_step}"

        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resolution = resolution
        self.centre_crop = centre_crop
        self.mask_resolution = mask_resolution
        self.augment = augment and (split == "train")
        self.horizontal_flip_prob = horizontal_flip_prob
        self.use_synthetic_masks = use_synthetic_masks
        self.label_strategy = label_strategy
        self.clip_subsample = clip_subsample
        self.binarize_threshold = binarize_threshold

        # Instance-level copy of CLASS_MAP to avoid mutating the class attribute
        self.CLASS_MAP = dict(self.__class__.CLASS_MAP)
        self.NUM_SEG_CLASSES = self.__class__.NUM_SEG_CLASSES
        if include_gallbladder:
            self.CLASS_MAP[5] = 5
            self.NUM_SEG_CLASSES = 6

        # Paths
        self.frame_dir = self.root_dir / split
        self.semseg_dir = Path(gt_masks_dir) if gt_masks_dir else self.root_dir / "semseg"
        self.synthetic_dir = (
            Path(synthetic_masks_dir) if synthetic_masks_dir
            else self.root_dir / "synthetic_masks"
        )

        # Load metadata
        metadata_path = self.root_dir / "all_metadata.csv"
        self.metadata = pd.read_csv(metadata_path)

        # Load video IDs for this split
        split_file = self.root_dir / f"{split}_vids.txt"
        with open(split_file, "r") as f:
            self.video_ids = [int(float(line.strip())) for line in f.readlines()]

        self.metadata = self.metadata[
            self.metadata["vid"].isin(self.video_ids)
        ].reset_index(drop=True)

        # Build indices
        self._build_indices()
        self._build_label_lookup()

        # Create samples with filtering
        self._create_samples()

        # Print statistics
        self._print_statistics()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_indices(self):
        """Build per-video frame list and mask sets."""
        self.video_frames: Dict[int, List[int]] = {}
        for vid in self.video_ids:
            vid_meta = self.metadata[self.metadata["vid"] == vid]
            self.video_frames[vid] = sorted(vid_meta["frame"].unique())

        # GT masks
        self.gt_masks: set = set()
        if self.semseg_dir.exists():
            for f in self.semseg_dir.glob("*.png"):
                self.gt_masks.add(f.stem)

        # Synthetic masks
        self.synthetic_masks: set = set()
        if self.use_synthetic_masks and self.synthetic_dir.exists():
            for split_dir in ["train", "val"]:
                sem_dir = self.synthetic_dir / split_dir / "semantic"
                if sem_dir.exists():
                    for f in sem_dir.glob("*.png"):
                        self.synthetic_masks.add(f.stem)

        print(
            f"[Improved {self.split}] GT masks: {len(self.gt_masks)}, "
            f"Synthetic masks: {len(self.synthetic_masks)}"
        )

    def _build_label_lookup(self):
        """Build (vid, frame) -> raw [C1, C2, C3] lookup."""
        self.frame_labels: Dict[Tuple[int, int], np.ndarray] = {}
        for _, row in self.metadata.iterrows():
            vid = int(row["vid"])
            frame = int(row["frame"])
            self.frame_labels[(vid, frame)] = np.array(
                [row["C1"], row["C2"], row["C3"]], dtype=np.float32
            )

    # ------------------------------------------------------------------
    # Clip construction
    # ------------------------------------------------------------------

    def _get_frame_sequence(self, video_id: int, center_frame: int) -> List[int]:
        """
        Build a clip of num_frames frames centred on center_frame.

        Frames are selected at stride=frame_step through available_frames.
        Pads by repeating boundary frames when the window exceeds the video.
        """
        available = self.video_frames.get(video_id, [])
        if not available:
            return [center_frame] * self.num_frames

        try:
            center_idx = available.index(center_frame)
        except ValueError:
            center_idx = min(
                range(len(available)),
                key=lambda i: abs(available[i] - center_frame),
            )

        half = self.num_frames // 2
        sequence = []
        for i in range(self.num_frames):
            idx = center_idx + (i - half) * self.frame_step
            idx = max(0, min(idx, len(available) - 1))
            sequence.append(available[idx])

        return sequence

    # ------------------------------------------------------------------
    # Label consistency
    # ------------------------------------------------------------------

    def _compute_clip_labels(
        self, video_id: int, frame_sequence: List[int], center_row
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute labels for a clip according to label_strategy.

        Returns:
            (labels, agreement)  -- labels is None when strict mode rejects.
            agreement is fraction of clip frames whose binarized labels
            match the center frame.
        """
        center_raw = np.array(
            [center_row["C1"], center_row["C2"], center_row["C3"]],
            dtype=np.float32,
        )

        # Fast path: centre-only mode
        if self.label_strategy == "center":
            binary = (center_raw >= self.binarize_threshold).astype(np.float32)
            return binary, 1.0

        # Gather raw labels for every frame in the clip
        all_raw: List[np.ndarray] = []
        for f in frame_sequence:
            lbl = self.frame_labels.get((video_id, f))
            if lbl is not None:
                all_raw.append(lbl)

        if not all_raw:
            binary = (center_raw >= self.binarize_threshold).astype(np.float32)
            return binary, 1.0

        all_raw_arr = np.stack(all_raw)  # (N, 3)

        # Agreement: fraction of frames matching centre after binarization
        center_bin = (center_raw >= self.binarize_threshold).astype(float)
        all_bin = (all_raw_arr >= self.binarize_threshold).astype(float)
        agreement = float(np.mean(np.all(all_bin == center_bin, axis=1)))

        if self.label_strategy == "strict":
            if agreement < 1.0:
                return None, agreement  # reject clip
            return center_bin.astype(np.float32), agreement

        if self.label_strategy == "soft":
            # Continuous average of raw annotator-averaged labels
            soft = np.mean(all_raw_arr, axis=0).astype(np.float32)
            return soft, agreement

        if self.label_strategy == "majority":
            votes = (all_raw_arr >= self.binarize_threshold).astype(float)
            majority = (np.mean(votes, axis=0) >= 0.5).astype(np.float32)
            return majority, agreement

        # Fallback (shouldn't reach here)
        return center_bin.astype(np.float32), agreement

    # ------------------------------------------------------------------
    # Sample creation with filtering
    # ------------------------------------------------------------------

    def _create_samples(self):
        """Create filtered sample list with label checking and subsampling."""
        self.samples: List[Dict] = []
        self.stats = {
            "total_candidates": 0,
            "skipped_missing_frame": 0,
            "skipped_label_inconsistency": 0,
            "skipped_subsample": 0,
            "kept": 0,
            "agreements": [],
            "label_sums": np.zeros(3, dtype=np.float64),
            "label_positive_count": np.zeros(3, dtype=int),
            "total_labelled": 0,
        }

        # Collect per-video to apply subsampling
        video_samples: Dict[int, List[Dict]] = {}

        for _, row in self.metadata.iterrows():
            self.stats["total_candidates"] += 1
            vid = int(row["vid"])
            frame = int(row["frame"])

            # Check centre frame file exists
            frame_file = self.frame_dir / f"{vid}_{frame}.jpg"
            if not frame_file.exists():
                self.stats["skipped_missing_frame"] += 1
                continue

            # Build clip
            frame_sequence = self._get_frame_sequence(vid, frame)

            # Label consistency
            labels, agreement = self._compute_clip_labels(vid, frame_sequence, row)
            self.stats["agreements"].append(agreement)

            if labels is None:
                self.stats["skipped_label_inconsistency"] += 1
                continue

            # Mask info
            mask_info = []
            for i, f in enumerate(frame_sequence):
                mask_name = f"{vid}_{f}"
                mask_type = None
                if mask_name in self.gt_masks:
                    mask_type = "gt"
                elif mask_name in self.synthetic_masks:
                    mask_type = "synthetic"
                if mask_type:
                    mask_info.append(
                        {"frame_idx": i, "frame_num": f, "type": mask_type}
                    )

            sample = {
                "video_id": vid,
                "center_frame": frame,
                "frame_sequence": frame_sequence,
                "labels": labels,
                "agreement": agreement,
                "mask_info": mask_info,
                "has_any_mask": len(mask_info) > 0,
            }

            if vid not in video_samples:
                video_samples[vid] = []
            video_samples[vid].append(sample)

        # Subsample per video (keep every Nth clip, preserving temporal order)
        for vid in sorted(video_samples.keys()):
            clips = video_samples[vid]
            for i, sample in enumerate(clips):
                if i % self.clip_subsample == 0:
                    self.samples.append(sample)
                    lbl = sample["labels"]
                    self.stats["label_sums"] += lbl.astype(np.float64)
                    self.stats["label_positive_count"] += (
                        lbl >= self.binarize_threshold
                    ).astype(int)
                    self.stats["total_labelled"] += 1
                else:
                    self.stats["skipped_subsample"] += 1

        self.stats["kept"] = len(self.samples)
        self.clips_with_masks = sum(
            1 for s in self.samples if s["has_any_mask"]
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _print_statistics(self):
        """Print detailed dataset statistics."""
        s = self.stats
        print(f"\n{'='*65}")
        print(f"  Improved Dataset [{self.split}]")
        print(f"{'='*65}")
        print(f"  Config:")
        print(f"    label_strategy  = {self.label_strategy}")
        print(f"    num_frames      = {self.num_frames}")
        print(f"    frame_step      = {self.frame_step}")
        print(f"    clip_subsample  = {self.clip_subsample}")
        if self.centre_crop:
            print(f"    centre_crop     = {self.centre_crop} -> {self.resolution}")
        else:
            print(f"    centre_crop     = disabled (direct resize to {self.resolution})")

        # Temporal span estimate
        if self.video_frames:
            first_vid = next(iter(self.video_frames))
            frames = self.video_frames[first_vid]
            if len(frames) >= 2:
                raw_step = frames[1] - frames[0]
                clip_span = (self.num_frames - 1) * self.frame_step * raw_step
                print(f"    raw_frame_step  = {raw_step} (between available frames)")
                print(
                    f"    clip_span       = {clip_span} raw frames "
                    f"({clip_span / 30:.1f}s @ 30fps)"
                )

        print(f"\n  Filtering:")
        print(f"    Total candidates           : {s['total_candidates']}")
        print(f"    Skipped (missing frame)    : {s['skipped_missing_frame']}")
        print(f"    Skipped (label mismatch)   : {s['skipped_label_inconsistency']}")
        print(f"    Skipped (subsample)        : {s['skipped_subsample']}")
        print(f"    Kept                       : {s['kept']}")
        if s["total_candidates"] > 0:
            rate = 100 * s["kept"] / s["total_candidates"]
            print(f"    Keep rate                  : {rate:.1f}%")
        print(
            f"    Clips with masks           : {self.clips_with_masks} "
            f"({100 * self.clips_with_masks / max(len(self.samples), 1):.1f}%)"
        )

        # Label distribution
        n = s["total_labelled"]
        if n > 0:
            print(f"\n  Label distribution ({n} clips):")
            for i, name in enumerate(["C1", "C2", "C3"]):
                pos = s["label_positive_count"][i]
                avg = s["label_sums"][i] / n
                print(
                    f"    {name}: {pos}/{n} positive ({100 * pos / n:.1f}%), "
                    f"mean value = {avg:.3f}"
                )

        # Agreement
        agreements = np.array(s["agreements"]) if s["agreements"] else np.array([])
        if len(agreements) > 0:
            perfect = np.sum(agreements == 1.0)
            print(f"\n  Label agreement within clips (before filtering):")
            print(f"    Mean   : {np.mean(agreements):.3f}")
            print(f"    Median : {np.median(agreements):.3f}")
            print(f"    Min    : {np.min(agreements):.3f}")
            print(
                f"    100%   : {perfect}/{len(agreements)} "
                f"({100 * perfect / len(agreements):.1f}%)"
            )

        print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Frame / mask loading
    # ------------------------------------------------------------------

    def _centre_crop_pil(self, img: Image.Image) -> Image.Image:
        """Centre-crop a PIL image to a square of min(centre_crop, w, h)."""
        if not self.centre_crop:
            return img
        w, h = img.size
        crop = min(self.centre_crop, w, h)
        left = (w - crop) // 2
        top = (h - crop) // 2
        return img.crop((left, top, left + crop, top + crop))

    def _load_frame(self, video_id: int, frame_num: int) -> np.ndarray:
        """Load a single frame: centre crop + resize to resolution."""
        frame_path = self.frame_dir / f"{video_id}_{frame_num}.jpg"
        if not frame_path.exists():
            return np.zeros(
                (self.resolution, self.resolution, 3), dtype=np.uint8
            )

        img = Image.open(frame_path).convert("RGB")
        img = self._centre_crop_pil(img)
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def _load_mask(
        self, video_id: int, frame_num: int, mask_type: str
    ) -> Optional[np.ndarray]:
        """Load segmentation mask with centre crop + resize."""
        mask_name = f"{video_id}_{frame_num}.png"

        if mask_type == "gt":
            mask_path = self.semseg_dir / mask_name
        else:
            mask_path = None
            splits_to_check = [self.split] + [
                s for s in ["train", "val"] if s != self.split
            ]
            for split_dir in splits_to_check:
                candidate = (
                    self.synthetic_dir / split_dir / "semantic" / mask_name
                )
                if candidate.exists():
                    mask_path = candidate
                    break

        if mask_path is None or not mask_path.exists():
            return None

        mask = Image.open(mask_path)
        mask = self._centre_crop_pil(mask)
        mask = mask.resize(
            (self.mask_resolution, self.mask_resolution), Image.NEAREST
        )
        mask = np.array(mask, dtype=np.uint8)

        remapped = np.zeros_like(mask)
        for src, dst in self.CLASS_MAP.items():
            remapped[mask == src] = dst
        return remapped

    # ------------------------------------------------------------------
    # __getitem__ / __len__  (same return format as MultiTaskCVSDataset)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dict with:
            video:        numpy (T, H, W, C) uint8
            labels:       tensor (3,) float32
            masks:        tensor (N, H, W) int64
            mask_indices: tensor (N,) int64
            has_masks:    bool
            meta:         dict with video_id, center_frame, agreement
        """
        sample = self.samples[idx]
        video_id = sample["video_id"]
        frame_sequence = sample["frame_sequence"]

        # Load frames
        frames = [self._load_frame(video_id, fn) for fn in frame_sequence]
        video = np.stack(frames, axis=0)

        # Load masks
        masks = []
        mask_indices = []
        for info in sample["mask_info"]:
            m = self._load_mask(video_id, info["frame_num"], info["type"])
            if m is not None:
                masks.append(m)
                mask_indices.append(info["frame_idx"])

        # Augmentation
        if self.augment:
            if np.random.random() < self.horizontal_flip_prob:
                video = video[:, :, ::-1, :].copy()
                masks = [m[:, ::-1].copy() for m in masks]

        # Pack masks
        if masks:
            masks_tensor = torch.tensor(np.stack(masks), dtype=torch.long)
            indices_tensor = torch.tensor(mask_indices, dtype=torch.long)
        else:
            masks_tensor = torch.zeros(
                (0, self.mask_resolution, self.mask_resolution), dtype=torch.long
            )
            indices_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            "video": video,
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
            "masks": masks_tensor,
            "mask_indices": indices_tensor,
            "has_masks": len(masks) > 0,
            "meta": {
                "video_id": video_id,
                "center_frame": sample["center_frame"],
                "num_masks": len(masks),
                "agreement": sample["agreement"],
            },
        }


# ------------------------------------------------------------------
# Collate (identical interface to dataset_multitask.collate_fn)
# ------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate handling variable mask counts per sample."""
    videos = [item["video"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    has_masks = [item["has_masks"] for item in batch]
    metas = [item["meta"] for item in batch]

    all_masks = []
    all_mask_indices = []
    all_batch_indices = []

    for batch_idx, item in enumerate(batch):
        if item["has_masks"]:
            all_masks.append(item["masks"])
            all_mask_indices.append(item["mask_indices"])
            all_batch_indices.extend(
                [batch_idx] * len(item["mask_indices"])
            )

    if all_masks:
        masks = torch.cat(all_masks, dim=0)
        mask_frame_indices = torch.cat(all_mask_indices, dim=0)
        mask_batch_indices = torch.tensor(all_batch_indices, dtype=torch.long)
    else:
        h = batch[0]["masks"].shape[1]
        w = batch[0]["masks"].shape[2]
        masks = torch.zeros((0, h, w), dtype=torch.long)
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


# ------------------------------------------------------------------
# Comparison helper
# ------------------------------------------------------------------

def compare_datasets(root_dir: str, split: str = "train", **shared_kwargs):
    """
    Instantiate original and improved datasets side-by-side and print
    a comparison table.  Does NOT load pixel data -- only builds sample lists.
    """
    from dataset_multitask import MultiTaskCVSDataset

    print("\n" + "#" * 65)
    print(f"  DATASET COMPARISON  [{split}]")
    print("#" * 65)

    # Original
    print("\n>>> Original (MultiTaskCVSDataset):")
    orig = MultiTaskCVSDataset(
        root_dir=root_dir,
        split=split,
        num_frames=16,
        frame_step=25,
        **shared_kwargs,
    )

    # Improved -- soft labels, subsample=4
    print("\n>>> Improved (soft, subsample=4, frame_step=5):")
    imp_soft = ImprovedMultiTaskCVSDataset(
        root_dir=root_dir,
        split=split,
        num_frames=16,
        frame_step=5,
        centre_crop=480,
        label_strategy="soft",
        clip_subsample=4,
        **shared_kwargs,
    )

    # Improved -- strict
    print("\n>>> Improved (strict, subsample=1, frame_step=1):")
    imp_strict = ImprovedMultiTaskCVSDataset(
        root_dir=root_dir,
        split=split,
        num_frames=16,
        frame_step=1,
        centre_crop=480,
        label_strategy="strict",
        clip_subsample=1,
        **shared_kwargs,
    )

    # Improved -- majority, 5 frames
    print("\n>>> Improved (majority, 5 frames, subsample=1, frame_step=1):")
    imp_5f = ImprovedMultiTaskCVSDataset(
        root_dir=root_dir,
        split=split,
        num_frames=5,
        frame_step=1,
        centre_crop=480,
        label_strategy="majority",
        clip_subsample=1,
        **shared_kwargs,
    )

    # Summary table
    print("\n" + "=" * 65)
    print(f"  {'Config':<40} {'Clips':>8}  {'w/ masks':>8}")
    print("-" * 65)
    configs = [
        ("Original (16f, step=25, center)",       orig),
        ("Soft (16f, step=5, sub=4, crop=480)",   imp_soft),
        ("Strict (16f, step=1, sub=1, crop=480)", imp_strict),
        ("Majority (5f, step=1, sub=1, crop=480)", imp_5f),
    ]
    for name, ds in configs:
        print(f"  {name:<40} {len(ds):>8}  {ds.clips_with_masks:>8}")
    print("=" * 65 + "\n")


# ------------------------------------------------------------------
# Main: self-test and comparison
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Detect environment
    if os.path.exists("/workspace/vjepa/data/endoscapes"):
        ROOT = "/workspace/vjepa/data/endoscapes"
        EXTRA = dict(
            gt_masks_dir="/workspace/vjepa/data/endoscapes/semseg",
            synthetic_masks_dir="/workspace/vjepa/data/synthetic_masks",
        )
    elif os.path.exists(
        r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
    ):
        ROOT = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
        EXTRA = {}
    else:
        print("ERROR: Endoscapes dataset not found.")
        sys.exit(1)

    shared = dict(
        resolution=256,
        mask_resolution=64,
        use_synthetic_masks=True,
        **EXTRA,
    )

    # --- Run comparison ---
    compare_datasets(ROOT, split="train", **shared)

    # --- Quick functional test of improved dataset ---
    print("\n" + "=" * 65)
    print("  Functional Test: ImprovedMultiTaskCVSDataset")
    print("=" * 65)

    ds = ImprovedMultiTaskCVSDataset(
        root_dir=ROOT,
        split="train",
        num_frames=16,
        frame_step=1,
        centre_crop=480,
        label_strategy="soft",
        clip_subsample=1,
        **shared,
    )

    print(f"\nDataset length: {len(ds)}")

    # Load first sample
    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  video.shape  = {sample['video'].shape}")
    print(f"  labels       = {sample['labels'].tolist()}")
    print(f"  has_masks    = {sample['has_masks']}")
    print(f"  masks.shape  = {sample['masks'].shape}")
    print(f"  agreement    = {sample['meta']['agreement']:.3f}")

    # Find a sample with masks
    for i in range(min(len(ds), 500)):
        s = ds[i]
        if s["has_masks"]:
            print(f"\nSample {i} (with masks):")
            print(f"  video.shape  = {s['video'].shape}")
            print(f"  labels       = {s['labels'].tolist()}")
            print(f"  masks.shape  = {s['masks'].shape}")
            print(f"  mask_indices = {s['mask_indices'].tolist()}")
            print(f"  mask classes = {torch.unique(s['masks']).tolist()}")
            break

    # Test collate_fn
    print("\nCollate test (batch_size=4):")
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))
    print(f"  videos       : {len(batch['videos'])} x {batch['videos'][0].shape}")
    print(f"  labels       : {batch['labels'].shape}")
    print(f"  masks        : {batch['masks'].shape}")
    print(f"  frame_indices: {batch['mask_frame_indices'].shape}")
    print(f"  batch_indices: {batch['mask_batch_indices'].shape}")

    print("\n" + "=" * 65)
    print("  All tests passed!")
    print("=" * 65)
