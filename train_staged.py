"""
Staged Fine-Tuning Training Script (Exp9)
=========================================
Two-stage approach:
  Stage 1: Train heads only with frozen backbone
  Stage 2: Minimal backbone fine-tuning with very low LR

This preserves V-JEPA's pretrained representations while adapting to surgical domain.
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multitask import MultiTaskCVSDataset, collate_fn
from model_multitask import VJEPA_MultiTask
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)


class AugmentedMultiTaskDataset(MultiTaskCVSDataset):
    """
    MultiTask dataset with aggressive data augmentation.

    Adds:
    - Random rotation
    - Color jitter
    - Random erasing
    - Gaussian blur
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        frame_step: int = 25,
        resolution: int = 256,
        mask_resolution: int = 64,
        augment: bool = False,
        use_synthetic_masks: bool = True,
        gt_masks_dir: Optional[str] = None,
        synthetic_masks_dir: Optional[str] = None,
        # Augmentation params
        horizontal_flip_prob: float = 0.5,
        rotation_degrees: float = 15.0,
        color_jitter: Optional[Dict] = None,
        random_erasing_prob: float = 0.2,
        gaussian_blur_prob: float = 0.1,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        super().__init__(
            root_dir=root_dir,
            split=split,
            num_frames=num_frames,
            frame_step=frame_step,
            resolution=resolution,
            mask_resolution=mask_resolution,
            augment=augment,
            horizontal_flip_prob=horizontal_flip_prob,
            use_synthetic_masks=use_synthetic_masks,
            gt_masks_dir=gt_masks_dir,
            synthetic_masks_dir=synthetic_masks_dir,
        )

        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter or {}
        self.random_erasing_prob = random_erasing_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_sigma = gaussian_blur_sigma

    def _apply_color_jitter(self, frame: np.ndarray) -> np.ndarray:
        """Apply color jitter to a frame."""
        if not self.color_jitter:
            return frame

        frame = frame.astype(np.float32)

        # Brightness
        if 'brightness' in self.color_jitter:
            factor = 1.0 + np.random.uniform(-self.color_jitter['brightness'],
                                              self.color_jitter['brightness'])
            frame = frame * factor

        # Contrast
        if 'contrast' in self.color_jitter:
            factor = 1.0 + np.random.uniform(-self.color_jitter['contrast'],
                                              self.color_jitter['contrast'])
            mean = frame.mean()
            frame = (frame - mean) * factor + mean

        # Saturation
        if 'saturation' in self.color_jitter:
            factor = 1.0 + np.random.uniform(-self.color_jitter['saturation'],
                                              self.color_jitter['saturation'])
            gray = frame.mean(axis=-1, keepdims=True)
            frame = frame * factor + gray * (1 - factor)

        # Hue (simplified - shift channels)
        if 'hue' in self.color_jitter:
            shift = np.random.uniform(-self.color_jitter['hue'], self.color_jitter['hue'])
            # Simple hue shift approximation
            frame = np.roll(frame, int(shift * frame.shape[-1]), axis=-1)

        return np.clip(frame, 0, 255).astype(np.uint8)

    def _apply_random_erasing(self, frame: np.ndarray) -> np.ndarray:
        """Randomly erase a rectangular patch."""
        if np.random.random() > self.random_erasing_prob:
            return frame

        h, w = frame.shape[:2]

        # Random patch size (5-20% of image)
        area_ratio = np.random.uniform(0.05, 0.20)
        aspect_ratio = np.random.uniform(0.5, 2.0)

        area = h * w * area_ratio
        patch_h = int(np.sqrt(area / aspect_ratio))
        patch_w = int(np.sqrt(area * aspect_ratio))

        patch_h = min(patch_h, h - 1)
        patch_w = min(patch_w, w - 1)

        # Random position
        top = np.random.randint(0, h - patch_h)
        left = np.random.randint(0, w - patch_w)

        # Fill with random values or mean
        frame = frame.copy()
        frame[top:top+patch_h, left:left+patch_w] = np.random.randint(0, 255, (patch_h, patch_w, 3))

        return frame

    def _apply_gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        if np.random.random() > self.gaussian_blur_prob:
            return frame

        from scipy.ndimage import gaussian_filter

        sigma = np.random.uniform(*self.gaussian_blur_sigma)
        frame = gaussian_filter(frame.astype(np.float32), sigma=(sigma, sigma, 0))
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _apply_rotation(self, frame: np.ndarray, mask: Optional[np.ndarray], angle: float):
        """Apply rotation to frame and mask."""
        from scipy.ndimage import rotate as scipy_rotate

        # Rotate frame
        frame_rotated = scipy_rotate(frame, angle, reshape=False, order=1, mode='reflect')
        frame_rotated = np.clip(frame_rotated, 0, 255).astype(np.uint8)

        # Rotate mask (nearest neighbor to preserve class IDs)
        if mask is not None:
            mask_rotated = scipy_rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0)
            mask_rotated = mask_rotated.astype(np.uint8)
        else:
            mask_rotated = None

        return frame_rotated, mask_rotated

    def __getitem__(self, idx: int) -> Dict:
        """Get sample with augmentation."""
        # Get base sample
        sample = super().__getitem__(idx)

        if not self.augment:
            return sample

        video = sample["video"]  # (T, H, W, C)
        masks = sample["masks"]  # (N, H, W) tensor

        # Determine rotation angle (same for all frames)
        rotation_angle = 0
        if self.rotation_degrees > 0:
            rotation_angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)

        # Apply augmentations to each frame
        augmented_frames = []
        for t in range(len(video)):
            frame = video[t]

            # Color jitter
            frame = self._apply_color_jitter(frame)

            # Random erasing
            frame = self._apply_random_erasing(frame)

            # Gaussian blur
            frame = self._apply_gaussian_blur(frame)

            # Rotation (applied consistently)
            if rotation_angle != 0:
                frame, _ = self._apply_rotation(frame, None, rotation_angle)

            augmented_frames.append(frame)

        video = np.stack(augmented_frames, axis=0)

        # Apply rotation to masks
        if rotation_angle != 0 and len(masks) > 0:
            rotated_masks = []
            for i in range(len(masks)):
                mask = masks[i].numpy()
                _, mask_rotated = self._apply_rotation(np.zeros_like(video[0]), mask, rotation_angle)
                rotated_masks.append(torch.tensor(mask_rotated, dtype=torch.long))
            masks = torch.stack(rotated_masks, dim=0)

        sample["video"] = video
        sample["masks"] = masks

        return sample


class MultiTaskLoss(nn.Module):
    """Combined loss for CVS classification and segmentation."""

    def __init__(
        self,
        cvs_weight: float = 1.0,
        seg_weight: float = 0.3,
        cvs_pos_weight: list = None,
        seg_class_weights: list = None,
    ):
        super().__init__()
        self.cvs_weight = cvs_weight
        self.seg_weight = seg_weight

        if cvs_pos_weight is not None:
            pos_weight = torch.tensor(cvs_pos_weight, dtype=torch.float32)
            self.cvs_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.cvs_loss = nn.BCEWithLogitsLoss()

        if seg_class_weights is not None:
            weight = torch.tensor(seg_class_weights, dtype=torch.float32)
            self.seg_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        else:
            self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, cvs_logits, cvs_labels, seg_logits=None, seg_labels=None):
        # Move weights to correct device
        if hasattr(self.cvs_loss, 'pos_weight') and self.cvs_loss.pos_weight is not None:
            self.cvs_loss.pos_weight = self.cvs_loss.pos_weight.to(cvs_logits.device)

        cvs_loss = self.cvs_loss(cvs_logits, cvs_labels)

        if seg_logits is not None and seg_labels is not None and seg_logits.shape[0] > 0:
            if hasattr(self.seg_loss, 'weight') and self.seg_loss.weight is not None:
                self.seg_loss.weight = self.seg_loss.weight.to(seg_logits.device)
            seg_loss = self.seg_loss(seg_logits, seg_labels)
            total_loss = self.cvs_weight * cvs_loss + self.seg_weight * seg_loss
        else:
            seg_loss = torch.tensor(0.0, device=cvs_logits.device)
            total_loss = self.cvs_weight * cvs_loss

        return {"total_loss": total_loss, "cvs_loss": cvs_loss, "seg_loss": seg_loss}


def compute_seg_metrics(seg_logits, seg_labels, num_classes=5):
    """Compute segmentation metrics."""
    if seg_logits.shape[0] == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}

    preds = seg_logits.argmax(dim=1)
    preds_flat = preds.view(-1)
    labels_flat = seg_labels.view(-1)

    valid_mask = labels_flat != 255
    preds_flat = preds_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    if len(labels_flat) == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}

    acc = (preds_flat == labels_flat).float().mean().item()

    ious = []
    for c in range(num_classes):
        pred_c = preds_flat == c
        label_c = labels_flat == c
        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())

    miou = np.mean(ious) if ious else 0.0
    return {"seg_miou": miou, "seg_acc": acc}


def create_model_for_stage(config: dict, stage: int, checkpoint_path: str = None):
    """Create model with appropriate freezing for each stage."""
    model_cfg = config["model"]
    stage_cfg = config[f"stage{stage}"]

    unfreeze_layers = stage_cfg["unfreeze_layers"]

    print(f"\n{'='*60}")
    print(f"Creating model for Stage {stage}")
    print(f"Unfreezing last {unfreeze_layers} layers")
    print(f"{'='*60}")

    model = VJEPA_MultiTask(
        model_name=model_cfg["name"],
        unfreeze_last_n_layers=unfreeze_layers,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        cvs_hidden=model_cfg.get("cvs_hidden", 512),
        cvs_dropout=model_cfg.get("cvs_dropout", 0.5),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        num_seg_classes=model_cfg.get("num_seg_classes", 5),
        seg_output_size=model_cfg.get("seg_output_size", 64),
        seg_dropout=model_cfg.get("seg_dropout", 0.1),
    )

    # Load checkpoint if provided (for Stage 2)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    return model


def create_optimizer_for_stage(model, config: dict, stage: int):
    """Create optimizer with appropriate LRs for each stage."""
    stage_cfg = config[f"stage{stage}"]
    training_cfg = config["training"]

    if stage == 1:
        # Stage 1: Only heads are trainable
        head_lr = stage_cfg["head_lr"]
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=head_lr,
            weight_decay=training_cfg["weight_decay"],
        )
        print(f"Stage 1 optimizer: head_lr={head_lr}")

    else:
        # Stage 2: Differential LR for backbone and heads
        backbone_lr = stage_cfg["backbone_lr"]
        head_lr = stage_cfg["head_lr"]

        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': head_lr})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=training_cfg["weight_decay"],
        )
        print(f"Stage 2 optimizer: backbone_lr={backbone_lr}, head_lr={head_lr}")

    return optimizer


def create_scheduler(optimizer, num_training_steps: int, num_warmup_steps: int, min_lr: float):
    """Create cosine scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, scaler, logger):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    cvs_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    accum_steps = config.get("gradient_accumulation", 1)
    use_amp = config.get("mixed_precision", True)

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)

        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(train_loader))

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)
                seg_logits = outputs.get("seg_logits", None)
                loss_dict = criterion(
                    outputs["cvs_logits"], cvs_labels,
                    seg_logits, masks if seg_logits is not None else None
                )
                loss = loss_dict["total_loss"] / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)
            seg_logits = outputs.get("seg_logits", None)
            loss_dict = criterion(
                outputs["cvs_logits"], cvs_labels,
                seg_logits, masks if seg_logits is not None else None
            )
            loss = loss_dict["total_loss"] / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            if not is_accumulating:
                nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        loss_meter.update(loss_dict["total_loss"].item(), cvs_labels.size(0))
        cvs_loss_meter.update(loss_dict["cvs_loss"].item(), cvs_labels.size(0))
        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_loss_meter.update(loss_dict["seg_loss"].item(), seg_logits.size(0))

        with torch.no_grad():
            probs = torch.sigmoid(outputs["cvs_logits"])
            all_preds.append(probs.cpu().numpy())
            all_targets.append(cvs_labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "cvs": f"{cvs_loss_meter.avg:.4f}"})

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    cvs_metrics = compute_metrics(all_preds, all_targets, 0.5)

    return {
        "loss": loss_meter.avg,
        "cvs_loss": cvs_loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "mAP": cvs_metrics["mAP"],
        "AP_C1": cvs_metrics.get("AP_C1", 0),
        "AP_C2": cvs_metrics.get("AP_C2", 0),
        "AP_C3": cvs_metrics.get("AP_C3", 0),
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []
    seg_miou_sum = 0.0
    seg_count = 0

    for batch in tqdm(val_loader, desc="Validating"):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)
        outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)

        seg_logits = outputs.get("seg_logits", None)
        loss_dict = criterion(
            outputs["cvs_logits"], cvs_labels,
            seg_logits, masks if seg_logits is not None else None
        )

        loss_meter.update(loss_dict["total_loss"].item(), cvs_labels.size(0))

        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_metrics = compute_seg_metrics(seg_logits, masks)
            seg_miou_sum += seg_metrics["seg_miou"] * seg_logits.size(0)
            seg_count += seg_logits.size(0)

        probs = torch.sigmoid(outputs["cvs_logits"])
        all_preds.append(probs.cpu().numpy())
        all_targets.append(cvs_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    cvs_metrics = compute_metrics(all_preds, all_targets, 0.5)

    return {
        "loss": loss_meter.avg,
        "mAP": cvs_metrics["mAP"],
        "AP_C1": cvs_metrics.get("AP_C1", 0),
        "AP_C2": cvs_metrics.get("AP_C2", 0),
        "AP_C3": cvs_metrics.get("AP_C3", 0),
        "seg_miou": seg_miou_sum / max(seg_count, 1),
    }


def train_stage(
    stage: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    results_dir: Path,
    logger,
    device: torch.device,
):
    """Train a single stage."""
    stage_cfg = config[f"stage{stage}"]
    training_cfg = config["training"]

    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE {stage}: {stage_cfg.get('description', '')}")
    logger.info(f"{'='*60}")

    # Create optimizer and scheduler
    optimizer = create_optimizer_for_stage(model, config, stage)

    accum_steps = stage_cfg.get("gradient_accumulation", 1)
    steps_per_epoch = len(train_loader) // accum_steps
    num_training_steps = steps_per_epoch * stage_cfg["epochs"]
    num_warmup_steps = steps_per_epoch * stage_cfg.get("warmup_epochs", 1)

    scheduler = create_scheduler(
        optimizer,
        num_training_steps,
        num_warmup_steps,
        training_cfg["min_lr"],
    )

    # Loss
    criterion = MultiTaskLoss(
        cvs_weight=config["loss"]["cvs_weight"],
        seg_weight=config["loss"]["seg_weight"],
        cvs_pos_weight=config["loss"].get("cvs_pos_weight"),
        seg_class_weights=config["loss"].get("seg_class_weights"),
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=stage_cfg["early_stopping_patience"],
        mode="max",
    )

    # Mixed precision
    scaler = GradScaler() if training_cfg.get("mixed_precision", True) else None

    # Training config for epoch
    epoch_config = {
        "gradient_accumulation": accum_steps,
        "mixed_precision": training_cfg.get("mixed_precision", True),
        "grad_clip": training_cfg.get("grad_clip", 1.0),
    }

    best_metric = 0.0
    best_epoch = 0

    for epoch in range(1, stage_cfg["epochs"] + 1):
        logger.info(f"\nStage {stage} - Epoch {epoch}/{stage_cfg['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch_config, scaler, logger
        )
        logger.info(f"Train | mAP: {train_metrics['mAP']*100:.2f}% | Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch_config)
        logger.info(
            f"Val   | mAP: {val_metrics['mAP']*100:.2f}% | "
            f"AP: C1={val_metrics['AP_C1']*100:.2f}%, C2={val_metrics['AP_C2']*100:.2f}%, C3={val_metrics['AP_C3']*100:.2f}%"
        )

        # Check best
        current_metric = val_metrics["mAP"]
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            logger.info(f"New best mAP: {best_metric*100:.2f}%")

        # Save checkpoint
        checkpoint = {
            "stage": stage,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": config,
        }

        save_checkpoint(
            checkpoint,
            str(results_dir / f"stage{stage}_epoch_{epoch}.pt"),
            is_best=is_best,
            best_path=str(results_dir / f"stage{stage}_best.pt"),
        )

        # Early stopping
        if early_stopping(current_metric):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"\nStage {stage} complete. Best mAP: {best_metric*100:.2f}% at epoch {best_epoch}")

    return results_dir / f"stage{stage}_best.pt", best_metric


def main(config_path: str = "configs/exp9_staged_finetune.yaml"):
    """Main training function."""
    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["data"]["results_dir"]) / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(results_dir), "train")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")
    logger.info(f"Results: {results_dir}")

    # Get augmentation config
    aug_config = config.get("augmentation", {})

    # Create datasets
    logger.info("\nCreating datasets...")

    train_dataset = AugmentedMultiTaskDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="train",
        num_frames=config["dataset"]["num_frames"],
        resolution=config["dataset"]["resolution"],
        mask_resolution=config["dataset"].get("mask_resolution", 64),
        augment=True,
        use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
        gt_masks_dir=config["data"].get("gt_masks_dir"),
        synthetic_masks_dir=config["data"].get("synthetic_masks_dir"),
        # Augmentation params
        horizontal_flip_prob=aug_config.get("horizontal_flip_prob", 0.5),
        rotation_degrees=aug_config.get("rotation_degrees", 15.0),
        color_jitter=aug_config.get("color_jitter"),
        random_erasing_prob=aug_config.get("random_erasing_prob", 0.2),
        gaussian_blur_prob=aug_config.get("gaussian_blur_prob", 0.1),
        gaussian_blur_sigma=tuple(aug_config.get("gaussian_blur_sigma", [0.1, 2.0])),
    )

    val_dataset = MultiTaskCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="val",
        num_frames=config["dataset"]["num_frames"],
        resolution=config["dataset"]["resolution"],
        mask_resolution=config["dataset"].get("mask_resolution", 64),
        augment=False,
        use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
        gt_masks_dir=config["data"].get("gt_masks_dir"),
        synthetic_masks_dir=config["data"].get("synthetic_masks_dir"),
    )

    logger.info(f"Train: {len(train_dataset)} clips")
    logger.info(f"Val: {len(val_dataset)} clips")

    # ==================== STAGE 1 ====================
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: Training heads with frozen backbone")
    logger.info("=" * 70)

    stage1_cfg = config["stage1"]

    # Create data loaders for stage 1
    train_loader_s1 = DataLoader(
        train_dataset,
        batch_size=stage1_cfg["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=stage1_cfg["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model for stage 1 (frozen backbone)
    model_s1 = create_model_for_stage(config, stage=1)
    model_s1 = model_s1.to(device)

    logger.info(f"Stage 1 - Total params: {model_s1.get_num_total_params() / 1e6:.1f}M")
    logger.info(f"Stage 1 - Trainable params: {model_s1.get_num_trainable_params() / 1e6:.1f}M")

    # Train stage 1
    stage1_best_path, stage1_best_metric = train_stage(
        stage=1,
        model=model_s1,
        train_loader=train_loader_s1,
        val_loader=val_loader,
        config=config,
        results_dir=results_dir,
        logger=logger,
        device=device,
    )

    # Clean up stage 1 model
    del model_s1
    torch.cuda.empty_cache()

    # ==================== STAGE 2 ====================
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: Minimal backbone fine-tuning")
    logger.info("=" * 70)

    stage2_cfg = config["stage2"]

    # Create data loaders for stage 2 (smaller batch)
    train_loader_s2 = DataLoader(
        train_dataset,
        batch_size=stage2_cfg["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader_s2 = DataLoader(
        val_dataset,
        batch_size=stage2_cfg["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model for stage 2 (unfreeze 1 layer) and load stage 1 weights
    model_s2 = create_model_for_stage(config, stage=2, checkpoint_path=str(stage1_best_path))
    model_s2 = model_s2.to(device)

    logger.info(f"Stage 2 - Total params: {model_s2.get_num_total_params() / 1e6:.1f}M")
    logger.info(f"Stage 2 - Trainable params: {model_s2.get_num_trainable_params() / 1e6:.1f}M")

    # Train stage 2
    stage2_best_path, stage2_best_metric = train_stage(
        stage=2,
        model=model_s2,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        config=config,
        results_dir=results_dir,
        logger=logger,
        device=device,
    )

    # Save final model
    final_checkpoint = torch.load(stage2_best_path)
    torch.save(final_checkpoint, results_dir / "best_model.pt")

    # ==================== SUMMARY ====================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Stage 1 best mAP: {stage1_best_metric*100:.2f}%")
    logger.info(f"Stage 2 best mAP: {stage2_best_metric*100:.2f}%")
    logger.info(f"Final model saved to: {results_dir / 'best_model.pt'}")

    improvement = stage2_best_metric - stage1_best_metric
    if improvement > 0:
        logger.info(f"Stage 2 improved over Stage 1 by {improvement*100:.2f}%")
    else:
        logger.info(f"Stage 2 did not improve over Stage 1 (diff: {improvement*100:.2f}%)")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staged fine-tuning for V-JEPA CVS model")
    parser.add_argument("--config", type=str, default="configs/exp9_staged_finetune.yaml")
    args = parser.parse_args()

    main(args.config)
