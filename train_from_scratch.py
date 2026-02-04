"""
Train-from-Scratch Training Script (Exp13)
===========================================
Small ViT with SwinV2-style window attention trained from scratch
on surgical video data for CVS classification + segmentation.

Based on train_regularized.py with these changes:
  - No PEFT/LoRA - all parameters trainable from scratch
  - Single optimizer with single LR (1e-3)
  - 50 epochs with 5 epoch warmup
  - Early stopping patience 10
  - Weight decay 0.05
  - MixUp/CutMix probs 0.3 each
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
from models import ViTSmallWindowCVS
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)


# ============================================================================
# Augmentation / regularization functions (from train_regularized.py)
# ============================================================================

def mixup_data(videos: torch.Tensor, labels: torch.Tensor, alpha: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """MixUp augmentation: blend video pairs and their labels using Beta distribution."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = videos.size(0)
    index = torch.randperm(batch_size, device=videos.device)

    mixed_videos = lam * videos + (1 - lam) * videos[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_videos, mixed_labels, lam


def cutmix_data(videos: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """CutMix augmentation: paste random rectangle from shuffled sample onto each video."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = videos.size(0)
    index = torch.randperm(batch_size, device=videos.device)

    _, T, C, H, W = videos.shape

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    mixed_videos = videos.clone()
    mixed_videos[:, :, :, y1:y2, x1:x2] = videos[index, :, :, y1:y2, x1:x2]

    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_videos, mixed_labels, lam


def apply_batch_augmentation(
    videos: torch.Tensor,
    labels: torch.Tensor,
    aug_config: dict,
) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Apply MixUp OR CutMix OR nothing (mutually exclusive)."""
    mixup_prob = aug_config.get("mixup_prob", 0.0)
    cutmix_prob = aug_config.get("cutmix_prob", 0.0)
    mixup_alpha = aug_config.get("mixup_alpha", 0.8)
    cutmix_alpha = aug_config.get("cutmix_alpha", 1.0)

    r = np.random.rand()

    if r < mixup_prob:
        videos, labels, lam = mixup_data(videos, labels, mixup_alpha)
        return videos, labels, lam, True
    elif r < mixup_prob + cutmix_prob:
        videos, labels, lam = cutmix_data(videos, labels, cutmix_alpha)
        return videos, labels, lam, True
    else:
        return videos, labels, 1.0, False


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Apply label smoothing to binary labels."""
    return labels * (1.0 - smoothing) + (1.0 - labels) * smoothing


def create_spatial_mask_from_segmentation(
    seg_masks: torch.Tensor,
    mask_batch_indices: torch.Tensor,
    batch_size: int,
    spatial_size: int = 16,
    num_temporal_bins: int = 8,
) -> torch.Tensor:
    """Create [B, num_tokens] float mask from segmentation annotations."""
    spatial_tokens = spatial_size * spatial_size
    num_tokens = num_temporal_bins * spatial_tokens
    device = seg_masks.device

    token_mask = torch.ones(batch_size, num_tokens, device=device)

    if len(mask_batch_indices) == 0:
        return token_mask

    unique_batch_ids = torch.unique(mask_batch_indices)

    for b_idx in unique_batch_ids:
        b = b_idx.item()
        mask_selector = (mask_batch_indices == b_idx)
        batch_masks = seg_masks[mask_selector]

        mask = batch_masks[0].float().unsqueeze(0).unsqueeze(0)
        downsampled = F.interpolate(
            mask, size=(spatial_size, spatial_size), mode="nearest"
        ).squeeze()

        spatial_mask = (downsampled > 0).float().view(-1)
        tiled_mask = spatial_mask.unsqueeze(0).expand(num_temporal_bins, -1).reshape(-1)
        token_mask[b] = tiled_mask

    return token_mask


def apply_hard_attention_mask(
    features: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Element-wise multiply features by mask to zero out background tokens."""
    return features * token_mask.unsqueeze(-1)


# ============================================================================
# Loss and metrics (from train_regularized.py)
# ============================================================================

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


def create_cosine_scheduler_with_warmup(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
):
    """Create cosine annealing scheduler with warmup."""
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Training and validation loops (from train_regularized.py)
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, scaler, logger):
    """Train for one epoch with MixUp/CutMix, label smoothing, and hard attention masking."""
    model.train()

    loss_meter = AverageMeter()
    cvs_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    accum_steps = config.get("gradient_accumulation", 1)
    use_amp = config.get("mixed_precision", True)

    aug_config = config.get("augmentation", {})
    ham_config = config.get("hard_attention_masking", {})
    label_smoothing = config.get("label_smoothing", 0.0)
    ham_enabled = ham_config.get("enabled", False) and ham_config.get("apply_during_training", False)
    ham_spatial_size = ham_config.get("spatial_size", 16)

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)

        original_cvs_labels = cvs_labels.clone()

        pixel_values, cvs_labels, lam, augmented = apply_batch_augmentation(
            pixel_values, cvs_labels, aug_config
        )

        if label_smoothing > 0:
            cvs_labels = smooth_labels(cvs_labels, label_smoothing)

        attention_mask = None
        if ham_enabled and len(mask_batch_indices) > 0:
            num_temporal_bins = 8
            attention_mask = create_spatial_mask_from_segmentation(
                masks, mask_batch_indices,
                batch_size=pixel_values.size(0),
                spatial_size=ham_spatial_size,
                num_temporal_bins=num_temporal_bins,
            )

        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(train_loader))

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(pixel_values, mask_frame_indices, mask_batch_indices, attention_mask)
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
            outputs = model(pixel_values, mask_frame_indices, mask_batch_indices, attention_mask)
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

        loss_meter.update(loss_dict["total_loss"].item(), original_cvs_labels.size(0))
        cvs_loss_meter.update(loss_dict["cvs_loss"].item(), original_cvs_labels.size(0))
        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_loss_meter.update(loss_dict["seg_loss"].item(), seg_logits.size(0))

        with torch.no_grad():
            probs = torch.sigmoid(outputs["cvs_logits"])
            all_preds.append(probs.cpu().numpy())
            all_targets.append(original_cvs_labels.cpu().numpy())

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
    """Validate the model. No MixUp/CutMix, no label smoothing, no hard attention masking."""
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


# Import augmented dataset
try:
    from train_staged import AugmentedMultiTaskDataset
except ImportError:
    AugmentedMultiTaskDataset = MultiTaskCVSDataset


def main(config_path: str = "configs/exp13_vit_small_window.yaml"):
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

    aug_config = config.get("augmentation", {})
    training_cfg = config["training"]

    # Log settings
    logger.info("\n" + "=" * 70)
    logger.info("Training Settings (Exp13 - Train from Scratch)")
    logger.info("=" * 70)
    logger.info(f"Learning rate: {training_cfg['learning_rate']}")
    logger.info(f"Weight decay: {training_cfg['weight_decay']}")
    logger.info(f"Epochs: {training_cfg['epochs']}")
    logger.info(f"Warmup epochs: {training_cfg['warmup_epochs']}")
    logger.info(f"Label smoothing: {training_cfg.get('label_smoothing', 0.0)}")
    logger.info(f"MixUp alpha: {aug_config.get('mixup_alpha', 0.0)}, prob: {aug_config.get('mixup_prob', 0.0)}")
    logger.info(f"CutMix alpha: {aug_config.get('cutmix_alpha', 0.0)}, prob: {aug_config.get('cutmix_prob', 0.0)}")

    ham_config = config.get("hard_attention_masking", {})
    logger.info(f"Hard attention masking: enabled={ham_config.get('enabled', False)}, "
                f"train={ham_config.get('apply_during_training', False)}, "
                f"eval={ham_config.get('apply_during_eval', False)}")

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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    logger.info("\n" + "=" * 70)
    logger.info("Creating ViTSmallWindowCVS model (from scratch)")
    logger.info("=" * 70)

    model_cfg = config["model"]
    model = ViTSmallWindowCVS(
        embed_dim=model_cfg.get("embed_dim", 384),
        depth=model_cfg.get("depth", 12),
        num_heads=model_cfg.get("num_heads", 6),
        window_size=model_cfg.get("window_size", 8),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        num_frames=model_cfg.get("num_frames", 16),
        spatial_size=model_cfg.get("spatial_size", 16),
        temporal_kernel=model_cfg.get("temporal_kernel", 2),
        spatial_kernel=model_cfg.get("spatial_kernel", 16),
        cvs_hidden=model_cfg.get("cvs_hidden", 512),
        cvs_dropout=model_cfg.get("cvs_dropout", 0.5),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        num_seg_classes=model_cfg.get("num_seg_classes", 5),
        seg_output_size=model_cfg.get("seg_output_size", 64),
        seg_dropout=model_cfg.get("seg_dropout", 0.1),
    )
    model = model.to(device)

    total_params = model.get_num_total_params()
    trainable_params = model.get_num_trainable_params()
    logger.info(f"Total params: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    # Single optimizer for all params
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    logger.info(f"Optimizer: AdamW, lr={training_cfg['learning_rate']}, wd={training_cfg['weight_decay']}")

    # Create scheduler
    accum_steps = training_cfg.get("gradient_accumulation", 1)
    steps_per_epoch = len(train_loader) // accum_steps
    num_training_steps = steps_per_epoch * training_cfg["epochs"]
    warmup_ratio = training_cfg.get("warmup_epochs", 5) / training_cfg["epochs"]

    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        num_training_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=0.01,
    )
    logger.info(f"Scheduler: cosine with {warmup_ratio*100:.1f}% warmup ({training_cfg.get('warmup_epochs', 5)} epochs)")

    # Loss
    criterion = MultiTaskLoss(
        cvs_weight=config["loss"]["cvs_weight"],
        seg_weight=config["loss"]["seg_weight"],
        cvs_pos_weight=config["loss"].get("cvs_pos_weight"),
        seg_class_weights=config["loss"].get("seg_class_weights"),
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_cfg["early_stopping_patience"],
        mode="max",
    )

    # Mixed precision
    scaler = GradScaler() if training_cfg.get("mixed_precision", True) else None

    # Training config for epoch
    epoch_config = {
        "gradient_accumulation": accum_steps,
        "mixed_precision": training_cfg.get("mixed_precision", True),
        "grad_clip": training_cfg.get("grad_clip", 1.0),
        "augmentation": aug_config,
        "hard_attention_masking": ham_config,
        "label_smoothing": training_cfg.get("label_smoothing", 0.0),
    }

    best_metric = 0.0
    best_epoch = 0

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Starting Training from Scratch (Exp13)")
    logger.info("=" * 70)

    for epoch in range(1, training_cfg["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{training_cfg['epochs']}")

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
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": config,
        }

        save_checkpoint(
            checkpoint,
            str(results_dir / f"epoch_{epoch}.pt"),
            is_best=is_best,
            best_path=str(results_dir / "best_model.pt"),
        )

        # Early stopping
        if early_stopping(current_metric):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best mAP: {best_metric*100:.2f}% at epoch {best_epoch}")
    logger.info(f"Final model saved to: {results_dir / 'best_model.pt'}")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT with window attention from scratch (Exp13)")
    parser.add_argument("--config", type=str, default="configs/exp13_vit_small_window.yaml")
    args = parser.parse_args()

    main(args.config)
