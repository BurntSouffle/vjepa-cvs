"""
Multi-task Training Script for V-JEPA CVS Classification + Segmentation.

Combines CVS classification loss with segmentation loss on frames with masks.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_multitask import MultiTaskCVSDataset, collate_fn
from model_multitask import create_multitask_model
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    format_metrics,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for CVS classification and segmentation.

    Loss = cvs_weight * BCE(cvs_pred, cvs_label) + seg_weight * CE(seg_pred, seg_label)
    """

    def __init__(
        self,
        cvs_weight: float = 1.0,
        seg_weight: float = 0.5,
        cvs_pos_weight: list = None,
        seg_class_weights: list = None,
        seg_ignore_index: int = 255,
    ):
        super().__init__()
        self.cvs_weight = cvs_weight
        self.seg_weight = seg_weight
        self.seg_ignore_index = seg_ignore_index

        # CVS loss (BCE with optional pos_weight)
        if cvs_pos_weight is not None:
            pos_weight = torch.tensor(cvs_pos_weight, dtype=torch.float32)
            self.cvs_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.cvs_loss = nn.BCEWithLogitsLoss()

        # Segmentation loss (CE with optional class weights)
        if seg_class_weights is not None:
            weight = torch.tensor(seg_class_weights, dtype=torch.float32)
            self.seg_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=seg_ignore_index)
        else:
            self.seg_loss = nn.CrossEntropyLoss(ignore_index=seg_ignore_index)

    def forward(
        self,
        cvs_logits: torch.Tensor,
        cvs_labels: torch.Tensor,
        seg_logits: torch.Tensor = None,
        seg_labels: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            cvs_logits: (B, 3) CVS classification logits
            cvs_labels: (B, 3) CVS labels
            seg_logits: (N, C, H, W) segmentation logits (optional)
            seg_labels: (N, H, W) segmentation labels (optional)

        Returns:
            dict with total_loss, cvs_loss, seg_loss
        """
        # Move pos_weight to correct device if needed
        if hasattr(self.cvs_loss, 'pos_weight') and self.cvs_loss.pos_weight is not None:
            self.cvs_loss.pos_weight = self.cvs_loss.pos_weight.to(cvs_logits.device)

        # CVS loss
        cvs_loss = self.cvs_loss(cvs_logits, cvs_labels)

        # Segmentation loss (only if we have seg data)
        if seg_logits is not None and seg_labels is not None and seg_logits.shape[0] > 0:
            # Move class weights to correct device if needed
            if hasattr(self.seg_loss, 'weight') and self.seg_loss.weight is not None:
                self.seg_loss.weight = self.seg_loss.weight.to(seg_logits.device)

            seg_loss = self.seg_loss(seg_logits, seg_labels)
            total_loss = self.cvs_weight * cvs_loss + self.seg_weight * seg_loss
        else:
            seg_loss = torch.tensor(0.0, device=cvs_logits.device)
            total_loss = self.cvs_weight * cvs_loss

        return {
            "total_loss": total_loss,
            "cvs_loss": cvs_loss,
            "seg_loss": seg_loss,
        }


def compute_seg_metrics(seg_logits: torch.Tensor, seg_labels: torch.Tensor, num_classes: int = 5) -> dict:
    """Compute segmentation metrics (mIoU, accuracy)."""
    if seg_logits.shape[0] == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}

    # Predictions
    preds = seg_logits.argmax(dim=1)  # (N, H, W)

    # Flatten
    preds_flat = preds.view(-1)
    labels_flat = seg_labels.view(-1)

    # Mask ignore regions
    valid_mask = labels_flat != 255
    preds_flat = preds_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    if len(labels_flat) == 0:
        return {"seg_miou": 0.0, "seg_acc": 0.0}

    # Accuracy
    acc = (preds_flat == labels_flat).float().mean().item()

    # Per-class IoU
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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    scaler: GradScaler = None,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Meters
    loss_meter = AverageMeter()
    cvs_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    all_cvs_preds = []
    all_cvs_targets = []
    seg_miou_sum = 0.0
    seg_count = 0

    use_mixed_precision = config["training"].get("mixed_precision", True)
    accum_steps = config["training"].get("gradient_accumulation_steps", 1)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        # Process videos
        pixel_values = model.process_videos(videos, device)

        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(train_loader))

        # Forward pass
        if use_mixed_precision and scaler is not None:
            with autocast():
                outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)

                # Compute loss
                seg_logits = outputs.get("seg_logits", None)
                loss_dict = criterion(
                    outputs["cvs_logits"],
                    cvs_labels,
                    seg_logits,
                    masks if seg_logits is not None else None,
                )
                loss = loss_dict["total_loss"] / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at step {batch_idx}, skipping")
                continue

            scaler.scale(loss).backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"NaN/Inf gradients at step {batch_idx}, skipping")
                    scaler.update()
                    optimizer.zero_grad()
                    continue

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)

            seg_logits = outputs.get("seg_logits", None)
            loss_dict = criterion(
                outputs["cvs_logits"],
                cvs_labels,
                seg_logits,
                masks if seg_logits is not None else None,
            )
            loss = loss_dict["total_loss"] / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at step {batch_idx}, skipping")
                continue

            loss.backward()

            if not is_accumulating:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"NaN/Inf gradients at step {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Update meters
        loss_meter.update(loss_dict["total_loss"].item(), cvs_labels.size(0))
        cvs_loss_meter.update(loss_dict["cvs_loss"].item(), cvs_labels.size(0))

        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_loss_meter.update(loss_dict["seg_loss"].item(), seg_logits.size(0))
            seg_metrics = compute_seg_metrics(seg_logits.detach(), masks.detach())
            seg_miou_sum += seg_metrics["seg_miou"] * seg_logits.size(0)
            seg_count += seg_logits.size(0)

        # Store CVS predictions
        with torch.no_grad():
            probs = torch.sigmoid(outputs["cvs_logits"])
            all_cvs_preds.append(probs.cpu().numpy())
            all_cvs_targets.append(cvs_labels.cpu().numpy())

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "cvs": f"{cvs_loss_meter.avg:.4f}",
            "seg": f"{seg_loss_meter.avg:.4f}",
        })

        if (batch_idx + 1) % config["logging"]["log_every_n_steps"] == 0:
            logger.info(
                f"Epoch {epoch} | Step {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss_meter.avg:.4f} | CVS: {cvs_loss_meter.avg:.4f} | Seg: {seg_loss_meter.avg:.4f}"
            )

    # Compute CVS metrics
    all_cvs_preds = np.concatenate(all_cvs_preds, axis=0)
    all_cvs_targets = np.concatenate(all_cvs_targets, axis=0)
    cvs_metrics = compute_metrics(all_cvs_preds, all_cvs_targets, config["evaluation"]["threshold"])

    metrics = {
        "loss": loss_meter.avg,
        "cvs_loss": cvs_loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "mAP": cvs_metrics["mAP"],
        "AP_C1": cvs_metrics.get("AP_C1", 0),
        "AP_C2": cvs_metrics.get("AP_C2", 0),
        "AP_C3": cvs_metrics.get("AP_C3", 0),
        "seg_miou": seg_miou_sum / max(seg_count, 1),
    }

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    config: dict,
) -> dict:
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    cvs_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    all_cvs_preds = []
    all_cvs_targets = []
    seg_miou_sum = 0.0
    seg_count = 0

    pbar = tqdm(val_loader, desc="Validating")

    for batch in pbar:
        videos = batch["videos"]
        cvs_labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        mask_frame_indices = batch["mask_frame_indices"].to(device)
        mask_batch_indices = batch["mask_batch_indices"].to(device)

        pixel_values = model.process_videos(videos, device)

        outputs = model(pixel_values, mask_frame_indices, mask_batch_indices)

        seg_logits = outputs.get("seg_logits", None)
        loss_dict = criterion(
            outputs["cvs_logits"],
            cvs_labels,
            seg_logits,
            masks if seg_logits is not None else None,
        )

        loss_meter.update(loss_dict["total_loss"].item(), cvs_labels.size(0))
        cvs_loss_meter.update(loss_dict["cvs_loss"].item(), cvs_labels.size(0))

        if seg_logits is not None and seg_logits.shape[0] > 0:
            seg_loss_meter.update(loss_dict["seg_loss"].item(), seg_logits.size(0))
            seg_metrics = compute_seg_metrics(seg_logits, masks)
            seg_miou_sum += seg_metrics["seg_miou"] * seg_logits.size(0)
            seg_count += seg_logits.size(0)

        probs = torch.sigmoid(outputs["cvs_logits"])
        all_cvs_preds.append(probs.cpu().numpy())
        all_cvs_targets.append(cvs_labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    all_cvs_preds = np.concatenate(all_cvs_preds, axis=0)
    all_cvs_targets = np.concatenate(all_cvs_targets, axis=0)
    cvs_metrics = compute_metrics(all_cvs_preds, all_cvs_targets, config["evaluation"]["threshold"])

    metrics = {
        "loss": loss_meter.avg,
        "cvs_loss": cvs_loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "mAP": cvs_metrics["mAP"],
        "AP_C1": cvs_metrics.get("AP_C1", 0),
        "AP_C2": cvs_metrics.get("AP_C2", 0),
        "AP_C3": cvs_metrics.get("AP_C3", 0),
        "seg_miou": seg_miou_sum / max(seg_count, 1),
    }

    return metrics


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
):
    """Create cosine learning rate schedule with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main(config_path: str = "configs/exp8_finetune_multitask.yaml"):
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

    # Datasets
    logger.info("Creating datasets...")
    train_dataset = MultiTaskCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="train",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"].get("frame_step", 25),
        resolution=config["dataset"]["resolution"],
        mask_resolution=config["dataset"].get("mask_resolution", 64),
        augment=config["dataset"]["augment_train"],
        horizontal_flip_prob=config["dataset"]["horizontal_flip_prob"],
        use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
    )

    val_dataset = MultiTaskCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="val",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"].get("frame_step", 25),
        resolution=config["dataset"]["resolution"],
        mask_resolution=config["dataset"].get("mask_resolution", 64),
        augment=False,
        use_synthetic_masks=config["dataset"].get("use_synthetic_masks", True),
    )

    logger.info(f"Train: {len(train_dataset)} clips, {train_dataset.clips_with_masks} with masks")
    logger.info(f"Val: {len(val_dataset)} clips, {val_dataset.clips_with_masks} with masks")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    logger.info("Creating model...")
    model = create_multitask_model(config)
    model = model.to(device)

    logger.info(f"Total params: {model.get_num_total_params() / 1e6:.1f}M")
    logger.info(f"Trainable params: {model.get_num_trainable_params() / 1e6:.1f}M")

    # Loss function
    criterion = MultiTaskLoss(
        cvs_weight=config["loss"].get("cvs_weight", 1.0),
        seg_weight=config["loss"].get("seg_weight", 0.5),
        cvs_pos_weight=config["loss"].get("cvs_pos_weight", None),
        seg_class_weights=config["loss"].get("seg_class_weights", None),
    )

    logger.info(f"Loss weights: CVS={criterion.cvs_weight}, Seg={criterion.seg_weight}")

    # Optimizer with differential LR
    backbone_lr = config["training"].get("backbone_lr", 1e-5)
    head_lr = config["training"].get("head_lr", 5e-4)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    logger.info(f"Backbone LR: {backbone_lr}, Head LR: {head_lr}")
    logger.info(f"Backbone params: {sum(p.numel() for p in backbone_params) / 1e6:.1f}M")
    logger.info(f"Head params: {sum(p.numel() for p in head_params) / 1e6:.1f}M")

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    steps_per_epoch = len(train_loader) // accum_steps
    num_training_steps = steps_per_epoch * config["training"]["epochs"]
    num_warmup_steps = steps_per_epoch * config["training"]["warmup_epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=config["training"]["min_lr"] / head_lr,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        mode="max",
    ) if config["training"]["early_stopping"] else None

    # Mixed precision
    scaler = GradScaler() if config["training"].get("mixed_precision", True) else None

    # Training loop
    best_metric = 0.0
    best_epoch = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, config, logger, scaler
        )
        logger.info(
            f"Train | mAP: {train_metrics['mAP']*100:.2f}% | "
            f"CVS Loss: {train_metrics['cvs_loss']:.4f} | "
            f"Seg Loss: {train_metrics['seg_loss']:.4f} | "
            f"Seg mIoU: {train_metrics['seg_miou']*100:.2f}%"
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        logger.info(
            f"Val   | mAP: {val_metrics['mAP']*100:.2f}% | "
            f"CVS Loss: {val_metrics['cvs_loss']:.4f} | "
            f"Seg Loss: {val_metrics['seg_loss']:.4f} | "
            f"Seg mIoU: {val_metrics['seg_miou']*100:.2f}%"
        )
        logger.info(
            f"Val   | AP: C1={val_metrics['AP_C1']*100:.2f}%, "
            f"C2={val_metrics['AP_C2']*100:.2f}%, "
            f"C3={val_metrics['AP_C3']*100:.2f}%"
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
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": config,
        }

        save_checkpoint(
            checkpoint,
            str(results_dir / f"checkpoint_epoch_{epoch}.pt"),
            is_best=is_best,
            best_path=str(results_dir / "best_model.pt"),
        )

        # Early stopping
        if early_stopping and early_stopping(current_metric):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best mAP: {best_metric*100:.2f}% at epoch {best_epoch}")
    logger.info(f"Model saved to: {results_dir / 'best_model.pt'}")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task V-JEPA CVS model")
    parser.add_argument("--config", type=str, default="configs/exp8_finetune_multitask.yaml")
    args = parser.parse_args()

    main(args.config)
