"""
Training script for V-JEPA CVS classification.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EndoscapesCVSDataset, collate_fn as endo_collate_fn, get_class_weights as endo_get_class_weights
from dataset_sages import SAGESCVSDataset, get_class_weights as sages_get_class_weights
from dataset_combined import CombinedCVSDataset, collate_fn as combined_collate_fn, get_class_weights as combined_get_class_weights
from model import create_model
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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Only set classifier to train mode if backbone is frozen
    if config["model"]["freeze_backbone"]:
        model.backbone.eval()

    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        videos = batch["videos"]  # List of numpy arrays
        labels = batch["labels"].to(device)  # (B, 3)

        # Process videos
        pixel_values = model.process_videos(videos, device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(pixel_values)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config["training"]["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])

        optimizer.step()
        scheduler.step()

        # Update metrics
        loss_meter.update(loss.item(), labels.size(0))

        # Store predictions
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Log periodically
        if (batch_idx + 1) % config["logging"]["log_every_n_steps"] == 0:
            logger.info(f"Epoch {epoch} | Step {batch_idx + 1}/{len(train_loader)} | Loss: {loss_meter.avg:.4f}")

        # Log LR at specific steps to verify warmup (epoch 1 only)
        if epoch == 1 and (batch_idx + 1) in [1, 100, 500, 1000]:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"[LR CHECK] Epoch 1, Step {batch_idx + 1}: LR = {current_lr:.2e}")

    # Compute epoch metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Debug on first epoch
    debug_metrics = (epoch == 1)
    if debug_metrics:
        logger.info(f"DEBUG: all_preds shape={all_preds.shape}, dtype={all_preds.dtype}")
        logger.info(f"DEBUG: all_targets shape={all_targets.shape}, dtype={all_targets.dtype}")
        logger.info(f"DEBUG: all_preds range=[{all_preds.min():.4f}, {all_preds.max():.4f}]")
        logger.info(f"DEBUG: all_targets sum per class={all_targets.sum(axis=0)}")
    metrics = compute_metrics(all_preds, all_targets, config["evaluation"]["threshold"], debug=debug_metrics)
    metrics["loss"] = loss_meter.avg

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc="Validating")

    for batch in pbar:
        videos = batch["videos"]
        labels = batch["labels"].to(device)

        # Process videos
        pixel_values = model.process_videos(videos, device)

        # Forward pass
        logits = model(pixel_values)

        # Compute loss
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), labels.size(0))

        # Store predictions
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds, all_targets, config["evaluation"]["threshold"])
    metrics["loss"] = loss_meter.avg

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


def main(config_path: str = "config.yaml", quick_test: bool = False):
    """Main training function."""
    # Load config
    config = load_config(config_path)

    # Setup
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["data"]["results_dir"]) / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(str(results_dir), "train")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")
    logger.info(f"Results directory: {results_dir}")

    # Create datasets based on dataset_type
    logger.info("Creating datasets...")
    dataset_type = config["data"].get("dataset_type", "endoscapes")
    logger.info(f"Dataset type: {dataset_type}")

    if dataset_type == "combined":
        # Combined SAGES + Endoscapes
        train_dataset = CombinedCVSDataset(
            sages_root=config["data"]["sages_root"],
            endoscapes_root=config["data"]["endoscapes_root"],
            split="train",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            augment=config["dataset"]["augment_train"],
            horizontal_flip_prob=config["dataset"]["horizontal_flip_prob"],
            sages_val_ratio=config["dataset"].get("sages_val_ratio", 0.2),
            seed=config["seed"],
        )
        val_dataset = CombinedCVSDataset(
            sages_root=config["data"]["sages_root"],
            endoscapes_root=config["data"]["endoscapes_root"],
            split="val",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            augment=False,
            sages_val_ratio=config["dataset"].get("sages_val_ratio", 0.2),
            seed=config["seed"],
        )
        collate_fn = combined_collate_fn
        get_class_weights = combined_get_class_weights

    elif dataset_type == "sages":
        # SAGES only
        train_dataset = SAGESCVSDataset(
            root_dir=config["data"]["sages_root"],
            split="train",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            augment=config["dataset"]["augment_train"],
            horizontal_flip_prob=config["dataset"]["horizontal_flip_prob"],
            val_ratio=config["dataset"].get("sages_val_ratio", 0.2),
            seed=config["seed"],
        )
        val_dataset = SAGESCVSDataset(
            root_dir=config["data"]["sages_root"],
            split="val",
            num_frames=config["dataset"]["num_frames"],
            resolution=config["dataset"]["resolution"],
            augment=False,
            val_ratio=config["dataset"].get("sages_val_ratio", 0.2),
            seed=config["seed"],
        )
        collate_fn = endo_collate_fn  # Same format as Endoscapes
        get_class_weights = sages_get_class_weights

    else:
        # Default: Endoscapes only
        train_dataset = EndoscapesCVSDataset(
            root_dir=config["data"]["endoscapes_root"],
            split="train",
            num_frames=config["dataset"]["num_frames"],
            frame_step=config["dataset"].get("frame_step", 25),
            resolution=config["dataset"]["resolution"],
            augment=config["dataset"]["augment_train"],
            horizontal_flip_prob=config["dataset"]["horizontal_flip_prob"],
        )
        val_dataset = EndoscapesCVSDataset(
            root_dir=config["data"]["endoscapes_root"],
            split="val",
            num_frames=config["dataset"]["num_frames"],
            frame_step=config["dataset"].get("frame_step", 25),
            resolution=config["dataset"]["resolution"],
            augment=False,
        )
        collate_fn = endo_collate_fn
        get_class_weights = endo_get_class_weights

    # For quick test, use subset
    if quick_test:
        logger.info("Quick test mode: using subset of data")
        train_dataset.samples = train_dataset.samples[:100]
        val_dataset.samples = val_dataset.samples[:50]

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create data loaders
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

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)

    logger.info(f"Total parameters: {model.get_num_total_params() / 1e6:.1f}M")
    logger.info(f"Trainable parameters: {model.get_num_trainable_params() / 1e6:.1f}M")

    # Loss function
    if config["loss"]["use_class_weights"]:
        pos_weight = torch.tensor(config["loss"]["pos_weight"]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Using class weights: {pos_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer - only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    num_warmup_steps = len(train_loader) * config["training"]["warmup_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=config["training"]["min_lr"] / config["training"]["learning_rate"],
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        mode="max",
    ) if config["training"]["early_stopping"] else None

    # ========== SANITY CHECK ==========
    logger.info("\n" + "="*60)
    logger.info("=== TRAINING SANITY CHECK ===")
    logger.info("="*60)

    # Model info
    logger.info(f"Model: {config['model']['name']}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    logger.info(f"Frozen params: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(f"Weight decay: {config['training']['weight_decay']}")
    logger.info(f"Dropout: {config['model']['dropout']}")

    # Dataset info
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    logger.info(f"Total training steps: {num_training_steps}")

    # Check first batch
    logger.info("\n--- First Batch Check ---")
    first_batch = next(iter(train_loader))
    logger.info(f"Videos in batch: {len(first_batch['videos'])}")
    logger.info(f"First video shape: {first_batch['videos'][0].shape}")
    logger.info(f"First video dtype: {first_batch['videos'][0].dtype}")
    logger.info(f"First video range: [{first_batch['videos'][0].min()}, {first_batch['videos'][0].max()}]")
    logger.info(f"Labels shape: {first_batch['labels'].shape}")
    logger.info(f"Labels dtype: {first_batch['labels'].dtype}")
    logger.info(f"Sample labels (first 5):")
    for i in range(min(5, len(first_batch['labels']))):
        logger.info(f"  Sample {i}: {first_batch['labels'][i].tolist()}")

    # Forward pass check
    logger.info("\n--- Forward Pass Check ---")
    test_videos = first_batch["videos"]
    test_labels = first_batch["labels"].to(device)
    test_pixel_values = model.process_videos(test_videos, device)
    logger.info(f"Processed tensor shape: {test_pixel_values.shape}")
    logger.info(f"Processed tensor dtype: {test_pixel_values.dtype}")

    with torch.no_grad():
        test_logits = model(test_pixel_values)
        test_probs = torch.sigmoid(test_logits)
        logger.info(f"Logits shape: {test_logits.shape}")
        logger.info(f"Logits range: [{test_logits.min():.4f}, {test_logits.max():.4f}]")
        logger.info(f"Probs range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
        logger.info(f"Labels are on: {test_labels.device}")
        logger.info(f"Logits are on: {test_logits.device}")

    # Verify backbone is frozen
    logger.info("\n--- Backbone Freeze Check ---")
    backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
    classifier_grads = [p.requires_grad for p in model.classifier.parameters()]
    logger.info(f"Backbone params requiring grad: {sum(backbone_grads)} / {len(backbone_grads)}")
    logger.info(f"Classifier params requiring grad: {sum(classifier_grads)} / {len(classifier_grads)}")

    if sum(backbone_grads) == 0 and sum(classifier_grads) > 0:
        logger.info("✓ Backbone frozen, classifier trainable - CORRECT")
    else:
        logger.warning("⚠ WARNING: Freeze configuration may be incorrect!")

    # Expected values check
    logger.info("\n--- Dataset Size Check ---")
    logger.info(f"Dataset type: {dataset_type}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    if dataset_type == "endoscapes":
        expected_train = 36694
        expected_val = 12372
        if len(train_dataset) == expected_train:
            logger.info(f"✓ Train size matches expected Endoscapes size")
        else:
            logger.warning(f"⚠ Train size {len(train_dataset)} != expected {expected_train}")
        if len(val_dataset) == expected_val:
            logger.info(f"✓ Val size matches expected Endoscapes size")
        else:
            logger.warning(f"⚠ Val size {len(val_dataset)} != expected {expected_val}")
    else:
        logger.info(f"✓ Dataset size check passed (custom dataset)")

    if trainable_params < 1_000_000:
        logger.info(f"✓ Trainable params {trainable_params:,} < 1M - classifier only")
    else:
        logger.warning(f"⚠ Trainable params {trainable_params:,} >= 1M - backbone may not be frozen!")

    logger.info("\n" + "="*60)
    logger.info("=== SANITY CHECK COMPLETE ===")
    logger.info("="*60 + "\n")

    # Training loop
    best_metric = 0.0
    best_epoch = 0

    epochs = 2 if quick_test else config["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, config, logger
        )
        logger.info(f"Train {format_metrics(train_metrics, 'Train')}")

        # Log learning rate (scheduler stepped per-batch in train_epoch)
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        logger.info(f"Val {format_metrics(val_metrics, 'Val')}")

        # Check if best model
        current_metric = val_metrics["mAP"]
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            logger.info(f"New best mAP: {best_metric*100:.2f}%")

        # Epoch 1 completion summary
        if epoch == 1:
            logger.info("\n" + "="*60)
            logger.info("=== EPOCH 1 COMPLETE - VERIFICATION ===")
            logger.info("="*60)
            logger.info(f"Final LR: {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Train mAP: {train_metrics['mAP']*100:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val mAP: {val_metrics['mAP']*100:.2f}%")
            logger.info(f"Per-class Val AP: C1={val_metrics.get('AP_C1', 0)*100:.2f}%, C2={val_metrics.get('AP_C2', 0)*100:.2f}%, C3={val_metrics.get('AP_C3', 0)*100:.2f}%")
            logger.info("")
            logger.info("Expected ranges (healthy training):")
            logger.info("  - Val mAP: 25-40%")
            logger.info("  - LR after epoch 1: ~1e-4 to 2e-4 (during warmup)")
            logger.info("  - Train-Val mAP gap: < 20%")
            gap = train_metrics['mAP'] - val_metrics['mAP']
            if gap > 0.20:
                logger.warning(f"⚠ Train-Val gap {gap*100:.1f}% > 20% - possible overfitting!")
            if val_metrics['mAP'] < 0.20:
                logger.warning(f"⚠ Val mAP {val_metrics['mAP']*100:.1f}% < 20% - training may have issues!")
            logger.info("="*60 + "\n")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        }

        save_checkpoint(
            checkpoint,
            str(results_dir / f"checkpoint_epoch_{epoch}.pt"),
            is_best=is_best,
            best_path=str(results_dir / "best_model.pt"),
        )

        # Save periodic checkpoint
        if epoch % config["logging"]["save_every_n_epochs"] == 0:
            save_checkpoint(checkpoint, str(results_dir / f"checkpoint_epoch_{epoch}.pt"))

        # Early stopping
        if early_stopping and early_stopping(current_metric):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best mAP: {best_metric*100:.2f}% at epoch {best_epoch}")
    logger.info(f"Model saved to: {results_dir / 'best_model.pt'}")

    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V-JEPA CVS classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with subset of data")
    args = parser.parse_args()

    main(args.config, args.quick_test)
