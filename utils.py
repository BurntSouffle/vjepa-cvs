"""
Utility functions for V-JEPA CVS classification.
"""

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(results_dir: str, name: str = "train") -> logging.Logger:
    """Setup logging to both console and file."""
    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("vjepa_cvs")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(results_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    debug: bool = False
) -> dict:
    """
    Compute evaluation metrics for multi-label classification.

    Args:
        predictions: (N, 3) array of predicted probabilities
        targets: (N, 3) array of ground truth labels
        threshold: threshold for binary predictions
        debug: whether to print debug info

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if debug:
        print(f"DEBUG: predictions shape={predictions.shape}, range=[{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"DEBUG: targets shape={targets.shape}, sum per class={targets.sum(axis=0)}")

    # Apply sigmoid if predictions are logits (values outside [0,1])
    if predictions.min() < 0 or predictions.max() > 1:
        if debug:
            print("DEBUG: Applying sigmoid to predictions")
        predictions = 1 / (1 + np.exp(-predictions))

    # Ensure correct dtypes
    predictions = predictions.astype(np.float64)
    targets = targets.astype(np.float64)

    # Overall mAP
    try:
        mAP = average_precision_score(targets, predictions, average="macro")
        if debug:
            print(f"DEBUG: Computed mAP = {mAP}")
        metrics["mAP"] = float(mAP) if not np.isnan(mAP) else 0.0
    except (ValueError, ZeroDivisionError) as e:
        if debug:
            print(f"DEBUG: mAP error: {e}")
        metrics["mAP"] = 0.0

    # Per-class AP
    class_names = ["C1", "C2", "C3"]
    for i, name in enumerate(class_names):
        try:
            # Check if there are both positive and negative samples for this class
            pos_count = targets[:, i].sum()
            if pos_count > 0 and pos_count < len(targets):
                ap = average_precision_score(targets[:, i], predictions[:, i])
                if debug:
                    print(f"DEBUG: AP_{name} = {ap}, pos_count={pos_count}")
                metrics[f"AP_{name}"] = float(ap) if not np.isnan(ap) else 0.0
            else:
                if debug:
                    print(f"DEBUG: AP_{name} skipped, pos_count={pos_count}")
                metrics[f"AP_{name}"] = 0.0
        except (ValueError, ZeroDivisionError) as e:
            if debug:
                print(f"DEBUG: AP_{name} error: {e}")
            metrics[f"AP_{name}"] = 0.0

    # Binary predictions
    binary_preds = (predictions >= threshold).astype(int)

    # Balanced accuracy per class
    for i, name in enumerate(class_names):
        try:
            ba = balanced_accuracy_score(targets[:, i], binary_preds[:, i])
            metrics[f"BA_{name}"] = ba
        except ValueError:
            metrics[f"BA_{name}"] = 0.0

    # Average balanced accuracy
    metrics["balanced_accuracy"] = np.mean([metrics[f"BA_{name}"] for name in class_names])

    # F1 scores
    for i, name in enumerate(class_names):
        try:
            f1 = f1_score(targets[:, i], binary_preds[:, i], zero_division=0)
            metrics[f"F1_{name}"] = f1
        except ValueError:
            metrics[f"F1_{name}"] = 0.0

    # Macro F1
    metrics["F1_macro"] = np.mean([metrics[f"F1_{name}"] for name in class_names])

    # Precision and Recall per class
    for i, name in enumerate(class_names):
        try:
            prec = precision_score(targets[:, i], binary_preds[:, i], zero_division=0)
            rec = recall_score(targets[:, i], binary_preds[:, i], zero_division=0)
            metrics[f"Precision_{name}"] = prec
            metrics[f"Recall_{name}"] = rec
        except ValueError:
            metrics[f"Precision_{name}"] = 0.0
            metrics[f"Recall_{name}"] = 0.0

    return metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dictionary as a readable string."""
    lines = []
    if prefix:
        lines.append(f"{prefix}:")

    # Main metrics
    lines.append(f"  mAP: {metrics.get('mAP', 0)*100:.2f}%")
    lines.append(f"  Per-class AP: C1={metrics.get('AP_C1', 0)*100:.2f}%, "
                f"C2={metrics.get('AP_C2', 0)*100:.2f}%, C3={metrics.get('AP_C3', 0)*100:.2f}%")
    lines.append(f"  Balanced Acc: {metrics.get('balanced_accuracy', 0)*100:.2f}%")
    lines.append(f"  F1 Macro: {metrics.get('F1_macro', 0)*100:.2f}%")

    return "\n".join(lines)


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "max" or "min" - whether higher or lower is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
    best_path: str = None
):
    """Save model checkpoint."""
    torch.save(state, filepath)
    if is_best and best_path:
        torch.save(state, best_path)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0)
