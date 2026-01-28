"""
Evaluation script for V-JEPA CVS classification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EndoscapesCVSDataset, collate_fn
from model import create_model
from utils import compute_metrics, format_metrics, load_config, set_seed


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: dict,
) -> tuple:
    """
    Evaluate model on test set.

    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Array of predictions
        targets: Array of ground truth
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_metas = []

    pbar = tqdm(test_loader, desc="Evaluating")

    for batch in pbar:
        videos = batch["videos"]
        labels = batch["labels"].to(device)
        metas = batch["metas"]

        # Process videos
        pixel_values = model.process_videos(videos, device)

        # Forward pass
        logits = model(pixel_values)

        # Store predictions
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_metas.extend(metas)

    # Concatenate
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    metrics = compute_metrics(predictions, targets, config["evaluation"]["threshold"])

    return metrics, predictions, targets, all_metas


def print_comparison_table(metrics: dict):
    """Print comparison table with SwinCVS baseline."""
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE")
    print("=" * 70)

    # SwinCVS baseline (from the paper)
    swincvs_results = {
        "mAP": 67.45,
        "AP_C1": 65.02,
        "AP_C2": 61.38,
        "AP_C3": 75.95,
    }

    print(f"\n{'Metric':<20} {'V-JEPA (Ours)':<20} {'SwinCVS':<20} {'Delta':<15}")
    print("-" * 70)

    # Overall mAP
    our_map = metrics["mAP"] * 100
    delta = our_map - swincvs_results["mAP"]
    sign = "+" if delta > 0 else ""
    print(f"{'mAP':<20} {our_map:>18.2f}% {swincvs_results['mAP']:>18.2f}% {sign}{delta:>13.2f}%")

    # Per-class AP
    for criterion in ["C1", "C2", "C3"]:
        our_ap = metrics[f"AP_{criterion}"] * 100
        baseline_ap = swincvs_results[f"AP_{criterion}"]
        delta = our_ap - baseline_ap
        sign = "+" if delta > 0 else ""
        print(f"{'AP_' + criterion:<20} {our_ap:>18.2f}% {baseline_ap:>18.2f}% {sign}{delta:>13.2f}%")

    print("-" * 70)


def save_results(
    results_dir: Path,
    metrics: dict,
    predictions: np.ndarray,
    targets: np.ndarray,
    metas: list,
):
    """Save evaluation results to files."""
    # Save metrics
    metrics_file = results_dir / "test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Save predictions
    np.save(results_dir / "test_predictions.npy", predictions)
    np.save(results_dir / "test_targets.npy", targets)

    # Save detailed results
    detailed_results = []
    for i, meta in enumerate(metas):
        detailed_results.append({
            "video_id": meta["video_id"],
            "center_frame": meta["center_frame"],
            "pred_C1": float(predictions[i, 0]),
            "pred_C2": float(predictions[i, 1]),
            "pred_C3": float(predictions[i, 2]),
            "true_C1": float(targets[i, 0]),
            "true_C2": float(targets[i, 1]),
            "true_C3": float(targets[i, 2]),
        })

    with open(results_dir / "test_detailed_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


def main(checkpoint_path: str, config_path: str = None, output_dir: str = None):
    """Main evaluation function."""
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load config from checkpoint or file
    if config_path:
        config = load_config(config_path)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = load_config("config.yaml")

    # Setup
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create test dataset
    print("Loading test dataset...")
    test_dataset = EndoscapesCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="test",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"]["frame_step"],
        resolution=config["dataset"]["resolution"],
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("Creating model...")
    model = create_model(config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint best mAP: {checkpoint.get('best_metric', 0)*100:.2f}%")

    # Evaluate
    print("\nRunning evaluation...")
    metrics, predictions, targets, metas = evaluate(model, test_loader, device, config)

    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(format_metrics(metrics, "Test"))

    # Additional metrics
    print(f"\n  Per-class Balanced Accuracy:")
    print(f"    C1: {metrics['BA_C1']*100:.2f}%")
    print(f"    C2: {metrics['BA_C2']*100:.2f}%")
    print(f"    C3: {metrics['BA_C3']*100:.2f}%")

    print(f"\n  Per-class F1 Score:")
    print(f"    C1: {metrics['F1_C1']*100:.2f}%")
    print(f"    C2: {metrics['F1_C2']*100:.2f}%")
    print(f"    C3: {metrics['F1_C3']*100:.2f}%")

    # Comparison with baseline
    print_comparison_table(metrics)

    # Save results
    if output_dir:
        results_dir = Path(output_dir)
    else:
        results_dir = Path(checkpoint_path).parent

    save_results(results_dir, metrics, predictions, targets, metas)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA CVS classifier")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional, uses checkpoint config if not provided)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    main(args.checkpoint, args.config, args.output_dir)
