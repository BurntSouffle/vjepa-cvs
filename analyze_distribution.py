"""
Analyze class distribution in the CVS dataset.
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from dataset import EndoscapesCVSDataset
from utils import load_config


def analyze_distribution(dataset):
    """Compute and print class distribution statistics."""
    all_labels = np.array([s["labels"] for s in dataset.samples])

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.split} | Total samples: {len(dataset)}")
    print(f"{'='*60}")

    class_names = ["C1 (Hepatocystic Triangle)", "C2 (Cystic Plate)", "C3 (Two Structures)"]

    for i, name in enumerate(class_names):
        pos = (all_labels[:, i] > 0).sum()
        neg = (all_labels[:, i] == 0).sum()
        total = pos + neg
        pos_ratio = pos / total * 100
        neg_ratio = neg / total * 100
        imbalance_ratio = neg / (pos + 1e-6)

        print(f"\n{name}:")
        print(f"  Positive (=1): {pos:5d} ({pos_ratio:5.1f}%)")
        print(f"  Negative (=0): {neg:5d} ({neg_ratio:5.1f}%)")
        print(f"  Imbalance ratio (neg/pos): {imbalance_ratio:.2f}x")

    # Multi-label combinations
    print(f"\n{'='*60}")
    print("Multi-label combinations:")
    print(f"{'='*60}")

    # Count each combination
    combinations = {}
    for label in all_labels:
        key = tuple(label.astype(int))
        combinations[key] = combinations.get(key, 0) + 1

    # Sort by count
    sorted_combos = sorted(combinations.items(), key=lambda x: -x[1])

    for combo, count in sorted_combos:
        pct = count / len(all_labels) * 100
        c1, c2, c3 = combo
        print(f"  C1={c1}, C2={c2}, C3={c3}: {count:5d} ({pct:5.1f}%)")

    # All-positive vs all-negative
    all_positive = (all_labels.sum(axis=1) == 3).sum()
    all_negative = (all_labels.sum(axis=1) == 0).sum()
    any_positive = (all_labels.sum(axis=1) > 0).sum()

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"  All positive (C1=C2=C3=1): {all_positive:5d} ({all_positive/len(all_labels)*100:.1f}%)")
    print(f"  All negative (C1=C2=C3=0): {all_negative:5d} ({all_negative/len(all_labels)*100:.1f}%)")
    print(f"  Any positive (at least 1): {any_positive:5d} ({any_positive/len(all_labels)*100:.1f}%)")

    return all_labels


def compute_sample_weights(all_labels, strategy="class_balanced"):
    """
    Compute sample weights for WeightedRandomSampler.

    Strategies:
    - "class_balanced": Weight by inverse class frequency (avg across classes)
    - "any_positive": Oversample samples with any positive label
    - "rare_combo": Oversample rare label combinations
    """
    n_samples = len(all_labels)

    if strategy == "class_balanced":
        # For each sample, compute weight as average inverse frequency of its classes
        weights = np.zeros(n_samples)

        for i in range(3):  # For each class
            pos_count = (all_labels[:, i] > 0).sum()
            neg_count = (all_labels[:, i] == 0).sum()

            # Inverse frequency weights
            pos_weight = n_samples / (2 * pos_count + 1e-6)
            neg_weight = n_samples / (2 * neg_count + 1e-6)

            # Assign weights based on label
            for j in range(n_samples):
                if all_labels[j, i] > 0:
                    weights[j] += pos_weight
                else:
                    weights[j] += neg_weight

        # Average across classes
        weights /= 3

    elif strategy == "any_positive":
        # Simple: oversample any sample with at least one positive
        any_pos = all_labels.sum(axis=1) > 0
        pos_count = any_pos.sum()
        neg_count = n_samples - pos_count

        # Balance positive vs negative
        pos_weight = n_samples / (2 * pos_count + 1e-6)
        neg_weight = n_samples / (2 * neg_count + 1e-6)

        weights = np.where(any_pos, pos_weight, neg_weight)

    elif strategy == "rare_combo":
        # Weight by inverse frequency of label combination
        combo_counts = {}
        for label in all_labels:
            key = tuple(label.astype(int))
            combo_counts[key] = combo_counts.get(key, 0) + 1

        weights = np.zeros(n_samples)
        for j, label in enumerate(all_labels):
            key = tuple(label.astype(int))
            # Inverse frequency
            weights[j] = n_samples / combo_counts[key]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize so weights sum to n_samples
    weights = weights / weights.sum() * n_samples

    return weights


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp2_local_attention.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Analyze train set
    train_dataset = EndoscapesCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="train",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"].get("frame_step", 25),
        resolution=config["dataset"]["resolution"],
        augment=False,
    )

    all_labels = analyze_distribution(train_dataset)

    # Show sample weights for different strategies
    print(f"\n{'='*60}")
    print("Sample weight statistics (for WeightedRandomSampler):")
    print(f"{'='*60}")

    for strategy in ["class_balanced", "any_positive", "rare_combo"]:
        weights = compute_sample_weights(all_labels, strategy)
        print(f"\n{strategy}:")
        print(f"  Min weight:  {weights.min():.4f}")
        print(f"  Max weight:  {weights.max():.4f}")
        print(f"  Mean weight: {weights.mean():.4f}")
        print(f"  Std weight:  {weights.std():.4f}")

    # Also analyze validation set
    val_dataset = EndoscapesCVSDataset(
        root_dir=config["data"]["endoscapes_root"],
        split="val",
        num_frames=config["dataset"]["num_frames"],
        frame_step=config["dataset"].get("frame_step", 25),
        resolution=config["dataset"]["resolution"],
        augment=False,
    )

    analyze_distribution(val_dataset)
