"""
Compare results from multiple V-JEPA CVS experiments.

Loads results from experiment directories and generates:
- Comparison table
- Per-class AP analysis
- Recommendations for best configuration
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Experiment configurations for reference
EXPERIMENTS = {
    "exp1_sages_baseline": {
        "name": "SAGES + Mean + MLP",
        "description": "Baseline with more data",
        "dataset": "SAGES + Endoscapes",
        "pooling": "mean",
        "head": "mlp",
    },
    "exp2_sages_attention": {
        "name": "SAGES + Attention + MLP",
        "description": "Attention pooling test",
        "dataset": "SAGES + Endoscapes",
        "pooling": "attention",
        "head": "mlp",
    },
    "exp3_sages_simple": {
        "name": "SAGES + Mean + Simple",
        "description": "Simple head test",
        "dataset": "SAGES + Endoscapes",
        "pooling": "mean",
        "head": "simple",
    },
    "exp4_endo_attention_simple": {
        "name": "Endo + Attention + Simple",
        "description": "Maximum regularisation",
        "dataset": "Endoscapes only",
        "pooling": "attention",
        "head": "simple",
    },
    "baseline_endo_mean_mlp": {
        "name": "Endo + Mean + MLP (Local)",
        "description": "Current baseline",
        "dataset": "Endoscapes only",
        "pooling": "mean",
        "head": "mlp",
    },
}

# Reference: SwinCVS baseline
SWIN_CVS_BASELINE = {
    "mAP": 0.6745,
    "AP_C1": 0.6000,
    "AP_C2": 0.5500,
    "AP_C3": 0.8700,
}


def load_experiment_results(results_dir: Path) -> Optional[Dict]:
    """Load metrics from an experiment's results directory."""
    # Look for metrics files
    metrics_file = results_dir / "test_metrics.json"
    val_metrics_file = results_dir / "val_metrics.json"

    metrics = None

    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    elif val_metrics_file.exists():
        with open(val_metrics_file) as f:
            metrics = json.load(f)

    # Also check for best_model.pt and extract epoch info
    best_model = results_dir / "best_model.pt"
    if best_model.exists():
        if metrics is None:
            metrics = {}
        metrics["has_checkpoint"] = True

    # Try to extract metrics from log file if no metrics file
    if metrics is None:
        log_files = list(results_dir.glob("*.log"))
        if log_files:
            metrics = parse_log_file(log_files[0])

    return metrics


def parse_log_file(log_path: Path) -> Optional[Dict]:
    """Parse metrics from training log file."""
    metrics = {}

    try:
        with open(log_path, "r") as f:
            content = f.read()

        # Look for best validation mAP
        import re

        # Pattern: "New best mAP: XX.XX%"
        best_map_match = re.search(r"New best mAP: (\d+\.?\d*)%", content)
        if best_map_match:
            metrics["mAP"] = float(best_map_match.group(1)) / 100

        # Pattern: "Val mAP: XX.XX%"
        val_map_matches = re.findall(r"Val mAP: (\d+\.?\d*)%", content)
        if val_map_matches:
            metrics["best_val_mAP"] = max(float(m) for m in val_map_matches) / 100

        # Per-class AP
        ap_match = re.search(
            r"C1=(\d+\.?\d*)%, C2=(\d+\.?\d*)%, C3=(\d+\.?\d*)%",
            content
        )
        if ap_match:
            metrics["AP_C1"] = float(ap_match.group(1)) / 100
            metrics["AP_C2"] = float(ap_match.group(2)) / 100
            metrics["AP_C3"] = float(ap_match.group(3)) / 100

        # Look for final epoch
        epoch_matches = re.findall(r"Epoch (\d+)/", content)
        if epoch_matches:
            metrics["final_epoch"] = max(int(e) for e in epoch_matches)

    except Exception as e:
        print(f"Warning: Could not parse log file {log_path}: {e}")

    return metrics if metrics else None


def find_experiment_results(base_dir: Path) -> Dict[str, Dict]:
    """Find all experiment results in the base directory."""
    results = {}

    # Check for results in various locations
    results_paths = [
        base_dir / "results",
        base_dir,
    ]

    for results_path in results_paths:
        if not results_path.exists():
            continue

        for exp_dir in results_path.iterdir():
            if not exp_dir.is_dir():
                continue

            # Match experiment directories
            dir_name = exp_dir.name

            # Check for run_ prefix (timestamped runs)
            if dir_name.startswith("run_"):
                # This might be a baseline run
                metrics = load_experiment_results(exp_dir)
                if metrics:
                    results[f"run_{dir_name}"] = {
                        "path": str(exp_dir),
                        "metrics": metrics,
                    }

            # Check for experiment prefixes
            for exp_key in EXPERIMENTS:
                if dir_name.startswith(exp_key) or exp_key in dir_name:
                    metrics = load_experiment_results(exp_dir)
                    if metrics:
                        results[exp_key] = {
                            "path": str(exp_dir),
                            "metrics": metrics,
                        }
                    break

    return results


def format_table(results: Dict[str, Dict]) -> str:
    """Format results as a comparison table."""
    lines = []

    # Header
    lines.append("=" * 100)
    lines.append("V-JEPA CVS EXPERIMENT COMPARISON")
    lines.append("=" * 100)
    lines.append("")

    # Column headers
    header = f"{'Experiment':<35} {'mAP':>8} {'C1':>8} {'C2':>8} {'C3':>8} {'Epoch':>6}"
    lines.append(header)
    lines.append("-" * 80)

    # Add SwinCVS baseline for reference
    lines.append(
        f"{'[REF] SwinCVS Baseline':<35} "
        f"{SWIN_CVS_BASELINE['mAP']*100:>7.2f}% "
        f"{SWIN_CVS_BASELINE['AP_C1']*100:>7.2f}% "
        f"{SWIN_CVS_BASELINE['AP_C2']*100:>7.2f}% "
        f"{SWIN_CVS_BASELINE['AP_C3']*100:>7.2f}% "
        f"{'N/A':>6}"
    )
    lines.append("-" * 80)

    # Sort by mAP (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("metrics", {}).get("mAP", 0),
        reverse=True,
    )

    for exp_key, exp_data in sorted_results:
        metrics = exp_data.get("metrics", {})
        exp_info = EXPERIMENTS.get(exp_key, {"name": exp_key})
        name = exp_info.get("name", exp_key)[:35]

        mAP = metrics.get("mAP", metrics.get("best_val_mAP", 0))
        c1 = metrics.get("AP_C1", 0)
        c2 = metrics.get("AP_C2", 0)
        c3 = metrics.get("AP_C3", 0)
        epoch = metrics.get("final_epoch", "?")

        lines.append(
            f"{name:<35} "
            f"{mAP*100:>7.2f}% "
            f"{c1*100:>7.2f}% "
            f"{c2*100:>7.2f}% "
            f"{c3*100:>7.2f}% "
            f"{epoch:>6}"
        )

    lines.append("=" * 100)

    return "\n".join(lines)


def analyze_results(results: Dict[str, Dict]) -> Dict:
    """Analyze results and generate recommendations."""
    analysis = {
        "best_overall": None,
        "best_per_class": {"C1": None, "C2": None, "C3": None},
        "recommendations": [],
        "insights": [],
    }

    if not results:
        analysis["recommendations"].append("No experiment results found.")
        return analysis

    # Find best overall mAP
    best_map = 0
    for exp_key, exp_data in results.items():
        metrics = exp_data.get("metrics", {})
        mAP = metrics.get("mAP", metrics.get("best_val_mAP", 0))
        if mAP > best_map:
            best_map = mAP
            analysis["best_overall"] = {
                "experiment": exp_key,
                "mAP": mAP,
                "config": EXPERIMENTS.get(exp_key, {}),
            }

    # Find best per class
    for class_name in ["C1", "C2", "C3"]:
        best_ap = 0
        for exp_key, exp_data in results.items():
            metrics = exp_data.get("metrics", {})
            ap = metrics.get(f"AP_{class_name}", 0)
            if ap > best_ap:
                best_ap = ap
                analysis["best_per_class"][class_name] = {
                    "experiment": exp_key,
                    "AP": ap,
                }

    # Generate recommendations
    if analysis["best_overall"]:
        best = analysis["best_overall"]
        rec = f"Best configuration: {best['experiment']} with {best['mAP']*100:.2f}% mAP"
        analysis["recommendations"].append(rec)

        # Compare to baseline
        if best["mAP"] > SWIN_CVS_BASELINE["mAP"]:
            improvement = (best["mAP"] - SWIN_CVS_BASELINE["mAP"]) * 100
            analysis["recommendations"].append(
                f"Exceeds SwinCVS baseline by +{improvement:.2f}%!"
            )
        else:
            gap = (SWIN_CVS_BASELINE["mAP"] - best["mAP"]) * 100
            analysis["recommendations"].append(
                f"Still {gap:.2f}% below SwinCVS baseline."
            )

    # Analyze component effects
    attention_results = [r for k, r in results.items()
                         if EXPERIMENTS.get(k, {}).get("pooling") == "attention"]
    mean_results = [r for k, r in results.items()
                    if EXPERIMENTS.get(k, {}).get("pooling") == "mean"]

    if attention_results and mean_results:
        attn_avg = np.mean([r["metrics"].get("mAP", 0) for r in attention_results])
        mean_avg = np.mean([r["metrics"].get("mAP", 0) for r in mean_results])
        if attn_avg > mean_avg:
            analysis["insights"].append(
                f"Attention pooling shows +{(attn_avg-mean_avg)*100:.2f}% improvement on average"
            )
        else:
            analysis["insights"].append(
                f"Mean pooling performs {(mean_avg-attn_avg)*100:.2f}% better on average"
            )

    return analysis


def format_analysis(analysis: Dict) -> str:
    """Format analysis as readable text."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("ANALYSIS & RECOMMENDATIONS")
    lines.append("=" * 60)

    if analysis["recommendations"]:
        lines.append("\nRecommendations:")
        for rec in analysis["recommendations"]:
            lines.append(f"  - {rec}")

    if analysis["insights"]:
        lines.append("\nInsights:")
        for insight in analysis["insights"]:
            lines.append(f"  - {insight}")

    if analysis["best_per_class"]:
        lines.append("\nBest per class:")
        for class_name, info in analysis["best_per_class"].items():
            if info:
                lines.append(
                    f"  {class_name}: {info['experiment']} ({info['AP']*100:.2f}%)"
                )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_comparison(results: Dict, analysis: Dict, output_path: Path):
    """Save comparison results to JSON."""
    output = {
        "experiments": {},
        "analysis": analysis,
        "swin_cvs_baseline": SWIN_CVS_BASELINE,
    }

    for exp_key, exp_data in results.items():
        output["experiments"][exp_key] = {
            "path": exp_data.get("path"),
            "metrics": exp_data.get("metrics"),
            "config": EXPERIMENTS.get(exp_key, {}),
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare V-JEPA CVS experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Base directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_comparison.json",
        help="Output JSON file for comparison",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    print(f"Searching for results in: {base_dir.absolute()}")

    # Find all experiment results
    results = find_experiment_results(base_dir)
    print(f"Found {len(results)} experiment(s)")

    if not results:
        print("\nNo experiment results found.")
        print("Expected directories:")
        for exp_key in EXPERIMENTS:
            print(f"  - results/{exp_key}/")
        return

    # Generate comparison table
    table = format_table(results)
    print(table)

    # Analyze results
    analysis = analyze_results(results)
    analysis_text = format_analysis(analysis)
    print(analysis_text)

    # Save to JSON
    output_path = base_dir / args.output
    save_comparison(results, analysis, output_path)


if __name__ == "__main__":
    main()
