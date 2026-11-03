"""
compare_synthetic_vs_real.py -- Synthetic vs Real Data Comparison
================================================================
Loads Phase 2 (synthetic) and Phase 7 (real) results and produces
side-by-side comparison plots.

Usage:
    cd <project_root>
    python -m phase7_real_data.compare_synthetic_vs_real
"""

import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Mapping from dataset to modality
DATASET_TO_MODALITY = {
    "cifar10": "image",
    "agnews": "text",
    "speech": "audio",
    "ucf101": "video",
}

COLORS_SYNTH = {"image": "#81C784", "text": "#64B5F6", "audio": "#FFB74D", "video": "#F06292"}
COLORS_REAL = {"image": "#2E7D32", "text": "#1565C0", "audio": "#E65100", "video": "#AD1457"}


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def main():
    print(f"\n{'='*60}")
    print(f"  SYNTHETIC vs REAL DATA COMPARISON")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic vs Real Data -- FL Accuracy Comparison",
                fontsize=14, fontweight="bold")

    for idx, (dataset, modality) in enumerate(DATASET_TO_MODALITY.items()):
        ax = axes[idx // 2][idx % 2]

        # Load synthetic (Phase 2)
        synth_path = os.path.join(PROJECT_ROOT, "phase2_fl_sim", "results", f"{modality}_fl_log.csv")
        synth_data = load_csv(synth_path)

        # Load real (Phase 7)
        real_path = os.path.join(RESULTS_DIR, f"{dataset}_fl_log.csv")
        real_data = load_csv(real_path)

        has_data = False
        if synth_data:
            rounds = [int(r["round"]) for r in synth_data]
            accs = [float(r["global_accuracy"]) * 100 for r in synth_data]
            ax.plot(rounds, accs, "o--", color=COLORS_SYNTH[modality], linewidth=2,
                   markersize=5, label="Synthetic", alpha=0.7)
            has_data = True

        if real_data:
            rounds = [int(r["round"]) for r in real_data]
            accs = [float(r["global_accuracy"]) * 100 for r in real_data]
            ax.plot(rounds, accs, "s-", color=COLORS_REAL[modality], linewidth=2,
                   markersize=5, label="Real")
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                   transform=ax.transAxes, fontsize=12, color="gray")

        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{modality.upper()} ({dataset})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "synthetic_vs_real_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved: {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
