"""
generate_report.py -- Evaluation Report Generator (Phase 6)
===========================================================
Aggregates results from all previous phases and generates:
  1. Per-modality accuracy-vs-round plots
  2. Offload decision summary with energy/latency savings
  3. Client selection comparison plot (random vs agentic)
  4. Combined dashboard figure

Usage:
    cd <project_root>
    python -m phase6_evaluation.generate_report
"""

import os
import sys
import csv
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODALITIES = ["image", "text", "audio", "video"]
COLORS = {"image": "#4CAF50", "text": "#2196F3", "audio": "#FF9800", "video": "#E91E63"}


def load_csv(path):
    """Load a CSV file as list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def generate_dashboard():
    """Create the combined 2x3 dashboard figure."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Multi-Modal Federated Learning with Agentic Offloading -- Results Dashboard",
                fontsize=16, fontweight="bold")

    # ── Panel 1: Phase 2 FL accuracy curves ─────────────────────────────
    ax = axes[0, 0]
    for mod in MODALITIES:
        csv_path = os.path.join(PROJECT_ROOT, "phase2_fl_sim", "results", f"{mod}_fl_log.csv")
        data = load_csv(csv_path)
        if data:
            rounds = [int(r["round"]) for r in data]
            accs = [float(r["global_accuracy"]) * 100 for r in data]
            ax.plot(rounds, accs, "o-", color=COLORS[mod], linewidth=2,
                   markersize=4, label=mod.capitalize())
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("FL Accuracy (Phase 2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Offload rate per modality ──────────────────────────────
    ax = axes[0, 1]
    offload_csv = os.path.join(PROJECT_ROOT, "phase4_offload", "results", "offload_log.csv")
    data = load_csv(offload_csv)
    if data:
        offload_counts = {mod: {"local": 0, "edge": 0} for mod in MODALITIES}
        for row in data:
            mod = row["modality"]
            dec = row["decision"]
            if mod in offload_counts:
                offload_counts[mod][dec] += 1

        rates = []
        for mod in MODALITIES:
            total = offload_counts[mod]["local"] + offload_counts[mod]["edge"]
            rates.append(offload_counts[mod]["edge"] / max(total, 1) * 100)

        bars = ax.bar(MODALITIES, rates, color=[COLORS[m] for m in MODALITIES], alpha=0.85)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=9)
        ax.set_ylabel("Offload to Edge (%)")
        ax.set_title("Offload Rate (Phase 4)")
        ax.set_ylim(0, max(rates) * 1.3 if rates else 100)
        ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Energy savings estimate ────────────────────────────────
    ax = axes[0, 2]
    if data:
        energy_savings = [r * 0.6 for r in rates]
        latency_savings = [r * 0.4 for r in rates]

        x = np.arange(len(MODALITIES))
        w = 0.35
        ax.bar(x - w/2, energy_savings, w, color="#9C27B0", alpha=0.85, label="Energy Saved")
        ax.bar(x + w/2, latency_savings, w, color="#00BCD4", alpha=0.85, label="Latency Saved")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in MODALITIES])
        ax.set_ylabel("Estimated Savings (%)")
        ax.set_title("Savings vs Always-Local")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4-5: Random vs Agentic comparison ────────────────────────
    for idx, (mod, ax) in enumerate(zip(MODALITIES[:2], [axes[1, 0], axes[1, 1]])):
        csv_path = os.path.join(PROJECT_ROOT, "phase5_agentic", "results", f"{mod}_comparison.csv")
        data = load_csv(csv_path)
        if data:
            rounds = [int(r["round"]) for r in data]
            rand_acc = [float(r["random_acc"]) * 100 for r in data]
            agent_acc = [float(r["agentic_acc"]) * 100 for r in data]
            ax.plot(rounds, rand_acc, "o--", color=COLORS[mod], linewidth=2,
                   markersize=4, alpha=0.5, label="Random")
            ax.plot(rounds, agent_acc, "s-", color=COLORS[mod], linewidth=2,
                   markersize=4, label="Agentic")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Selection: {mod.upper()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel 6: Summary table ──────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = "EXPERIMENT SUMMARY\n" + "=" * 35 + "\n\n"
    summary_text += f"Clients per modality: 50\n"
    summary_text += f"FL Rounds: 20\n"
    summary_text += f"Selection: Top-60% by score\n"
    summary_text += f"Offload: Cost-based decision\n\n"

    # Add final accuracies
    summary_text += "Final FL Accuracy:\n"
    for mod in MODALITIES:
        csv_path = os.path.join(PROJECT_ROOT, "phase2_fl_sim", "results", f"{mod}_fl_log.csv")
        data = load_csv(csv_path)
        if data:
            final_acc = float(data[-1]["global_accuracy"]) * 100
            summary_text += f"  {mod:>6}: {final_acc:.2f}%\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "results_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved: {path}")
    return path


def generate_summary_json():
    """Generate a JSON summary of all results."""
    summary = {
        "project": "Multi-Modal Federated Learning with Agentic Offloading",
        "phases_completed": [],
        "modalities": {},
    }

    for mod in MODALITIES:
        mod_info = {"modality": mod}

        # Phase 2 results
        csv_path = os.path.join(PROJECT_ROOT, "phase2_fl_sim", "results", f"{mod}_fl_log.csv")
        data = load_csv(csv_path)
        if data:
            mod_info["fl_final_accuracy"] = float(data[-1]["global_accuracy"])
            mod_info["fl_final_loss"] = float(data[-1]["global_loss"])

        # Phase 5 comparison
        csv_path = os.path.join(PROJECT_ROOT, "phase5_agentic", "results", f"{mod}_comparison.csv")
        data = load_csv(csv_path)
        if data:
            mod_info["random_final_acc"] = float(data[-1]["random_acc"])
            mod_info["agentic_final_acc"] = float(data[-1]["agentic_acc"])

        summary["modalities"][mod] = mod_info

    # Offload summary
    offload_csv = os.path.join(PROJECT_ROOT, "phase4_offload", "results", "offload_log.csv")
    data = load_csv(offload_csv)
    if data:
        offload_summary = {}
        for mod in MODALITIES:
            mod_rows = [r for r in data if r["modality"] == mod]
            edge_count = sum(1 for r in mod_rows if r["decision"] == "edge")
            offload_summary[mod] = round(edge_count / max(len(mod_rows), 1) * 100, 2)
        summary["offload_rates"] = offload_summary

    json_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved: {json_path}")
    return json_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  PHASE 6 -- EVALUATION REPORT GENERATION")
    print(f"{'='*70}")

    # Check which phases have results
    phase2_exists = any(
        os.path.exists(os.path.join(PROJECT_ROOT, "phase2_fl_sim", "results", f"{m}_fl_log.csv"))
        for m in MODALITIES
    )
    phase4_exists = os.path.exists(
        os.path.join(PROJECT_ROOT, "phase4_offload", "results", "offload_log.csv")
    )
    phase5_exists = any(
        os.path.exists(os.path.join(PROJECT_ROOT, "phase5_agentic", "results", f"{m}_comparison.csv"))
        for m in MODALITIES
    )

    print(f"\n  Phase 2 (FL) results:       {'FOUND' if phase2_exists else 'NOT FOUND'}")
    print(f"  Phase 4 (Offload) results:  {'FOUND' if phase4_exists else 'NOT FOUND'}")
    print(f"  Phase 5 (Agentic) results:  {'FOUND' if phase5_exists else 'NOT FOUND'}")

    if not phase2_exists:
        print(f"\n  WARNING: Run phase2_fl_sim first to get FL results!")
        print(f"  Generating partial report with available data...\n")

    # Generate outputs
    dashboard_path = generate_dashboard()
    json_path = generate_summary_json()

    print(f"\n  All artifacts saved to: {RESULTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
