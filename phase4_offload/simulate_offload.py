"""
simulate_offload.py -- Multi-Round Offload Simulation
=====================================================
Simulates 20 rounds of offload decisions for 50 clients across 4 modalities
(200 total decisions per round), with randomized client resource states.

Done when: video/audio offload rate > text/image offload rate.

Usage:
    cd <project_root>
    python -m phase4_offload.simulate_offload
"""

import os
import sys
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase4_offload.offload_decision import should_offload, get_decision_details

# ── Configuration ───────────────────────────────────────────────────────────
NUM_ROUNDS = 20
CLIENTS_PER_MODALITY = 50
MODALITIES = ["image", "text", "audio", "video"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 42


def simulate_client_state(rng, modality, client_id):
    """Generate randomized client resource state."""
    return {
        "modality": modality,
        "client_id": client_id,
        "device_cpu_ghz": round(rng.uniform(0.8, 2.5), 2),
        "battery_pct": round(rng.uniform(10, 100), 1),
        "network_latency_ms": round(rng.uniform(5, 180), 1),
        "privacy_flag": bool(rng.random() < 0.10),  # 10% chance
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"\n{'='*70}")
    print(f"  PHASE 4 -- OFFLOAD DECISION SIMULATION")
    print(f"  {NUM_ROUNDS} rounds x {len(MODALITIES)} modalities x {CLIENTS_PER_MODALITY} clients")
    print(f"{'='*70}")

    # Open CSV log
    log_path = os.path.join(RESULTS_DIR, "offload_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "round", "modality", "client_id", "decision",
        "device_cpu_ghz", "battery_pct", "network_latency_ms",
        "privacy_flag", "cost_local", "cost_edge", "reason"
    ])

    # Track offload counts
    offload_counts = {mod: {"local": 0, "edge": 0} for mod in MODALITIES}
    per_round_offload = {mod: [] for mod in MODALITIES}

    for rnd in range(1, NUM_ROUNDS + 1):
        round_offload = {mod: 0 for mod in MODALITIES}
        round_total = {mod: 0 for mod in MODALITIES}

        for modality in MODALITIES:
            for cid in range(CLIENTS_PER_MODALITY):
                state = simulate_client_state(rng, modality, cid)
                details = get_decision_details(state)
                decision = details["decision"]

                offload_counts[modality][decision] += 1
                round_total[modality] += 1
                if decision == "edge":
                    round_offload[modality] += 1

                writer.writerow([
                    rnd, modality, cid, decision,
                    state["device_cpu_ghz"], state["battery_pct"],
                    state["network_latency_ms"], state["privacy_flag"],
                    details.get("cost_local", ""), details.get("cost_edge", ""),
                    details.get("reason", ""),
                ])

        for mod in MODALITIES:
            rate = round_offload[mod] / max(round_total[mod], 1) * 100
            per_round_offload[mod].append(rate)

    log_file.close()
    print(f"\n  Decision log saved: {log_path}")

    # Summary
    print(f"\n  {'Modality':>8}  {'Local':>8}  {'Edge':>8}  {'Offload %':>10}")
    print(f"  {'-'*38}")

    offload_rates = {}
    for mod in MODALITIES:
        local = offload_counts[mod]["local"]
        edge = offload_counts[mod]["edge"]
        total = local + edge
        rate = edge / total * 100
        offload_rates[mod] = rate
        print(f"  {mod:>8}  {local:>8}  {edge:>8}  {rate:>9.1f}%")

    # Check if heavy modalities offload more
    heavy_avg = (offload_rates["video"] + offload_rates["audio"]) / 2
    light_avg = (offload_rates["image"] + offload_rates["text"]) / 2
    passed = heavy_avg > light_avg

    print(f"\n  Heavy modalities (video+audio) avg offload: {heavy_avg:.1f}%")
    print(f"  Light modalities (image+text)  avg offload: {light_avg:.1f}%")
    print(f"  Result: {'PASSED' if passed else 'FAILED'} "
          f"-- heavy offloads {'>' if passed else '<='} light offloads")

    # Generate plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 4 -- Offload Decision Analysis", fontsize=14, fontweight="bold")

    colors = {"image": "#4CAF50", "text": "#2196F3", "audio": "#FF9800", "video": "#E91E63"}

    # Bar chart: overall offload rate per modality
    mods = MODALITIES
    rates = [offload_rates[m] for m in mods]
    bars = ax1.bar(mods, rates, color=[colors[m] for m in mods], alpha=0.85)
    ax1.set_ylabel("Offload to Edge (%)")
    ax1.set_title("Overall Offload Rate by Modality")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Line chart: offload rate per round
    rounds = range(1, NUM_ROUNDS + 1)
    for mod in MODALITIES:
        ax2.plot(rounds, per_round_offload[mod], "o-", color=colors[mod],
                linewidth=2, markersize=5, label=mod.capitalize())
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Offload Rate (%)")
    ax2.set_title("Offload Rate per Round")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "offload_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # Energy/latency savings estimate
    print(f"\n  Estimated Savings vs. Always-Local Baseline:")
    print(f"  {'Modality':>8}  {'Energy Saved':>14}  {'Latency Saved':>14}")
    print(f"  {'-'*40}")
    for mod in MODALITIES:
        # Rough estimates: offloading saves ~60% energy, ~40% latency for heavy models
        energy_saving = offload_rates[mod] * 0.6  # % energy saved
        latency_saving = offload_rates[mod] * 0.4  # % latency saved
        print(f"  {mod:>8}  {energy_saving:>13.1f}%  {latency_saving:>13.1f}%")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
