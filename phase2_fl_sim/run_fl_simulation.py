"""
run_fl_simulation.py -- Independent FL Simulation per Modality
==============================================================
Runs FedAvg federated learning simulation for all 4 modalities independently.
50 clients per modality, 20 rounds, all clients participate each round.

Usage:
    cd <project_root>
    python -m phase2_fl_sim.run_fl_simulation
    python -m phase2_fl_sim.run_fl_simulation --rounds 10 --modality image
"""

import os
import sys
import copy
import time
import csv
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase1_baselines.models import get_model, count_parameters
from phase2_fl_sim.fl_utils import (
    get_parameters, set_parameters, fedavg_aggregate,
    local_train, local_evaluate,
)
from torch_dataset_wrappers import get_all_client_dataloaders, get_global_test_dataloader


# ── Configuration ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
LR = 0.001
NUM_CLIENTS = 50
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_fl_for_modality(
    modality: str,
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    lr: float = LR,
    device: str = DEVICE,
) -> dict:
    """
    Run a complete FedAvg FL simulation for one modality.

    Returns dict with keys: 'rounds', 'global_accuracy', 'global_loss'
    """
    print(f"\n{'='*60}")
    print(f"  FL SIMULATION -- {modality.upper()}")
    print(f"  {num_rounds} rounds, {local_epochs} local epoch(s), lr={lr}")
    print(f"{'='*60}")

    # Load data
    train_loaders, test_loaders = get_all_client_dataloaders(
        modality, num_clients=NUM_CLIENTS, batch_size=32
    )
    global_test_loader = get_global_test_dataloader(modality, batch_size=64)

    actual_clients = len(train_loaders)
    print(f"  Clients: {actual_clients} | Device: {device}")
    print(f"  Global test set: {len(global_test_loader.dataset)} samples")

    # Initialize global model
    global_model = get_model(modality)
    n_params = count_parameters(global_model)
    print(f"  Model: {global_model.__class__.__name__} | Params: {n_params:,}")

    # History tracking
    history = {"rounds": [], "global_accuracy": [], "global_loss": []}

    # Initial evaluation
    loss0, acc0, _ = local_evaluate(global_model, global_test_loader, device)
    print(f"\n  Initial: loss={loss0:.4f}, acc={acc0*100:.2f}%")

    print(f"\n  {'Round':>5}  {'Loss':>8}  {'Acc':>8}  {'Clients':>8}  {'Time':>6}")
    print(f"  {'-'*42}")

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        global_params = get_parameters(global_model)

        # All clients participate (Phase 2 = no selection)
        round_results = []

        for cid in range(actual_clients):
            # Create a local copy of the global model
            local_model = get_model(modality)
            set_parameters(local_model, copy.deepcopy(global_params))

            # Local training
            avg_loss, n_samples = local_train(
                local_model, train_loaders[cid],
                epochs=local_epochs, lr=lr, device=device
            )

            # Collect updated parameters
            updated_params = get_parameters(local_model)
            round_results.append((updated_params, n_samples))

        # FedAvg aggregation
        new_global_params = fedavg_aggregate(round_results)
        set_parameters(global_model, new_global_params)

        # Evaluate global model
        g_loss, g_acc, _ = local_evaluate(global_model, global_test_loader, device)

        elapsed = time.time() - t0

        history["rounds"].append(rnd)
        history["global_accuracy"].append(g_acc)
        history["global_loss"].append(g_loss)

        print(f"  {rnd:>5}  {g_loss:>8.4f}  {g_acc*100:>7.2f}%  {actual_clients:>8}  {elapsed:>5.1f}s")

    final_acc = history["global_accuracy"][-1]
    print(f"\n  Final: loss={history['global_loss'][-1]:.4f}, acc={final_acc*100:.2f}%")
    print(f"{'='*60}")

    return history


def save_results(modality: str, history: dict):
    """Save CSV log and accuracy plot for one modality."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"{modality}_fl_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_loss", "global_accuracy"])
        for r, loss, acc in zip(history["rounds"], history["global_loss"], history["global_accuracy"]):
            writer.writerow([r, f"{loss:.6f}", f"{acc:.6f}"])
    print(f"  CSV saved: {csv_path}")

    return csv_path


def plot_all_results(all_histories: dict):
    """Plot accuracy-vs-round for all modalities on one figure."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    colors = {"image": "#4CAF50", "text": "#2196F3", "audio": "#FF9800", "video": "#E91E63"}
    random_baselines = {"image": 10, "text": 25, "audio": 10, "video": 20}

    # Individual plots per modality
    for modality, history in all_histories.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Phase 2 -- FedAvg FL: {modality.upper()}", fontsize=14, fontweight="bold")

        rounds = history["rounds"]
        accs = [a * 100 for a in history["global_accuracy"]]
        losses = history["global_loss"]

        ax1.plot(rounds, accs, "o-", color=colors[modality], linewidth=2, markersize=6)
        ax1.axhline(y=random_baselines[modality], color="red", linestyle=":", alpha=0.5,
                    label=f"Random ({random_baselines[modality]}%)")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Global Accuracy (%)")
        ax1.set_title("Accuracy vs Round")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2.plot(rounds, losses, "s-", color=colors[modality], linewidth=2, markersize=6)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Global Loss")
        ax2.set_title("Loss vs Round")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"{modality}_fl_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {path}")

    # Combined overlay plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 2 -- FedAvg FL: All Modalities", fontsize=14, fontweight="bold")

    for modality, history in all_histories.items():
        rounds = history["rounds"]
        accs = [a * 100 for a in history["global_accuracy"]]
        losses = history["global_loss"]

        ax1.plot(rounds, accs, "o-", color=colors[modality], linewidth=2,
                markersize=5, label=modality.capitalize())
        ax2.plot(rounds, losses, "s-", color=colors[modality], linewidth=2,
                markersize=5, label=modality.capitalize())

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Global Accuracy (%)")
    ax1.set_title("Accuracy vs Round")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Global Loss")
    ax2.set_title("Loss vs Round")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "all_modalities_fl_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Combined plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run FL simulation per modality")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "image", "text", "audio", "video"])
    args = parser.parse_args()

    modalities = ["image", "text", "audio", "video"] if args.modality == "all" else [args.modality]
    all_histories = {}

    for modality in modalities:
        history = run_fl_for_modality(
            modality, num_rounds=args.rounds,
            local_epochs=args.local_epochs, lr=args.lr,
        )
        save_results(modality, history)
        all_histories[modality] = history

    # Generate plots
    plot_all_results(all_histories)

    # Summary
    print(f"\n{'='*60}")
    print(f"  PHASE 2 SUMMARY")
    print(f"{'='*60}")
    for modality, history in all_histories.items():
        final_acc = history["global_accuracy"][-1] * 100
        print(f"  {modality:>6}: Final accuracy = {final_acc:.2f}%")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
