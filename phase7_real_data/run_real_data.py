"""
run_real_data.py -- FL on Real Datasets (Phase 7)
=================================================
Reuses Phase 2/5 FL pipeline with real_data_loaders.py.
Run order: CIFAR-10 -> AG News -> Speech Commands -> UCF101 subset.

Usage:
    cd <project_root>
    python -m phase7_real_data.run_real_data --dataset cifar10
    python -m phase7_real_data.run_real_data --dataset agnews
    python -m phase7_real_data.run_real_data --dataset all
"""

import os
import sys
import copy
import csv
import time
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase1_baselines.models import get_model, count_parameters
from phase2_fl_sim.fl_utils import (
    get_parameters, set_parameters, fedavg_aggregate,
    local_train, local_evaluate,
)
from real_data_loaders import load_cifar10, load_agnews, load_speech_commands, load_ucf101_subset

# ── Configuration ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
LR = 0.001
NUM_CLIENTS = 50
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DATASET_CONFIG = {
    "cifar10": {"modality": "image", "loader": load_cifar10, "num_classes": 10},
    "agnews": {"modality": "text", "loader": load_agnews, "num_classes": 4},
    "speech": {"modality": "audio", "loader": load_speech_commands, "num_classes": 10},
    "ucf101": {"modality": "video", "loader": load_ucf101_subset, "num_classes": 5},
}


def run_real_fl(dataset_name, num_rounds=NUM_ROUNDS):
    """Run FL on a real dataset."""
    config = DATASET_CONFIG[dataset_name]
    modality = config["modality"]

    print(f"\n{'='*60}")
    print(f"  PHASE 7 -- REAL DATA FL: {dataset_name.upper()} ({modality})")
    print(f"  {num_rounds} rounds, {NUM_CLIENTS} clients")
    print(f"{'='*60}")

    # Load real data
    print(f"\n  Loading {dataset_name}...")
    train_loaders, test_loaders, global_test_loader = config["loader"](
        num_clients=NUM_CLIENTS, alpha=0.5
    )
    actual_clients = len(train_loaders)
    print(f"  Loaded: {actual_clients} clients")

    # Initialize global model
    global_model = get_model(modality)
    n_params = count_parameters(global_model)
    print(f"  Model: {global_model.__class__.__name__} | Params: {n_params:,}")

    history = {"rounds": [], "accuracy": [], "loss": []}

    print(f"\n  {'Round':>5}  {'Loss':>8}  {'Acc':>8}  {'Clients':>8}  {'Time':>6}")
    print(f"  {'-'*42}")

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        global_params = get_parameters(global_model)
        round_results = []

        for cid in range(actual_clients):
            local_model = get_model(modality)
            set_parameters(local_model, copy.deepcopy(global_params))
            avg_loss, n_samples = local_train(
                local_model, train_loaders[cid],
                epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE
            )
            updated_params = get_parameters(local_model)
            round_results.append((updated_params, n_samples))

        new_params = fedavg_aggregate(round_results)
        set_parameters(global_model, new_params)

        g_loss, g_acc, _ = local_evaluate(global_model, global_test_loader, DEVICE)
        elapsed = time.time() - t0

        history["rounds"].append(rnd)
        history["accuracy"].append(g_acc)
        history["loss"].append(g_loss)

        print(f"  {rnd:>5}  {g_loss:>8.4f}  {g_acc*100:>7.2f}%  {actual_clients:>8}  {elapsed:>5.1f}s")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["all", "cifar10", "agnews", "speech", "ucf101"])
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    all_histories = {}

    for ds in datasets:
        history = run_real_fl(ds, num_rounds=args.rounds)
        all_histories[ds] = history

        # Save CSV
        csv_path = os.path.join(RESULTS_DIR, f"{ds}_fl_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "global_loss", "global_accuracy"])
            for r, loss, acc in zip(history["rounds"], history["loss"], history["accuracy"]):
                writer.writerow([r, f"{loss:.6f}", f"{acc:.6f}"])
        print(f"  CSV saved: {csv_path}")

    # Plot
    colors = {"cifar10": "#4CAF50", "agnews": "#2196F3", "speech": "#FF9800", "ucf101": "#E91E63"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 7 -- FL on Real Data", fontsize=14, fontweight="bold")

    for ds, history in all_histories.items():
        ax1.plot(history["rounds"], [a*100 for a in history["accuracy"]], "o-",
                color=colors[ds], linewidth=2, markersize=5, label=ds.upper())
        ax2.plot(history["rounds"], history["loss"], "s-",
                color=colors[ds], linewidth=2, markersize=5, label=ds.upper())

    ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Round"); ax2.set_ylabel("Loss"); ax2.set_title("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "real_data_fl_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  PHASE 7 SUMMARY")
    print(f"{'='*60}")
    for ds, h in all_histories.items():
        print(f"  {ds:>8}: Final acc = {h['accuracy'][-1]*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
