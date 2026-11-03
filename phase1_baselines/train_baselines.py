"""
train_baselines.py — Centralized Baseline Training for All 4 Modalities
=======================================================================
Trains each model on aggregated synthetic data (no FL) to confirm every
architecture can learn the synthetic class structure above chance.

Done when: each modality's val accuracy is clearly above 1/num_classes
  - Image: > 10% (10 classes → random = 10%)
  - Text:  > 25% (4 classes  → random = 25%)
  - Audio: > 10% (10 classes → random = 10%)
  - Video: > 20% (5 classes  → random = 20%)

Usage:
    cd <project_root>
    python -m phase1_baselines.train_baselines
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from synthetic_data import (
    generate_image_data, generate_text_data,
    generate_audio_data, generate_video_data,
)
from phase1_baselines.models import get_model, count_parameters


# ── Configuration ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_modality(modality, X, y, num_classes, dtype_X="float32"):
    """Train a single modality model and return history."""
    print(f"\n{'='*60}")
    print(f"  Training {modality.upper()} — {num_classes} classes, "
          f"random baseline = {100/num_classes:.1f}%")
    print(f"{'='*60}")

    # Create train/val split (80/20)
    rng = np.random.default_rng(42)
    n = len(y)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    if dtype_X == "int64":
        X_train = torch.tensor(X[train_idx], dtype=torch.long)
        X_val = torch.tensor(X[val_idx], dtype=torch.long)
    else:
        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)

    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = get_model(modality).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Model: {model.__class__.__name__} | Params: {n_params:,} | Device: {DEVICE}")
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Train Acc':>10}  {'Val Acc':>10}  {'Time':>6}")
    print(f"  {'-'*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Evaluate on train and val
        _, train_acc = evaluate(model, train_loader, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{train_acc*100:>9.2f}%  {val_acc*100:>9.2f}%  {elapsed:>5.1f}s")

    # Final verdict
    final_val_acc = history["val_acc"][-1]
    random_acc = 1.0 / num_classes
    passed = final_val_acc > random_acc * 1.5  # Should be clearly above chance

    print(f"\n  Result: Val Accuracy = {final_val_acc*100:.2f}% "
          f"(random = {random_acc*100:.1f}%) "
          f"-> {'PASSED' if passed else 'FAILED'}")

    return history, passed


def plot_results(all_histories):
    """Generate training curves for all modalities."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 1 — Centralized Baseline Training", fontsize=16, fontweight="bold")

    modalities = list(all_histories.keys())
    colors = {"image": "#4CAF50", "text": "#2196F3", "audio": "#FF9800", "video": "#E91E63"}
    random_baselines = {"image": 10, "text": 25, "audio": 10, "video": 20}

    for idx, modality in enumerate(modalities):
        ax = axes[idx // 2][idx % 2]
        history = all_histories[modality]
        epochs = range(1, len(history["val_acc"]) + 1)

        ax.plot(epochs, [a * 100 for a in history["train_acc"]], "o-",
                color=colors[modality], label="Train Acc", linewidth=2, markersize=6)
        ax.plot(epochs, [a * 100 for a in history["val_acc"]], "s--",
                color=colors[modality], label="Val Acc", linewidth=2, markersize=6, alpha=0.7)
        ax.axhline(y=random_baselines[modality], color="red", linestyle=":",
                   alpha=0.5, label=f"Random ({random_baselines[modality]}%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{modality.upper()}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "baseline_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


def main():
    print(f"\n{'='*70}")
    print(f"  PHASE 1 — CENTRALIZED BASELINE TRAINING")
    print(f"  Verifying all 4 models can learn synthetic class structure")
    print(f"{'='*70}")

    # Generate all synthetic data
    generators = {
        "image": (generate_image_data, 10, "float32"),
        "text":  (generate_text_data,   4, "int64"),
        "audio": (generate_audio_data, 10, "float32"),
        "video": (generate_video_data,  5, "float32"),
    }

    all_histories = {}
    results = {}

    for modality, (gen_fn, num_classes, dtype_X) in generators.items():
        X, y = gen_fn(seed=42)
        history, passed = train_modality(modality, X, y, num_classes, dtype_X)
        all_histories[modality] = history
        results[modality] = passed

    # Generate plot
    plot_results(all_histories)

    # Summary
    print(f"\n{'='*70}")
    print(f"  PHASE 1 SUMMARY")
    print(f"{'='*70}")
    all_passed = True
    for modality, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        final_acc = all_histories[modality]["val_acc"][-1] * 100
        print(f"  {modality:>6}: {final_acc:>6.2f}% — {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  ALL MODELS LEARN ABOVE CHANCE — Phase 1 complete!")
    else:
        print(f"\n  Some models failed — check architectures")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
