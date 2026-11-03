"""
fl_utils.py -- Federated Learning Utilities
============================================
Shared FL utilities for all phases:
  - get_parameters / set_parameters: NumPy <-> PyTorch conversion
  - fedavg_aggregate: weighted parameter averaging (FedAvg)
  - local_train / local_evaluate: client-side training/evaluation
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from collections import OrderedDict


# ── Parameter Serialization ─────────────────────────────────────────────────

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load parameters from NumPy arrays into a model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# ── FedAvg Aggregation ──────────────────────────────────────────────────────

def fedavg_aggregate(
    results: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """
    Weighted Federated Averaging of model parameters.

    Args:
        results: List of (parameters, num_samples) tuples from clients.

    Returns:
        Aggregated parameters as list of NumPy arrays.
    """
    total_samples = sum(n for _, n in results)
    if total_samples == 0:
        return results[0][0]

    # Weighted average
    aggregated = [
        np.zeros_like(param, dtype=np.float64) for param in results[0][0]
    ]
    for params, num_samples in results:
        weight = num_samples / total_samples
        for i, param in enumerate(params):
            aggregated[i] += param.astype(np.float64) * weight

    # Cast back to original dtypes
    aggregated = [
        agg.astype(orig.dtype) for agg, orig in zip(aggregated, results[0][0])
    ]

    return aggregated


# ── Local Training ──────────────────────────────────────────────────────────

def local_train(
    model: nn.Module,
    train_loader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
) -> Tuple[float, int]:
    """
    Train model locally on client data.

    Returns:
        (average_loss, num_training_samples)
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            num_samples += y_batch.size(0)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, num_samples


def local_evaluate(
    model: nn.Module,
    test_loader,
    device: str = "cpu",
) -> Tuple[float, float, int]:
    """
    Evaluate model on test data.

    Returns:
        (average_loss, accuracy, num_samples)
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, total


def compute_update_norm(
    params_before: List[np.ndarray],
    params_after: List[np.ndarray],
) -> float:
    """Compute L2 norm of the parameter update (delta)."""
    total_norm = 0.0
    for p_before, p_after in zip(params_before, params_after):
        delta = p_after - p_before
        total_norm += np.sum(delta ** 2)
    return float(np.sqrt(total_norm))


def compute_update_cosine_similarity(
    update_a: List[np.ndarray],
    update_b: List[np.ndarray],
) -> float:
    """Compute cosine similarity between two parameter updates."""
    flat_a = np.concatenate([u.flatten() for u in update_a])
    flat_b = np.concatenate([u.flatten() for u in update_b])
    dot = np.dot(flat_a, flat_b)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_parameter_delta(
    params_before: List[np.ndarray],
    params_after: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute the parameter update (delta = after - before)."""
    return [after - before for before, after in zip(params_before, params_after)]
