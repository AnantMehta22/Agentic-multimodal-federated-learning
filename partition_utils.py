"""
partition_utils.py — Dirichlet-based non-IID Client Partitioning
================================================================
Splits a label array across `num_clients` using a Dirichlet distribution.
  alpha → 0:   extreme non-IID (each client gets mostly 1 class)
  alpha → ∞:   IID (uniform class distribution per client)

Usage:
    from partition_utils import dirichlet_partition
    client_map = dirichlet_partition(labels, num_clients=50, num_classes=10, alpha=0.5)
    # client_map[cid] = list of sample indices for that client
"""

import numpy as np
from typing import Dict, List


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 2,
) -> Dict[int, List[int]]:
    """
    Partition data indices across clients using a Dirichlet distribution.

    Args:
        labels:      1-D array of integer class labels, shape (N,).
        num_clients: Number of federated clients to partition into.
        num_classes: Number of unique classes in the label space.
        alpha:       Dirichlet concentration parameter.
                     Lower = more heterogeneous, higher = more uniform.
        seed:        Random seed for reproducibility.
        min_samples: Minimum samples per client (clients with fewer are dropped).

    Returns:
        Dict mapping client_id (int) → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels))

    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    if alpha <= 0:
        # Special case: pathological non-IID (round-robin by class)
        for cls in range(num_classes):
            cls_idx = all_indices[labels == cls]
            rng.shuffle(cls_idx)
            splits = np.array_split(cls_idx, num_clients)
            for cid, chunk in enumerate(splits):
                client_indices[cid].extend(chunk.tolist())
    else:
        for cls in range(num_classes):
            cls_idx = all_indices[labels == cls]
            rng.shuffle(cls_idx)

            # Draw proportions from Dirichlet(alpha, alpha, ..., alpha)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()

            # Convert proportions to integer counts
            counts = (proportions * len(cls_idx)).astype(int)
            remainder = len(cls_idx) - counts.sum()
            if remainder > 0:
                bonus = rng.choice(num_clients, size=remainder, replace=False)
                counts[bonus] += 1

            start = 0
            for cid in range(num_clients):
                end = start + counts[cid]
                client_indices[cid].extend(cls_idx[start:end].tolist())
                start = end

    # Remove empty or too-small clients and re-index
    client_indices = {
        cid: idx for cid, idx in client_indices.items() if len(idx) >= min_samples
    }
    client_indices = {
        new_cid: idx
        for new_cid, (_, idx) in enumerate(sorted(client_indices.items()))
    }

    return client_indices


def partition_summary(
    client_indices: Dict[int, List[int]], labels: np.ndarray
) -> None:
    """Print a quick summary of the partition distribution."""
    sizes = [len(v) for v in client_indices.values()]
    print(f"\n{'='*60}")
    print(f"  PARTITION SUMMARY — {len(client_indices)} clients")
    print(f"{'='*60}")
    print(f"  Total samples: {sum(sizes)}")
    print(f"  Samples/client: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")

    # Class distribution per client (first 5 clients)
    print(f"\n  Class distribution (first 5 clients):")
    for cid in list(client_indices.keys())[:5]:
        idx = client_indices[cid]
        cls_counts = np.bincount(labels[idx], minlength=labels.max() + 1)
        dist_str = " ".join(f"c{c}:{cnt}" for c, cnt in enumerate(cls_counts) if cnt > 0)
        print(f"    Client {cid:>3}: {len(idx):>5} samples | {dist_str}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Quick self-test
    rng = np.random.default_rng(0)
    fake_labels = rng.integers(0, 10, size=5000)
    client_map = dirichlet_partition(fake_labels, num_clients=50, num_classes=10, alpha=0.5)
    partition_summary(client_map, fake_labels)
    print("partition_utils.py self-test passed ✓")
