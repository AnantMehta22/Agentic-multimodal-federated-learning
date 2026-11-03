"""
synthetic_data.py — Synthetic Data Generator for 4 Modalities
=============================================================
Generates synthetic image/text/audio/video data with learnable class structure,
then saves per-client shards to synthetic_fl_data/client_<id>/<modality>.npz.

Each modality has class-specific patterns so that even simple models can learn
above chance — essential for validating the FL pipeline.

Usage:
    python synthetic_data.py                    # Generate for 50 clients
    python synthetic_data.py --num_clients 10   # Fewer clients for quick test
"""

import os
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Import partitioning utility
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partition_utils import dirichlet_partition, partition_summary


# ── Configuration ───────────────────────────────────────────────────────────
DEFAULT_NUM_CLIENTS = 50
DEFAULT_ALPHA = 0.5
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_fl_data")

# Modality specs
IMAGE_SHAPE = (3, 32, 32)     # C, H, W
IMAGE_CLASSES = 10
IMAGE_SAMPLES = 5000

TEXT_SEQ_LEN = 50              # token sequence length
TEXT_VOCAB_SIZE = 5000
TEXT_CLASSES = 4
TEXT_SAMPLES = 4000

AUDIO_SHAPE = (1, 40, 100)    # C, n_mfcc, time_frames
AUDIO_CLASSES = 10
AUDIO_SAMPLES = 3000

VIDEO_SHAPE = (8, 3, 32, 32)  # T, C, H, W
VIDEO_CLASSES = 5
VIDEO_SAMPLES = 1500


# ── Synthetic Data Generators ──────────────────────────────────────────────

def generate_image_data(
    num_samples: int = IMAGE_SAMPLES,
    num_classes: int = IMAGE_CLASSES,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 32x32 RGB images with class-specific spatial patterns.
    Each class gets a different pattern (bars, corners, diagonals, etc.)
    overlaid on random noise, making the task learnable.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, *IMAGE_SHAPE)).astype(np.float32) * 0.3
    y = rng.integers(0, num_classes, size=num_samples)

    for i in range(num_samples):
        cls = y[i]
        # Each class gets a unique spatial pattern
        if cls == 0:    # Horizontal bar top
            X[i, :, 0:8, :] += 2.0
        elif cls == 1:  # Horizontal bar bottom
            X[i, :, 24:32, :] += 2.0
        elif cls == 2:  # Vertical bar left
            X[i, :, :, 0:8] += 2.0
        elif cls == 3:  # Vertical bar right
            X[i, :, :, 24:32] += 2.0
        elif cls == 4:  # Top-left corner
            X[i, :, 0:12, 0:12] += 2.0
        elif cls == 5:  # Top-right corner
            X[i, :, 0:12, 20:32] += 2.0
        elif cls == 6:  # Bottom-left corner
            X[i, :, 20:32, 0:12] += 2.0
        elif cls == 7:  # Bottom-right corner
            X[i, :, 20:32, 20:32] += 2.0
        elif cls == 8:  # Center blob
            X[i, :, 10:22, 10:22] += 2.0
        elif cls == 9:  # Diagonal
            for k in range(32):
                X[i, :, k, max(0, k-2):min(32, k+3)] += 2.0

    return X, y


def generate_text_data(
    num_samples: int = TEXT_SAMPLES,
    num_classes: int = TEXT_CLASSES,
    seq_len: int = TEXT_SEQ_LEN,
    vocab_size: int = TEXT_VOCAB_SIZE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic token-ID sequences with class-biased distributions.
    Each class favors a different range of the vocabulary, making it possible
    for an LSTM to learn class-distinguishing token patterns.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=num_samples)

    # Each class biases toward a different vocabulary slice
    slice_size = vocab_size // num_classes
    X = np.zeros((num_samples, seq_len), dtype=np.int64)

    for i in range(num_samples):
        cls = y[i]
        # 70% of tokens from class-specific slice, 30% random
        n_biased = int(seq_len * 0.7)
        n_random = seq_len - n_biased
        biased_start = cls * slice_size
        biased_end = biased_start + slice_size
        biased_tokens = rng.integers(biased_start, biased_end, size=n_biased)
        random_tokens = rng.integers(0, vocab_size, size=n_random)
        all_tokens = np.concatenate([biased_tokens, random_tokens])
        rng.shuffle(all_tokens)
        X[i] = all_tokens

    return X, y


def generate_audio_data(
    num_samples: int = AUDIO_SAMPLES,
    num_classes: int = AUDIO_CLASSES,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic MFCC-like spectrograms with class-specific
    frequency band emphasis. Shape: (N, 1, 40, 100).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, *AUDIO_SHAPE)).astype(np.float32) * 0.3
    y = rng.integers(0, num_classes, size=num_samples)

    bands_per_class = 40 // num_classes  # 4 frequency bands per class

    for i in range(num_samples):
        cls = y[i]
        band_start = cls * bands_per_class
        band_end = band_start + bands_per_class
        # Emphasize this class's frequency band
        X[i, 0, band_start:band_end, :] += 2.5
        # Add class-specific temporal pattern (pulse vs. sustained)
        if cls % 2 == 0:
            # Even classes: periodic pulses
            X[i, 0, band_start:band_end, ::10] += 1.5
        else:
            # Odd classes: sustained tone
            X[i, 0, band_start:band_end, 20:80] += 1.0

    return X, y


def generate_video_data(
    num_samples: int = VIDEO_SAMPLES,
    num_classes: int = VIDEO_CLASSES,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 8-frame video clips with class-specific motion patterns.
    Shape: (N, 8, 3, 32, 32).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, *VIDEO_SHAPE)).astype(np.float32) * 0.2
    y = rng.integers(0, num_classes, size=num_samples)

    for i in range(num_samples):
        cls = y[i]
        for t in range(8):
            if cls == 0:    # Object moving right
                col = min(28, t * 4)
                X[i, t, :, 12:20, col:col+4] += 3.0
            elif cls == 1:  # Object moving down
                row = min(28, t * 4)
                X[i, t, :, row:row+4, 12:20] += 3.0
            elif cls == 2:  # Expanding center
                size = 2 + t
                c = 16
                X[i, t, :, c-size:c+size, c-size:c+size] += 2.5
            elif cls == 3:  # Blinking (on every other frame)
                if t % 2 == 0:
                    X[i, t, :, 8:24, 8:24] += 3.0
            elif cls == 4:  # Static top-left + fading
                intensity = 3.0 * (1.0 - t / 8.0)
                X[i, t, :, 0:16, 0:16] += intensity

    return X, y


# ── Data Saving / Loading ──────────────────────────────────────────────────

def save_client_shards(
    X: np.ndarray,
    y: np.ndarray,
    client_indices: Dict[int, List[int]],
    modality: str,
    data_dir: str = DATA_DIR,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Save per-client train/test shards as .npz files."""
    rng = np.random.default_rng(seed)

    for cid, indices in client_indices.items():
        client_dir = os.path.join(data_dir, f"client_{cid}")
        os.makedirs(client_dir, exist_ok=True)

        idx = np.array(indices)
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_fraction))
        train_idx = idx[n_test:]
        test_idx = idx[:n_test]

        np.savez_compressed(
            os.path.join(client_dir, f"{modality}_train.npz"),
            X=X[train_idx],
            y=y[train_idx],
        )
        np.savez_compressed(
            os.path.join(client_dir, f"{modality}_test.npz"),
            X=X[test_idx],
            y=y[test_idx],
        )

    # Also save a global test set (aggregated from all clients' test data)
    global_dir = os.path.join(data_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    all_idx = np.concatenate([np.array(idx) for idx in client_indices.values()])
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(all_idx)
    n_global_test = max(10, int(len(all_idx) * test_fraction))
    global_test_idx = all_idx[:n_global_test]
    np.savez_compressed(
        os.path.join(global_dir, f"{modality}_test.npz"),
        X=X[global_test_idx],
        y=y[global_test_idx],
    )


def generate_all(num_clients: int = DEFAULT_NUM_CLIENTS, alpha: float = DEFAULT_ALPHA, seed: int = 42):
    """Generate synthetic data for all 4 modalities and partition across clients."""
    print(f"\n{'='*70}")
    print(f"  SYNTHETIC DATA GENERATION — {num_clients} clients, alpha={alpha}")
    print(f"{'='*70}")

    generators = [
        ("image", generate_image_data, IMAGE_SAMPLES, IMAGE_CLASSES),
        ("text",  generate_text_data,  TEXT_SAMPLES,  TEXT_CLASSES),
        ("audio", generate_audio_data, AUDIO_SAMPLES, AUDIO_CLASSES),
        ("video", generate_video_data, VIDEO_SAMPLES, VIDEO_CLASSES),
    ]

    for modality, gen_fn, num_samples, num_classes in generators:
        print(f"\n  Generating {modality} data: {num_samples} samples, {num_classes} classes...")
        X, y = gen_fn(num_samples=num_samples, num_classes=num_classes, seed=seed)
        print(f"    X.shape={X.shape}, y.shape={y.shape}")

        # Partition across clients using Dirichlet
        client_indices = dirichlet_partition(
            y, num_clients=num_clients, num_classes=num_classes, alpha=alpha, seed=seed
        )
        print(f"    Partitioned into {len(client_indices)} clients")

        # Save per-client shards
        save_client_shards(X, y, client_indices, modality, DATA_DIR, seed=seed)
        print(f"    Saved to {DATA_DIR}/client_*/")

    print(f"\n{'='*70}")
    print(f"  ALL DONE — data saved to {DATA_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic FL data")
    parser.add_argument("--num_clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_all(num_clients=args.num_clients, alpha=args.alpha, seed=args.seed)
