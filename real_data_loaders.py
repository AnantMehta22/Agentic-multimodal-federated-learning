"""
real_data_loaders.py -- Real Dataset Loaders for Phase 7
========================================================
Loaders for CIFAR-10 (image), AG News (text), Speech Commands (audio),
and UCF101 subset (video). Each gated behind explicit download.

Usage:
    from real_data_loaders import load_cifar10, load_agnews

    train_loaders, test_loader = load_cifar10(num_clients=50, alpha=0.5)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from partition_utils import dirichlet_partition


DATA_DIR = os.path.join(PROJECT_ROOT, "real_data")


# ── CIFAR-10 (Image) ───────────────────────────────────────────────────────

def load_cifar10(
    num_clients: int = 50,
    alpha: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Load CIFAR-10 and partition across clients using Dirichlet.

    Returns:
        (train_loaders, test_loaders, global_test_loader)
    """
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_path = os.path.join(DATA_DIR, "cifar10")
    os.makedirs(data_path, exist_ok=True)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )

    # Partition training data
    labels = np.array(train_dataset.targets)
    client_indices = dirichlet_partition(labels, num_clients, 10, alpha, seed)

    train_loaders = []
    test_loaders = []
    for cid in sorted(client_indices.keys()):
        indices = client_indices[cid]
        subset = Subset(train_dataset, indices)

        # Split 80/20 for local train/test
        n = len(indices)
        n_test = max(1, int(n * 0.2))
        train_sub = Subset(train_dataset, indices[n_test:])
        test_sub = Subset(train_dataset, indices[:n_test])

        train_loaders.append(DataLoader(train_sub, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_sub, batch_size=batch_size, shuffle=False))

    global_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"  CIFAR-10: {len(train_loaders)} clients, {len(test_dataset)} global test samples")
    return train_loaders, test_loaders, global_test_loader


# ── AG News (Text) ──────────────────────────────────────────────────────────

def load_agnews(
    num_clients: int = 50,
    alpha: float = 0.5,
    batch_size: int = 32,
    vocab_size: int = 5000,
    seq_len: int = 50,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Load AG News dataset, tokenize to fixed-length sequences, and partition.

    Returns:
        (train_loaders, test_loaders, global_test_loader)
    """
    try:
        import torchtext
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        tokenizer = get_tokenizer("basic_english")

        # Load dataset
        train_iter, test_iter = AG_NEWS(root=os.path.join(DATA_DIR, "agnews"))

        train_data = list(train_iter)
        test_data = list(test_iter)

        # Build vocab from training data
        def yield_tokens(data):
            for label, text in data:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(train_data), max_tokens=vocab_size,
                                         specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])

        # Tokenize and pad/truncate
        def encode(data):
            X_list, y_list = [], []
            for label, text in data:
                tokens = vocab(tokenizer(text))
                if len(tokens) >= seq_len:
                    tokens = tokens[:seq_len]
                else:
                    tokens = tokens + [vocab["<pad>"]] * (seq_len - len(tokens))
                X_list.append(tokens)
                y_list.append(label - 1)  # AG News labels are 1-4, convert to 0-3
            return np.array(X_list, dtype=np.int64), np.array(y_list, dtype=np.int64)

        X_train, y_train = encode(train_data)
        X_test, y_test = encode(test_data)

    except (ImportError, Exception) as e:
        print(f"  Warning: torchtext not available ({e})")
        print(f"  Falling back to synthetic-like AG News generation")
        # Fallback: generate synthetic text data matching AG News specs
        rng = np.random.default_rng(seed)
        X_train = rng.integers(0, vocab_size, size=(120000, seq_len)).astype(np.int64)
        y_train = rng.integers(0, 4, size=120000).astype(np.int64)
        X_test = rng.integers(0, vocab_size, size=(7600, seq_len)).astype(np.int64)
        y_test = rng.integers(0, 4, size=7600).astype(np.int64)

    # Partition
    client_indices = dirichlet_partition(y_train, num_clients, 4, alpha, seed)

    train_loaders, test_loaders = [], []
    for cid in sorted(client_indices.keys()):
        idx = client_indices[cid]
        n_test = max(1, int(len(idx) * 0.2))

        X_tr = torch.tensor(X_train[idx[n_test:]], dtype=torch.long)
        y_tr = torch.tensor(y_train[idx[n_test:]], dtype=torch.long)
        X_te = torch.tensor(X_train[idx[:n_test]], dtype=torch.long)
        y_te = torch.tensor(y_train[idx[:n_test]], dtype=torch.long)

        train_loaders.append(DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False))

    global_test = TensorDataset(torch.tensor(X_test, dtype=torch.long),
                                torch.tensor(y_test, dtype=torch.long))
    global_test_loader = DataLoader(global_test, batch_size=64, shuffle=False)

    print(f"  AG News: {len(train_loaders)} clients, {len(X_test)} global test samples")
    return train_loaders, test_loaders, global_test_loader


# ── Speech Commands (Audio) ─────────────────────────────────────────────────

def load_speech_commands(
    num_clients: int = 50,
    alpha: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Load a subset of Speech Commands, extract MFCC features, and partition.
    Falls back to synthetic audio if torchaudio is not available.
    """
    try:
        import torchaudio
        from torchaudio.transforms import MFCC

        data_path = os.path.join(DATA_DIR, "speech_commands")
        os.makedirs(data_path, exist_ok=True)

        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_path, download=True, subset="training"
        )

        # Use first 10 commands
        commands = sorted(list(set(item[2] for item in dataset)))[:10]
        cmd_to_idx = {cmd: i for i, cmd in enumerate(commands)}

        mfcc_transform = MFCC(sample_rate=16000, n_mfcc=40,
                              melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40})

        X_list, y_list = [], []
        for item in dataset:
            waveform, sr, label = item[0], item[1], item[2]
            if label not in cmd_to_idx:
                continue
            mfcc = mfcc_transform(waveform)
            # Pad/truncate to (1, 40, 100)
            if mfcc.shape[2] >= 100:
                mfcc = mfcc[:, :, :100]
            else:
                pad = torch.zeros(1, 40, 100 - mfcc.shape[2])
                mfcc = torch.cat([mfcc, pad], dim=2)
            X_list.append(mfcc.numpy())
            y_list.append(cmd_to_idx[label])

        X_all = np.array(X_list, dtype=np.float32)
        y_all = np.array(y_list, dtype=np.int64)

    except (ImportError, Exception) as e:
        print(f"  Warning: torchaudio not available ({e})")
        print(f"  Using synthetic audio data for Phase 7")
        from synthetic_data import generate_audio_data
        X_all, y_all = generate_audio_data(num_samples=3000, seed=seed)

    # Split train/test
    rng = np.random.default_rng(seed)
    n = len(y_all)
    perm = rng.permutation(n)
    n_test = max(100, int(n * 0.2))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    num_classes = len(np.unique(y_all))

    # Partition
    client_indices = dirichlet_partition(y_train, num_clients, num_classes, alpha, seed)

    train_loaders, test_loaders = [], []
    for cid in sorted(client_indices.keys()):
        idx = client_indices[cid]
        n_te = max(1, int(len(idx) * 0.2))
        X_tr = torch.tensor(X_train[idx[n_te:]], dtype=torch.float32)
        y_tr = torch.tensor(y_train[idx[n_te:]], dtype=torch.long)
        X_te = torch.tensor(X_train[idx[:n_te]], dtype=torch.float32)
        y_te = torch.tensor(y_train[idx[:n_te]], dtype=torch.long)
        train_loaders.append(DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False))

    global_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))
    global_test_loader = DataLoader(global_test, batch_size=64, shuffle=False)

    print(f"  Speech Commands: {len(train_loaders)} clients, {len(X_test)} global test")
    return train_loaders, test_loaders, global_test_loader


# ── UCF101 Subset (Video) ──────────────────────────────────────────────────

def load_ucf101_subset(
    num_clients: int = 50,
    alpha: float = 0.5,
    batch_size: int = 16,
    num_classes: int = 5,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Load a small subset of UCF101 (or fall back to synthetic video).
    UCF101 requires manual download; this provides a structured fallback.
    """
    print(f"  UCF101: Using synthetic video data (UCF101 requires manual download)")
    print(f"  To use real UCF101, download it and update this loader.")

    from synthetic_data import generate_video_data
    X_all, y_all = generate_video_data(num_samples=1500, num_classes=num_classes, seed=seed)

    # Split and partition
    rng = np.random.default_rng(seed)
    n = len(y_all)
    perm = rng.permutation(n)
    n_test = max(50, int(n * 0.2))
    X_train, y_train = X_all[perm[n_test:]], y_all[perm[n_test:]]
    X_test, y_test = X_all[perm[:n_test]], y_all[perm[:n_test]]

    client_indices = dirichlet_partition(y_train, num_clients, num_classes, alpha, seed)

    train_loaders, test_loaders = [], []
    for cid in sorted(client_indices.keys()):
        idx = client_indices[cid]
        n_te = max(1, int(len(idx) * 0.2))
        X_tr = torch.tensor(X_train[idx[n_te:]], dtype=torch.float32)
        y_tr = torch.tensor(y_train[idx[n_te:]], dtype=torch.long)
        X_te = torch.tensor(X_train[idx[:n_te]], dtype=torch.float32)
        y_te = torch.tensor(y_train[idx[:n_te]], dtype=torch.long)
        train_loaders.append(DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False))

    global_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))
    global_test_loader = DataLoader(global_test, batch_size=32, shuffle=False)

    print(f"  UCF101 subset: {len(train_loaders)} clients, {len(X_test)} global test")
    return train_loaders, test_loaders, global_test_loader
