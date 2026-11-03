"""
torch_dataset_wrappers.py — PyTorch Dataset/DataLoader Wrappers
===============================================================
Loads per-client .npz shards into PyTorch Datasets and DataLoaders.

Usage:
    from torch_dataset_wrappers import get_client_dataloader, get_global_test_dataloader

    train_loader = get_client_dataloader(client_id=0, modality="image", train=True)
    test_loader  = get_global_test_dataloader(modality="image")
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_fl_data")


class ModalityDataset(Dataset):
    """PyTorch Dataset that loads from a .npz file with keys 'X' and 'y'."""

    def __init__(self, npz_path: str, dtype_X: str = "float32"):
        """
        Args:
            npz_path: Path to a .npz file containing 'X' and 'y' arrays.
            dtype_X:  Data type for X tensor. Use 'float32' for image/audio/video,
                      'int64' for text token IDs.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")

        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]
        self.dtype_X = dtype_X

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.dtype_X == "int64":
            x_tensor = torch.tensor(x, dtype=torch.long)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor


def get_client_dataloader(
    client_id: int,
    modality: str,
    train: bool = True,
    batch_size: int = 32,
    data_dir: str = DATA_DIR,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Get a DataLoader for a specific client and modality.

    Args:
        client_id: Client index (0-based).
        modality:  One of 'image', 'text', 'audio', 'video'.
        train:     If True, load training shard; else load test shard.
        batch_size: Batch size for the DataLoader.
        data_dir:  Root directory for synthetic data.
        shuffle:   Shuffle the data. Defaults to True for train, False for test.

    Returns:
        PyTorch DataLoader for the requested client/modality/split.
    """
    split = "train" if train else "test"
    npz_path = os.path.join(data_dir, f"client_{client_id}", f"{modality}_{split}.npz")

    dtype_X = "int64" if modality == "text" else "float32"
    dataset = ModalityDataset(npz_path, dtype_X=dtype_X)

    if shuffle is None:
        shuffle = train

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def get_global_test_dataloader(
    modality: str,
    batch_size: int = 64,
    data_dir: str = DATA_DIR,
) -> DataLoader:
    """
    Get a DataLoader for the global test set of a given modality.

    Returns:
        PyTorch DataLoader for global test evaluation.
    """
    npz_path = os.path.join(data_dir, "global", f"{modality}_test.npz")
    dtype_X = "int64" if modality == "text" else "float32"
    dataset = ModalityDataset(npz_path, dtype_X=dtype_X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def get_all_client_dataloaders(
    modality: str,
    num_clients: int = 50,
    batch_size: int = 32,
    data_dir: str = DATA_DIR,
) -> Tuple[list, list]:
    """
    Get train and test DataLoaders for all clients of a modality.

    Returns:
        (train_loaders, test_loaders): Lists of DataLoaders, one per client.
    """
    train_loaders = []
    test_loaders = []

    for cid in range(num_clients):
        client_dir = os.path.join(data_dir, f"client_{cid}")
        train_path = os.path.join(client_dir, f"{modality}_train.npz")
        test_path = os.path.join(client_dir, f"{modality}_test.npz")

        if not os.path.exists(train_path):
            break  # No more clients for this modality

        dtype_X = "int64" if modality == "text" else "float32"
        train_ds = ModalityDataset(train_path, dtype_X=dtype_X)
        test_ds = ModalityDataset(test_path, dtype_X=dtype_X)

        train_loaders.append(
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        )
        test_loaders.append(
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        )

    return train_loaders, test_loaders


if __name__ == "__main__":
    # Quick self-test: check that synthetic data loads correctly
    for mod in ["image", "text", "audio", "video"]:
        try:
            loader = get_client_dataloader(0, mod, train=True, batch_size=4)
            X_batch, y_batch = next(iter(loader))
            print(f"{mod:>6}: X={tuple(X_batch.shape)}, y={tuple(y_batch.shape)}, "
                  f"dtype_X={X_batch.dtype}, dtype_y={y_batch.dtype}")
        except FileNotFoundError:
            print(f"{mod:>6}: [NOT FOUND] Run synthetic_data.py first!")

    print("\ntorch_dataset_wrappers.py self-test complete ✓")
