"""PyTorch Dataset wrapper for the climate resilience dataset."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ClimateDataset(Dataset):
    """
    Wraps pre-processed numpy arrays as a PyTorch Dataset.

    Parameters
    ----------
    X : float32 ndarray of shape (N, F)
    y : float32 ndarray of shape (N, 4)
    augment : bool  — adds Gaussian feature noise during training
    augment_std : float — noise level relative to feature std
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False, augment_std: float = 0.02):
        self.X       = torch.from_numpy(X).float()
        self.y       = torch.from_numpy(y).float()
        self.augment = augment
        self.augment_std = augment_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            # Mixup-style noise on continuous features only (not binary flags)
            noise = torch.randn_like(x) * self.augment_std
            # Zero out noise on binary features (indices where values are 0 or 1)
            is_binary = ((x == 0) | (x == 1)).float()
            x = x + noise * (1 - is_binary)
        return x, y


def build_dataloaders(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val:  np.ndarray, y_test:  np.ndarray,
    batch_size: int = 512,
    num_workers: int = 0,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    import torch
    # pin_memory only works with CUDA; MPS and CPU must leave it False
    pin = torch.cuda.is_available()

    train_ds = ClimateDataset(X_train, y_train, augment=augment)
    val_ds   = ClimateDataset(X_val,   y_val,   augment=False)
    test_ds  = ClimateDataset(X_test,  y_test,  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader
