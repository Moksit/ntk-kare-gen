"""
data_transforms.py


--------------------------------------------------------------
Utilities for wrapping NumPy arrays into PyTorch DataLoaders and
updating DataLoader batch sizes.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def wrap_data_in_dataloader(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int | None = None
) -> tuple[DataLoader, DataLoader]:
    """
    Wrap NumPy arrays into PyTorch DataLoaders.

    Converts input arrays to float32 tensors and constructs DataLoaders
    with the specified batch size (full-batch by default).

    Parameters
    ----------
    X_train : np.ndarray
        Training input data of shape (N_train, d).
    y_train : np.ndarray
        Training targets of shape (N_train, ...).
    X_test : np.ndarray
        Test input data of shape (N_test, d).
    y_test : np.ndarray
        Test targets of shape (N_test, ...).
    batch_size : int or None, default=None
        Batch size for training loader. If None, use full dataset size.

    Returns
    -------
    train_loader, test_loader : tuple of DataLoader
        DataLoaders for training and test sets.
    """
    
    #Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Determine batch sizes (full-batch if not specified)
    if batch_size is None:
        batch_size_train = x_train_tensor.shape[0]
    else:
        batch_size_train = batch_size
    batch_size_test = x_test_tensor.shape[0]

    # Create TensorDatasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Construct DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=False
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False
        )

    return train_loader, test_loader


def update_dataloader_batch_size(
    dataloader: DataLoader,
    new_batch_size: int
    ) -> DataLoader:
    """
    Create a new DataLoader with an updated batch size from an existing one.

    Parameters
    ----------
    dataloader : DataLoader
        Original DataLoader instance.
    new_batch_size : int
        Desired batch size for the new DataLoader.

    Returns
    -------
    DataLoader
        New DataLoader with the same dataset and settings but updated batch size.

    Raises
    ------
    ValueError
        If the original DataLoader uses a custom batch_sampler that cannot be updated.
    """
    
    # Reuse dataset and loader settings
    dataset = dataloader.dataset
    num_workers = dataloader.num_workers
    pin_memory = dataloader.pin_memory
    drop_last = dataloader.drop_last
    collate_fn = dataloader.collate_fn

    # Construct new DataLoader with updated batch_size
    new_loader = DataLoader(
        dataset,
        batch_size=new_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return new_loader