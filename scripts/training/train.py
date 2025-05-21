"""
train.py
-------------------------------------------------

Training utility that wraps data loaders and RunnerDNN to execute training on a dataset.

Handles device placement, batch-size adjustment, and invokes the RunnerDNN pipeline.
"""

import torch
from torch.utils.data import DataLoader

from utils.config import RIDGES
from runners.RunnerDNN import RunnerDNN

from utils.data_transforms import update_dataloader_batch_size


def train_on_data(
    train_loader: DataLoader,
    test_loader: DataLoader,
    folder: str,
    name: str,
    depth: int = 1,
    width: int = 32,
    z_kare: float = 0.1,
    adaptative_kare: bool = False,
    lr: float = 1.0,
    krr_ridge_penalty: list[float] | None = None,
    epochs: int = 200,
    seed: int = 0,
    optimizer: str = 'SGD',
    batch_size: int | None = None,
    scaler = None,
    model_name: str = 'SimpleNN',
    parameter_specific_lr: bool = False,
    outdim_nngp: int = 10
) -> None:
    
    """
    Prepare data loaders, configure RunnerDNN, and execute training.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for training data (expects `.dataset.tensors`).
    test_loader : DataLoader
        DataLoader for test data.
    folder : str
        Directory to save outputs and checkpoints.
    name : str
        Run identifier (e.g., 'KARE', 'MSE').
    depth : int, default=1
        Number of hidden layers.
    width : int, default=32
        Width of hidden layers.
    z_kare : float, default=0.1
        Ridge penalty for KARE loss.
    adaptative_kare : bool, default=False
        Whether to use adaptive ridge in KARELoss.
    lr : float, default=1.0
        Learning rate.
    krr_ridge_penalty : list of float, optional
        Grid of ridge penalties for kernel regression; defaults to RIDGES.
    epochs : int, default=200
        Number of training epochs.
    seed : int, default=0
        Random seed for reproducibility.
    optimizer : str, default='SGD'
        Optimizer type ('SGD' or 'Adam').
    batch_size : int or None, default=None
        If provided, updates `train_loader` batch size via sampler.
    scaler : object, optional
        Target scaler for inverse transforms (e.g., StandardScaler).
    model_name : str, default='SimpleNN'
        Architecture name for RunnerDNN.
    parameter_specific_lr : bool, default=False
        Flag for per-parameter learning rates.
    outdim_nngp : int, default=10
        Output dimension for NNGPNN when used.

    Returns
    -------
    None
    """
    
    # Default ridge penalties
    if krr_ridge_penalty is None:
        krr_ridge_penalty = RIDGES

    # Determine device (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set manual seed for reproducibility
    torch.manual_seed(seed)

    # Optionally adjust training batch size
    if batch_size is not None:
        train_loader = update_dataloader_batch_size(train_loader, batch_size)

    # Move dataset to right device
    train_loader.dataset.tensors = tuple(
        tensor.to(device) for tensor in train_loader.dataset.tensors
    )
    test_loader.dataset.tensors = tuple(
        tensor.to(device) for tensor in test_loader.dataset.tensors
    )

    # Instantiate the training runner 
    runner = RunnerDNN(train_loader=train_loader,
                       model_name=model_name,
                       depth=depth,
                       width=width,
                       z_kare=z_kare,
                       adaptative_kare=adaptative_kare,
                       lr=lr,
                       device=device,
                       optimizer=optimizer,
                       epochs=epochs,
                       seed=seed,
                       folder=folder,
                       name=name,
                       scaler=scaler,
                       krr_ridge_penalty=krr_ridge_penalty,
                       outdim_nngp=outdim_nngp,
                       parameter_specific_lr=parameter_specific_lr)
    
    # Execute the training loop
    runner.run(test_loader=test_loader)
