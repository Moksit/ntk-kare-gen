"""
RunnerDNN.py
=============

Orchestrates end-to-end training of SimpleNN or NNGPNN with KARE or MSE/CE.

Builds model, loss, optimizer, and delegates training to `train_model`, wiring
in a CheckpointCallback to save predictions and metadata.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from utils.config import RIDGES
from loss.KARELoss import KARELoss
from models.NTKNN import NTKNN
from runners.train import train_model, set_seed
from callbacks.checkpoint import CheckpointCallback


# -----------------------------------------------------------------------------


class RunnerDNN:
    """
    Helper to wire model → loss → optimizer → training loop.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader containing the training set (expects `.dataset.tensors`).
    model_name : str, default='SimpleNN'
        Architecture name to instantiate (currently 'SimpleNN').
    depth : int, default=2
        Number of hidden layers.
    width : int, default=64
        Units per hidden layer.
    z_kare : float, default=0.1
        Ridge penalty for KARELoss (ignored if name != 'KARE').
    adaptative_kare : bool, default=False
        If True, KARELoss uses adaptive ridge λ = Tr(K) / N².
    lr : float, default=1e-2
        Learning rate.
    device : str, default='cpu'
        PyTorch device ('cpu' or 'cuda').
    optimizer : str, default='SGD'
        Optimizer type ('SGD' or 'Adam').
    epochs : int, default=100
        Number of training epochs.
    seed : int, default=42
        Random seed for reproducibility.
    folder : str, default=''
        Root directory for checkpoint outputs.
    name : str, default='KARE'
        Training objective ('KARE', 'MSE', or 'KARE-NNGP').
    outdim_nngp : int, default=10
        Output dimension for NNGPNN model.
    scaler : optional
        Scaler to inverse-transform targets before saving.
    krr_ridge_penalty : Sequence[float], optional
        Ridge penalties grid for kernel ridge regression.
    parameter_specific_lr : bool, default=False
        Flag for per-parameter learning rates (unused by default).
    """

    # .....................................................................
    # Construction
    # .....................................................................

    def __init__(
        self,
        train_loader,
        model_name: str = "SimpleNN",
        depth: int = 2,
        width: int = 64,
        z_kare: float = 0.1,
        adaptative_kare: bool = False,
        lr: float = 0.01,
        device: str = "cpu",
        optimizer: str = "SGD",
        epochs: int = 100,
        seed: int = 42,
        folder: str = "",
        name: str = "KARE",
        outdim_nngp : int = 10,
        scaler=None,
        krr_ridge_penalty = None,
        parameter_specific_lr: bool = False
    ):
        # Simple attribute storage
        self.folder = folder
        self.model = None
        self.name = name
        self.model_name = model_name
        self.depth = depth
        self.width = width
        self.z_kare = z_kare
        self.adaptative_kare = adaptative_kare
        self.lr = lr
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.epochs = epochs
        self.seed = seed
        self.scaler = scaler
        self.outdim_nngp = outdim_nngp
        self.parameter_specific_lr = parameter_specific_lr
        
        # Default ridge penalties if not provided
        if krr_ridge_penalty is None:
            krr_ridge_penalty = RIDGES
        self.krr_ridge_penalty = krr_ridge_penalty

        # Logger keyed by hyper-parameters
        self.logger = logging.getLogger(
            f"KARE{model_name}_{depth}_{width}_{z_kare}_{adaptative_kare}_{lr}"
        )

        # Dump configuration for reproducibility
        self.parameter_dict = {
            "model_name": model_name,
            "depth": depth,
            "width": width,
            "z_kare": z_kare,
            "adaptive_kare": adaptative_kare,
            "lr": lr,
            "optimizer": optimizer,
            "epochs": epochs,
            "seed": seed,
            "ridge": RIDGES,
            "batch_size": train_loader.batch_size,
            "scaler": scaler,
            "krr_ridge_penalty": krr_ridge_penalty,
            "parameter_specific_lr": parameter_specific_lr
        }

    # .....................................................................
    # Helpers
    # .....................................................................

    def to_str(self) -> str:
        """
        Generate a unique, filesystem-friendly run identifier.

        Returns
        -------
        str
            Identifier string based on configuration.

        Example
        -------
        'KARE_SimpleNN_2_64_0.1_False_0.01_SGD_100_32_42'
        """
        return (
            f"{self.name}_{self.model_name}_{self.depth}_{self.width}_"
            f"{self.z_kare}_{self.adaptative_kare}_{self.lr}_{self.optimizer}_"
            f"{self.epochs}_{self.train_loader.batch_size}_{self.seed}"
        )

    # .....................................................................
    # Main entry point
    # .....................................................................

    def run(
        self,
        test_loader: Optional[torch.utils.data.DataLoader] = None
        ) -> torch.nn.Module:
        """
        Build components and launch training.

        Parameters
        ----------
        test_loader : DataLoader or None, default=None
            Optional DataLoader for checkpoint predictions.

        Returns
        -------
        model : nn.Module
            Trained model instance.
        """
        
        # 0) Deterministic setup ------------------------------------------------
        set_seed(self.seed)

        # 1) Architecture -------------------------------------------------------
        outdim = (
            1
            if len(self.train_loader.dataset.tensors[1].shape) == 1
            else self.train_loader.dataset.tensors[1].shape[1]
        )


        self.model = NTKNN(
            in_features=self.train_loader.dataset.tensors[0].shape[1],
            num_hidden_layers=self.depth,
            hidden_dim=self.width,
            out_features=outdim,
        ).to(self.device)
            
        # 2) Loss function ------------------------------------------------------
        if "KARE"in self.name:
            criterion = KARELoss(
                lambda_reg=self.z_kare,
                adaptative_ridge=self.adaptative_kare
            ).to(self.device)
        else:  # "MSE"
            #classification or regresion
            if len(self.train_loader.dataset.tensors[1].shape) > 1:
                criterion = torch.nn.CrossEntropyLoss().to(self.device)
            else:
                criterion = torch.nn.MSELoss().to(self.device)
      
        # 3) Optimiser ----------------------------------------------------------
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:  # "Adam"
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
        # 4) Checkpoint callback -----------------------------------------------
        checkpoint = CheckpointCallback(
            folder=self.folder,
            train_loader=self.train_loader,
            test_loader=test_loader,
            name=self.name,
            to_str=self.to_str(),
            parameter_dict=self.parameter_dict,
            scaler=self.scaler,
            krr_ridge_penalty=self.krr_ridge_penalty,
        )

        # 5) Launch training loop ----------------------------------------------
        model = train_model(
            epochs=self.epochs,
            device=self.device,
            train_loader=self.train_loader,
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint=checkpoint,
        )

        # Optionally return the trained model for chaining
        return model
