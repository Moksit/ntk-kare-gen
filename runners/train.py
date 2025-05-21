"""
train.py
==============

Utility helpers shared by all runners:

* ``mse`` – quick NumPy implementation of mean-squared error.
* ``set_seed`` – one-shot RNG seeding across Python `random`, NumPy and
  PyTorch (CPU + GPU).
* ``train_model`` – minimal training loop that understands both the custom
  :class:`loss.KARELoss.KARELoss` objective **and** any built-in PyTorch loss.

⚠️  **Behaviour is unchanged** – only docstrings and inline comments were
added to clarify intent.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
from loss.KARELoss import KARELoss


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #
def mse(y_true, y_pred):
    """
    Mean-squared error in NumPy.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth targets, shape ``(N,)`` or ``(N, 1)``.
    y_pred : ndarray
        Predicted values, broadcastable to ``y_true``.  A trailing singleton
        dim is flattened internally so the function works with model outputs
        of shape ``(N, 1)``.

    Returns
    -------
    ndarray
        Scalar (float) for a single target or 1-D array for multi-target.
    """
    return ((y_true - y_pred.reshape(-1, 1)) ** 2).mean(0)


# --------------------------------------------------------------------- #
# Reproducibility helper
# --------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    """
    Seed Python/NumPy/PyTorch for deterministic experiments.

    Notes
    -----
    * Sets ``torch.backends.cudnn.deterministic = True`` and
      ``benchmark = False`` so CUDA kernels become repeatable.
    * When multiple GPUs are available, the seed is broadcast to *all* of
      them via :pyfunc:`torch.cuda.manual_seed_all`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #
def train_model(
    epochs: int = 100,
    device: str = "cpu",  # "cuda",
    train_loader=None,
    model=None,
    optimizer=None,
    criterion=None,
    kare: str = "KARE",
    checkpoint=None,
    checkpoint_freq: float = 1.1,
):
    """
    One-file training loop that supports both **KARELoss** and standard
    PyTorch loss functions.

    Parameters
    ----------
    epochs : int, default=100
        Number of gradient-descent iterations over the entire training set.
    device : {"cuda", "cpu"}, default="cuda"
        Device on which tensors are placed.
    train_loader : torch.utils.data.DataLoader
        Iterable yielding ``(X_batch, y_batch)`` pairs.
    model : torch.nn.Module
        Model to be trained *in place*.
    optimizer : torch.optim.Optimizer
        Optimiser already bound to ``model.parameters()``.
    criterion : nn.Module
        Either :class:`loss.KARELoss.KARELoss` **or** any reduction='mean'
        PyTorch loss (e.g. ``nn.MSELoss``).
    kare : str, default="KARE"
        Prefix for console logging (used by the DNN runner).
    checkpoint : callbacks.checkpoint.CheckpointCallback or None
        When provided, intermediate predictions are saved.
    checkpoint_freq : float, default=1.1
        Fraction of ``epochs`` between two checkpoints.  Values > 1 mean
        “final checkpoint only”.

    Returns
    -------
    torch.nn.Module
        The **same** model instance after training.
    """
    
    success = True
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        # ────────────────────────────────────────────────────────────────
        # Mini-batch loop
        # ────────────────────────────────────────────────────────────────
        epoch_start = time.time()
        for X_batch, y_batch in train_loader:
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Forward + loss
            if isinstance(criterion, KARELoss):
                # KARE needs full model, inputs and targets
                loss, success = criterion(model, X_batch, y_batch)
            else:
                # Standard PyTorch loss: f(x) first
                preds = (
                    model(X_batch) if len(X_batch) == 1 else model(X_batch).squeeze()
                )
                loss = criterion(preds, y_batch)
                
            if not success:
                logging.info("KARE Loss failed, breaking the training at current step.")
                break

            # Back-prop + parameter update
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()
            
        if (isinstance(criterion, KARELoss) and epoch_loss < 0) or not success:
            print("Negative kare loss, breaking training.")
            break
        
        if torch.isnan(loss):
            logging.info("Loss is NaN, breaking the training at current step.")
            return model
            break
        
        

        avg_loss = epoch_loss / len(train_loader)
        epoch_end = time.time()
        #logging.info("Epoch time: {:.2f} seconds".format(epoch_end - epoch_start))
        # ────────────────────────────────────────────────────────────────
        # Console logging (every 10 epochs)
        # ────────────────────────────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            tag = "KARE Loss" if isinstance(criterion, KARELoss) else "Loss"
            logging.info(
                f"{kare} Epoch [{epoch + 1}/{epochs}], Avg {tag} = {avg_loss:.6f}"
            )
            
        torch.cuda.synchronize()

        # ────────────────────────────────────────────────────────────────
        # Optional checkpointing
        # ────────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            if checkpoint is not None and ((epoch + 1) % (1.1 * epochs) == 0):
                # The step identifier is a percentage string, e.g. "25", "50", …
                checkpoint.save_checkpoint(model, str(int((epoch + 1) / epochs * 100)))

    # Final save after training concludes
    if checkpoint is not None:
        checkpoint.save_checkpoint(model, "")

    return model
