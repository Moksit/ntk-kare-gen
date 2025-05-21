"""
KARELoss.py
============

PyTorch implementation of the Kernel Aligned Risk Estimator (KARE) loss.

Given the neural-tangent kernel K ∈ ℝ^(N×N) of a model evaluated on the
training data, the estimator reads:

    ρ =
        ( (1/N) * yᵀ * ( (1/N)K + λI )^{-2} * y )
        / ( [ (1/N) * Tr( (1/N)K + λI )^{-1} ]^2 )

The ratio is minimized when the spectrum of the NTK is most aligned with
the target signal.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# -----------------------------------------------------------------------------


class KARELoss(nn.Module):
    """
    Loss module that can replace :class:`torch.nn.MSELoss` during training.

    Parameters
    ----------
    lambda_reg : float, default=1e-2
        Ridge penalty :math:`\lambda`.
        When ``adaptive_ridge`` is *True*, this value is **ignored**.
    adaptive_ridge : bool, default=False
        If *True*, set ``lambda_reg = Tr(K) / N²`` on every forward pass
        (recommended in the original paper).
    """

    def __init__(self, lambda_reg=1e-2, adaptative_ridge=False):
        super(KARELoss, self).__init__()
        
        #Ridge penalty to compute the loss
        self.lambda_reg = lambda_reg
        
        #Whether to use the adaptive ridge penalty
        self.adaptative_ridge = adaptative_ridge

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        model: nn.Module,
        x_: Tensor,
        y_: Tensor
        ) -> tuple[Tensor, bool]: 
        
        """
        Compute the scalar KARE loss and success flag.

        Parameters
        ----------
        model : nn.Module
            Model implementing `compute_kernel(x_, x_)`.
        x_ : Tensor, shape (N, d)
            Input patterns on the same device as the model.
        y : Tensor, shape (N,) or (N, 1) or one-hot (N, n_classes)
            Targets. One-hot encoding required for multi-class labels.

        Returns
        -------
        kare : Tensor
            Zero-dimensional tensor holding the KARE value.
        success : bool
            Whether Cholesky decomposition succeeded.
        """

        # Set device and number of samples
        device = x_.device
        n_samples = x_.shape[0]
        success = True
        
        # Ensure targets are column-vectors (N, c)
        if len(y_.shape) == 1 or y_.shape[1] == 1:
            y_ = y_.view(-1, 1)

        # ------------------------------------------------------------------
        # 1) NTK (K) between *all* pairs in x_
        # ------------------------------------------------------------------
        kernel, _ = model.compute_kernel(x_, x_)

        # Optional adaptive ridge penalty λ := Tr(K) / N²
        self.lambda_reg = (
            (torch.trace(kernel) / n_samples**2) if self.adaptative_ridge else self.lambda_reg
        )
        # ------------------------------------------------------------------
        # 2) Form M = (1/N)·K + λ·I_N  and its inverse
        # ------------------------------------------------------------------
        id_n = torch.eye(n_samples, device=device, dtype=x_.dtype)
        mat = (1.0 / n_samples) * kernel + self.lambda_reg * id_n
        try:
            lmat = torch.linalg.cholesky(mat)
        except RuntimeError:
            lmat = id_n
            success = False
        mat_inv = torch.cholesky_solve(id_n, lmat)
        mat_inv2 = mat_inv @ mat_inv

        # ------------------------------------------------------------------
        # 3) ρ =  (1/N)·yᵀ M⁻² y      /      [(1/N)·Tr(M⁻¹)]²
        # ------------------------------------------------------------------
        numerator = (1.0 / n_samples) * (((mat_inv2 @ y_) * y_).mean(1).sum()).squeeze()

        tr_m_inv = torch.trace(mat_inv)
        denominator = ((1.0 / n_samples) * tr_m_inv) ** 2

        kare = numerator / denominator

        return kare, success 
