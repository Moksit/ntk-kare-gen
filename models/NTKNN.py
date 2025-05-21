"""
NTKNN.py
============

Central **SimpleNN** implementation used throughout the project.

The class provides:

* A configurable feed-forward network (`torch.nn.Sequential`).
* Utilities to compute per-sample Jacobians and the corresponding
  neural-tangent kernel (NTK).
* A cache-friendly routine to build very large NTKs in constant GPU memory.
* Closed-form kernel ridge-regression using a grid of ridge penalties.

"""

import math
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad

# ──────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────


class NTKNN(nn.Module):
    """
    Lightweight fully-connected neural network for NTK methods.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input sample.
    hidden_dim : int, default=64
        Width of hidden layers.
    out_features : int, default=1
        Output dimensionality.
    num_hidden_layers : int, default=1
        Number of non-linear hidden layers (excluding output layer).
    train_eta : bool, default=False
        If True, learnable scaling parameter for gradients.

    Attributes
    ----------
    model : nn.Sequential
        Feed-forward architecture.
    eta : nn.Parameter
        Log-scale parameters for gradient weighting.
    """

    # .....................................................................
    # Construction
    # .....................................................................

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_features: int = 1,
        num_hidden_layers: int = 1,
        train_eta: bool = False
    ) -> None:
        super().__init__()

        layers : list[nn.Module] = []

        # First hidden layer
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.GELU())

        # Additional hidden layers (if requested)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, out_features))

        self.model = nn.Sequential(*layers)
        
        #Initialize learnable gradient scaling if requested
        num_params = self.get_n_params()
        self.train_eta = train_eta
        
        #One eta per weight element
        self.eta = nn.Parameter(torch.zeros(num_params), requires_grad=train_eta)
        print(
            f"[INFO] Created a {self.__class__.__name__} with {num_params} params."
        )

    # .....................................................................
    # Forward inference
    # .....................................................................

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        
        """
        Standard forward pass through the network.

        Parameters
        ----------
        x : Tensor, shape (N, in_features)
            Input batch.

        Returns
        -------
        Tensor
            Network outputs of shape (N, out_features).
        """
        
        return self.model(x)

    # ---------------------------------------------------------------------
    # Jacobian utilities
    # ---------------------------------------------------------------------

    def gradient_matrix(self, x_: torch.Tensor) -> torch.Tensor:
        
        """
        Compute per-sample gradient matrix G of outputs w.r.t. parameters.

        For each sample x_i, G[i] = ∂f(x_i)/∂θ, flattened.

        Parameters
        ----------
        X : Tensor, shape (N, in_features)
            Input batch.

        Returns
        -------
        Tensor, shape (N, P)
            Stacked gradients, weighted by exp(eta).
        """
        
        # Mapping of {parameter name → tensor}
        params = dict(self.named_parameters())
        params.pop('eta')  # remove eta from the parameters

        # Scalar-output helper (required by `torch.func.grad`)
        def compute_output(p, x):
            # x : (d,)  →  unsqueeze to (1, d) for the network
            out = functional_call(self, p, (x.unsqueeze(0),))
            return out.sum()  # ensure scalar for grad

        # Gradient w.r.t. the *parameters* of the scalar output
        grad_fn = grad(compute_output)

        # Vectorise over the batch dimension: x ∈ X
        per_sample_grads = vmap(grad_fn, in_dims=(None, 0))(params, x_)

        # Flatten every per-parameter gradient and concatenate
        grads_flat = []
        n_samples = x_.shape[0]
        for g in per_sample_grads.values():
            grads_flat.append(g.reshape(n_samples, -1))

        grad_mat = torch.cat(grads_flat, dim=1)  # (N, P)
        grad_mat = grad_mat * torch.exp(self.eta).reshape(1, -1)
        return grad_mat

    # ---------------------------------------------------------------------
    # NTK computation
    # ---------------------------------------------------------------------
    def compute_kernel(
        self,
        train: torch.Tensor,
        test: torch.Tensor,
        return_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Compute NTK between train and test samples.

        Parameters
        ----------
        train : Tensor, shape (N_train, in_features)
        test : Tensor, shape (N_test, in_features)
        return_gradient : bool, default=False
            If True, return raw gradients (G_test, G_train) instead of Gram matrices.

        Returns
        -------
        NTK_test : Tensor
            Kernel between test and train samples.
        NTK_train : Tensor
            Kernel between train samples (self-kernel).
        """
        
        # Optimal blocking to keep the per-block gradient matrix within
        # a GPU-memory budget (empirical heuristic).
        n_blocks = self.optimal_n_block(train.shape[0], self.get_n_params(), 4)
        block_size = train.shape[0] // n_blocks

        if n_blocks != 1:
            print("[INFO] Optimal block size for NTK computation:", block_size)

        # Fast path: whole batch fits in memory
        if n_blocks == 1:
            g_train = self.gradient_matrix(train)
            g_test = self.gradient_matrix(test)
            
            if return_gradient:
                return g_test, g_train

            ntk_test = g_test @ g_train.T
            ntk_train = g_train @ g_train.T
            return ntk_test, ntk_train

        # Memory-constrained path
        ntk_train = self.compute_NTK_by_block(train, train, block_size)
        ntk_test = self.compute_NTK_by_block(test, train, block_size)
        
        return ntk_test, ntk_train

    # .....................................................................
    # Blocked NTK (memory-constant)
    # .....................................................................

    def compute_NTK_by_block(
        self,
        x_: torch.Tensor,
        y_: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """
        Memory-constant NTK computation using double blocking.

        Parameters
        ----------
        x_ : Tensor, shape (N, D)
        y_ : Tensor, shape (M, D)
        block_size : int
            Samples per block.

        Returns
        -------
        Tensor, shape (N, M)
            Kernel matrix.
        """
        device, dtype = x_.device, x_.dtype
        n_samples, m_samples = x_.size(0), y_.size(0)
        symmetric = x_.data_ptr() == y_.data_ptr()  # X and Y might alias

        # ---- Flatten parameters once ------------------------------------
        params = dict(self.named_parameters())
        items = list(params.items())
        shapes = [p.shape for _, p in items]
        numels = [p.numel() for _, p in items]
        names = [n for n, _ in items]
        flat_params = torch.cat([p.reshape(-1) for _, p in items])
        p = flat_params.numel()

        def unflatten(flat):
            chunks = torch.split(flat, numels)
            return {n: c.view(s) for c, n, s in zip(chunks, names, shapes)}

        # Scalar function f(θ, x) for functorch
        def f_flat(flat_p, x):
            return functional_call(self, unflatten(flat_p), (x.unsqueeze(0),)).sum()

        grad_flat = grad(f_flat)
        vmap_grad = vmap(grad_flat, in_dims=(None, 0))

        # ---- Allocate result + scratch buffers --------------------------
        kernel = torch.empty((n_samples, m_samples), device=device, dtype=dtype)

        pad_buf = torch.zeros((block_size, *x_.shape[1:]), device=device, dtype=dtype)
        gi_full = torch.empty((block_size, p), device=device, dtype=dtype)
        gj_full = torch.empty_like(gi_full)

        # Warm-up: compile kernels once
        _ = vmap_grad(flat_params, pad_buf)

        # ---- Blocked double loop ----------------------------------------
        for i in range(0, n_samples, block_size):
            bi = min(block_size, n_samples - i)

            # Prepare X-block (pad to full block_size)
            pad_buf[:bi].copy_(x_[i : i + bi])
            if bi < block_size:
                pad_buf[bi:] = 0
            gi_full[:bi].copy_(vmap_grad(flat_params, pad_buf)[:bi])

            for j in range(0, m_samples, block_size):
                bj = min(block_size, m_samples - j)

                # Prepare Y-block
                pad_buf[:bj].copy_(y_[j : j + bj])
                if bj < block_size:
                    pad_buf[bj:] = 0
                gj_full[:bj].copy_(vmap_grad(flat_params, pad_buf)[:bj])

                # Kernel block product
                kernel[i : i + bi, j : j + bj] = gi_full[:bi] @ gj_full[:bj].T
                # Fill symmetric counterpart if applicable
                if symmetric and j < i:
                    kernel[j : j + bj, i : i + bi] = kernel[i : i + bi, j : j + bj].T

            print(
                f"[INFO] Completed block {i // block_size + 1} / "
                f"{math.ceil(n_samples / block_size)}"
            )
            gc.collect()  # tidy Python refs

        return kernel

    # ---------------------------------------------------------------------
    # Kernel ridge-regression (closed form)
    # ---------------------------------------------------------------------

    def compute_kernel_regression(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        y_train: torch.Tensor,
        ridge_penalty_grid: list[float],
        k_train: torch.Tensor,
        k_test: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Closed-form kernel ridge regression over grid of penalties.

        Parameters
        ----------
        X_train : Tensor
            Training data (unused beyond shape checks).
        X_test : Tensor
            Test data.
        y_train : Tensor
            Training targets.
        ridge_penalty_grid : Sequence of floats
            Multiplicative factors for ridge hyperparameter.
        K_train : Tensor
            Kernel matrix on training set.
        K_test : Tensor
            Kernel matrix between test and train.

        Returns
        -------
        predictions : list of Tensors
            Predictions per ridge value.
        probabilities : dict
            Softmax probabilities for classification.
        """
        
        n = x_train.shape[0]
        
        # Promote to double for numerical stability when memory enables it
        if n < 50000:
            k_train = k_train.double()
            k_test = k_test.double()
            y_train = y_train.double()

        # Scale-invariant ridge: λ = α·Tr(K) / N²
        normalizer = torch.trace(k_train)
        effective_ridges = [alpha * normalizer / (n**2) for alpha in ridge_penalty_grid]

        # Ensure column vector (for regression) or keep one-hot (classification)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)


        #Sometimes torch.linalg.eigh is unstable, in that case we will switch to using cholesky decomposition
        try:
            
            # Eigen-decomposition of K_train and project the targets
            eigenvalues, eigenvectors = torch.linalg.eigh(k_train)
            y_tilde = eigenvectors.T @ y_train

            predictions = []
            probabilities = {}
            
            #We loop over ridge penalties but the inverse is fast because the matrix is diagonal
            for ridge in effective_ridges:
                coeff = 1 / (eigenvalues / n + ridge)  # (N,)
                beta = coeff.reshape(-1, 1) * y_tilde  # (N, c)
                preds = (1 / n) * k_test @ eigenvectors @ beta  # (N_test, c)
                
                #When we are doing a multi-class classification we will use softmax to get the probabilities and argmax to get the predictions
                probs = []
                if y_train.shape[1] > 1:
                    probs = torch.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                predictions.append(preds)
                probabilities[ridge] = probs

        #This is the case where eigh is unstable
        except RuntimeError:
            
            predictions = []
            probabilities = {}

            #We loop over ridges penalties but this time the predictions is slow
            for ridge in effective_ridges:
                
                id_n = torch.eye(n, device=k_train.device, dtype=k_train.dtype)
                
                #When we have less than 50k samples we can compute the predictions on a single GPU
                if n < 50000:
                    preds = 1/n * k_test @ torch.linalg.solve(1/n *k_train + ridge * id_n, id_n)@y_train
                    
                #When the sample size is larger than 50k we will use two GPUs to compute the predictions
                else:
                    device = torch.device('cuda:1')

                    # build A on cuda:1
                    a_mat = k_train.to(device).div(n).add_( ridge.to(device).view(1,1) * id_n.to(device))  # shape (N, N), lives on cuda:1

                    # solve A α = y_train  →  α
                    alpha = torch.linalg.solve(a_mat, y_train.to(device))   # shape (N, …), lives on cuda:1

                    # move the small α over to whatever device K_test is on
                    alpha = alpha.to(k_test.device)

                    # finally do your prediction
                    preds = k_test @ alpha
                    preds = preds.div_(n)
                    
                probs = []
                if y_train.shape[1] > 1:
                    probs = torch.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                predictions.append(preds)
                probabilities[ridge] = probs
                
        return predictions, probabilities

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def get_n_params(self) -> int:
        """
        Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def optimal_n_block(self,
                        num_samples: int,
                        num_params: int,
                        bytes_per_element: int,
                        max_block_size_gib: float = 5.0) -> int:
        """
        Compute wether G will fit in memory or not, if this is not the case, it provides a number of block we should use.

        Parameters
        ----------
        num_samples : int
            Number of data samples (N).
        num_params : int
            Number of parameters (P).
        bytes_per_element : int
            Bytes per tensor element (e.g., 4 for float32).
        max_block_size_gib : float, default=5.0
            Max block footprint in GiB.

        Returns
        -------
        int
            Number of blocks (≥1).
        """
        
        # total bytes we need to process (×2 for forward+backward, if that’s your intent)
        total_bytes = num_samples * num_params * bytes_per_element * 2

        # convert GiB → bytes
        threshold_bytes = max_block_size_gib * (1024**3)

        # compute how many chunks of size ≤ threshold_bytes we need
        blocks = math.ceil(total_bytes / threshold_bytes)

        return max(1, blocks)