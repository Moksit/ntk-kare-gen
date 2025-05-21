"""
checkpoint.py
======================

Utility callback that serializes predictions and targets at various
check-points during training/evaluation of neural-tangent-kernel (NTK),
kernel ridge-regression (KARE) and baseline models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from loss.KARELoss  import KARELoss
from models.NTKNN   import NTKNN
from utils.config   import RIDGES

# -----------------------------------------------------------------------------


class CheckpointCallback:
    """
    Serialize model predictions, targets and hyper-parameters at given steps.

    The same callback can be reused for different *modes* of an experiment
    (e.g. **KARE**, **MSE**, **SVM**) by passing the desired ``name`` when
    instantiating the object.

    Parameters
    ----------
    folder : str | Path
        Root directory in which sub-folders will be created.
    train_loader, test_loader : DataLoader
        PyTorch dataloaders used during training / evaluation.
        Their `.dataset.tensors` must be two-tuples ``(X, y)``.
    name : {"KARE", "MSE"}
        Experiment type that decides which artefacts are written.
    to_str : str
        Extra tag inserted in the generated path, typically describing the run.
    parameter_dict : dict
        Hyper-parameters to be dumped to *parameters.json* for full
        reproducibility.
    scaler : Optional[sklearn.base.BaseEstimator], default=None
        Scaler to invert transform the targets (e.g. `StandardScaler`).
    """

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        folder: str | Path,
        train_loader: DataLoader,
        test_loader: DataLoader,
        name: str,
        to_str: str,
        parameter_dict: Dict[str, Any],
        scaler: Optional[Any] = None,
        krr_ridge_penalty=None,
    ) -> None:
        
        #Store base directors for outputs
        self.folder = Path(folder)
        
        #Dataloaders for trianing and teting
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        #Normalize experiment name
        self.name = name.upper()
        self.to_str = to_str
        
        #Create a parameter dictionnary for reproducibility
        self.parameter_dict = parameter_dict
        self.scaler = scaler
        
        #If not ridge penalty is given, use the default one
        if krr_ridge_penalty is None:
            krr_ridge_penalty = RIDGES
        self.krr_ridge_penalty = krr_ridge_penalty
        
        #Detect ROCm backend
        self.is_rocm = torch.version.hip is not None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def save_checkpoint(self, model: NTKNN, step: str) -> None:
        """
        Run predictions for **all** configured models and dump artefacts.

        Parameters
        ----------
        model : SimpleNN
            Trained PyTorch network implementing ``compute_neural_tangent_kernel``.
        step : str
            Free-form identifier appended to the predictions file,
            e.g. `"epoch10"`, `"best"` …
        """
        
        # 1. ----------------------------------------------------------------
        # Compute NTK (or its gradient) once – reused by multiple predictors
        # -------------------------------------------------------------------
        K_test, K_train = model.compute_kernel(
            train=self.train_loader.dataset.tensors[0],
            test=self.test_loader.dataset.tensors[0],
            return_gradient=False,
        )

        # -------------------------------------------------------------------
        # 2. Kernel ridge-regression over a grid of ridge penalties
        # -------------------------------------------------------------------
        if self.name == "KARE":
            y_train = (
                self.train_loader.dataset.tensors[1]
                if not self.is_rocm
                else self.train_loader.dataset.tensors[1].cpu()
            )

        elif self.train_loader.dataset.tensors[1].max().item() > 1:
            num_cls = int(self.train_loader.dataset.tensors[1].max().item()) + 1
            y_cpu = self.train_loader.dataset.tensors[1].long()
            y_train = (
                torch.nn.functional.one_hot(y_cpu, num_classes=num_cls)
                if not self.is_rocm
                else torch.nn.functional.one_hot(y_cpu, num_classes=num_cls).cpu()
            )
        else:
            y_train = (
                self.train_loader.dataset.tensors[1]
                if not self.is_rocm
                else self.train_loader.dataset.tensors[1].cpu()
            )      
                 
        pred_kernel_grid, probabilities = model.compute_kernel_regression(
            x_train=(
                self.train_loader.dataset.tensors[0]
                if not self.is_rocm
                else self.train_loader.dataset.tensors[0].cpu()
            ),
            x_test=(
                self.test_loader.dataset.tensors[0]
                if not self.is_rocm
                else self.test_loader.dataset.tensors[0].cpu()
            ),
            y_train=y_train,
            ridge_penalty_grid=self.krr_ridge_penalty,
            k_train=K_train.detach() if not self.is_rocm else K_train.detach().cpu(),
            k_test=K_test.detach() if not self.is_rocm else K_test.detach().cpu(),
        )
        # → shape: (n_test, len(RIDGES))
        pred_kernel_grid = np.column_stack(  # stack list of tensors → ndarray
            [p.cpu().detach().numpy().ravel() for p in pred_kernel_grid]
        )
        
        # -------------------------------------------------------------------
        # 3. Derive *targets* and *naïve* (mean / majority) baseline
        # -------------------------------------------------------------------
        targets = self._extract_targets(self.test_loader)
        naive_pred = self._compute_naive_baseline(self.train_loader)

        # -------------------------------------------------------------------
        # 4. Dump artefacts according to the experiment *name*
        # -------------------------------------------------------------------
        # Always keep KARE results (KARE mode) or NTK (MSE mode)
        self._save_results_kernel(pred_kernel_grid, targets, step, probabilities, model)

        if self.name == "MSE" or ("BestNN" in self.to_str):
            # ────────────────────────────────────────────────────────────
            # Forward pass through the neural network itself
            # ────────────────────────────────────────────────────────────
            pred_nn = model(self.test_loader.dataset.tensors[0])
            
            if pred_nn.ndim > 1:
                if pred_nn.shape[1] > 1:# one-hot → class index
                    pred_nn = torch.argmax(pred_nn, dim=1).unsqueeze(1)
                    

            self._save_results_nn(pred_nn.cpu().detach().numpy(), targets, step)

        # Naïve baseline is always helpful for orientation
        self._save_results_naive(naive_pred, step)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _extract_targets(self, loader: DataLoader) -> np.ndarray:
        """
        Return ground-truth targets as 1-D array (optionally de-scaled).

        Parameters
        ----------
        loader : DataLoader
            DataLoader providing `.dataset.tensors[1]` as labels.

        Returns
        -------
        np.ndarray
            Array of true target values or class indices.
        """
        y = loader.dataset.tensors[1].cpu().detach().numpy()
        if y.ndim > 1:  # one-hot → class index
            y = np.argmax(y, axis=1)
        if self.scaler is not None:
            y = self.scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        return y

    @staticmethod
    def _compute_naive_baseline(loader: DataLoader) -> float | int:
        
        """
        Compute naive baseline prediction: mean for regression or majority class for classification.

        Parameters
        ----------
        loader : DataLoader
            DataLoader providing `.dataset.tensors[1]` as labels.

        Returns
        -------
        float or int
            Baseline prediction value.
        """
        
        y = loader.dataset.tensors[1].cpu().detach().numpy()
        if y.ndim == 1:  # regression
            return float(np.mean(y))
        # classification
        return int(np.argmax(np.sum(y, axis=0)))

    # --------------------------------------------------------------------- #
    # Writers
    # --------------------------------------------------------------------- #

    def _dump_parameters(self, path: Path, model: NTKNN = None) -> None:
        """
        Write parameters.json alongside predictions.

        Parameters
        ----------
        path : Path
            Directory where parameters.json will be saved.
        model : NTKNN, optional
            Model used to compute final KARE loss (not saved by default).
        """
        
        with (path / "parameters.json").open("w") as fp:

            if model is not None:
                
                #This is unused for now
                #kare = KARELoss(lambda_reg=self.parameter_dict["z_kare"])

                #final_train_kare = (
                #    kare(
                #        model,
                #        self.train_loader.dataset.tensors[0],
                #        self.train_loader.dataset.tensors[1],
                #    )[0]
                #    .detach()
                #    .cpu()
                #    .numpy()
                #    .tolist()
                #)
                

                self.parameter_dict["final_train_kare"] = 0

            json.dump(self.parameter_dict, fp, indent=4)

    # .......................   K E R N E L   ...............................

    def _save_results_kernel(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        step: str,
        probabilities: dict = {},
        model: NTKNN = None,
    ) -> None:
        
        """
        Save kernel-based predictions, targets and parameters.

        Parameters
        ----------
        predictions : np.ndarray
            Kernel regression predictions for each test sample and penalty.
        targets : np.ndarray
            True target values or class indices.
        step : str
            Identifier appended to the filenames.
        probabilities : dict, optional
            Class probabilities from kernel regression.
        model : NTKNN, optional
            Model instance for dumping additional metadata.
        """
        
        #Create the directory for the results
        save_dir = self.folder / self.to_str.replace("MSE", "MSENTK")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        #Save predictions (will be the class index for multiclass and the probability for binary classification)
        np.save(save_dir / f"predictions{step}.npy", predictions, allow_pickle=True)
        np.save(save_dir / "targets.npy", targets, allow_pickle=True)
        torch.save(probabilities, os.path.join(save_dir, f"probabilities{step}.pt"))

        #Save the parameters
        self._dump_parameters(save_dir, model)

    # .......................   N E U R A L   ...............................

    def _save_results_nn(
        self, predictions: np.ndarray, targets: np.ndarray, step: str
    ) -> None:
        """
        Save neural network predictions and targets.

        Parameters
        ----------
        predictions : np.ndarray
            Raw NN predictions (class or regression outputs).
        targets : np.ndarray
            True target values or class indices.
        step : str
            Identifier appended to the filenames.
        """
        
        #Create the directory for the results
        save_dir = self.folder / self.to_str.replace("MSE", "NN")
        save_dir.mkdir(parents=True, exist_ok=True)

        #Save the predictions (will be the class index for multiclass and the probability for binary classification)
        np.save(save_dir / f"predictions{step}.npy", predictions, allow_pickle=True)
        np.save(save_dir / "targets.npy", targets, allow_pickle=True)
        
        #Save the parameters
        self._dump_parameters(save_dir)

    # .......................   B A S E L I N E   ...........................

    def _save_results_naive(self, prediction: float | int, step: str) -> None:
        """
        Save naive baseline prediction (mean or majority class).

        Parameters
        ----------
        prediction : float or int
            Baseline value to save.
        step : str
            Identifier appended to the filename.
        """
        #Create the directory for the results
        save_dir = self.folder / self.to_str.replace("MSE", "NAIVE").replace(
            "KARE", "NAIVE"
        ).replace("NN", "NAIVE")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        #Save the predictions, will be a single constant value
        np.save(save_dir / f"predictions{step}.npy", prediction, allow_pickle=True)
