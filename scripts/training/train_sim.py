"""
train_sim.py
-------------------------------------------------------
This script trains a neural network using KARE and MSE loss functions on a synthetic dataset.
"""
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

from loss.KARELoss import KARELoss
from models.NTKNN import NTKNN
from utils.datasets import create_train_test_dataloaders_cos


def aggregate_scores(
    dict_list : List[Dict[Any, Tuple[List[float], List[float], float]]]
    ) -> Dict[Any, List[Any]]:
    """
    Aggregate multiple experiment result dictionaries by averaging lists and scalars.

    Parameters
    ----------
    results : list of dict
        Each dict maps a key to a tuple (list1, list2, val3).

    Returns
    -------
    dict
        Maps each key to [mean_list1, mean_list2, mean_val3].
    """
    
    # Initialize accumulators for sums and counts
    acc = defaultdict(lambda: {"sum1": None, "sum2": None, "sum3": 0.0, "count": 0})

    # Accumulate sums
    for d in dict_list:
        for key, (lst1, lst2, val3) in d.items():
            entry = acc[key]
            # initialize sum arrays on first encounter
            if entry["sum1"] is None:
                entry["sum1"] = [0.0] * len(lst1)
                entry["sum2"] = [0.0] * len(lst2)
            # accumulate element‐wise
            for i, x in enumerate(lst1):
                entry["sum1"][i] += x
            for i, x in enumerate(lst2):
                entry["sum2"][i] += x
            entry["sum3"]   += val3
            entry["count"] += 1

    # Compute means
    result = {}
    for key, entry in acc.items():
        n = entry["count"]
        mean1 = [s / n for s in entry["sum1"]]
        mean2 = [s / n for s in entry["sum2"]]
        mean3 = entry["sum3"] / n
        result[key] = [mean1, mean2, mean3]
    return result

def main(
    scale      = 1,
    sigma      = 0,
    train_size = 1000,
    depth      = 1,
    width      = 32,
    z_kare     = 0.1,
    lr_kare    = 1,
    krr_ridge_penalty = [1e-3, 1e-3, 1e-1, 1, 10, 100, 1000],
    kare_epochs = 200,
    mse_lr = 0.1,
    mse_epochs = 10000,
    d = 10,
    seed = 0
    ) -> Tuple[List[float], List[float], float]: 
    
    """
    Train NTKNN models under KARE and MSE objectives and evaluate MSE performance.

    Parameters
    ----------
    scale : float
        Scaling for synthetic features.
    sigma : float
        Noise standard deviation.
    train_size : int
        Number of training samples.
    depth : int
        Number of hidden layers.
    width : int
        Hidden layer dimension.
    z_kare : float
        Ridge penalty for KARE loss.
    lr_kare : float
        SGD learning rate for KARE training.
    krr_ridge_penalty : list of float
        Grid of ridge penalties for kernel regression.
    kare_epochs : int
        Number of epochs for KARE training.
    mse_lr : float
        Learning rate for MSE training.
    mse_epochs : int
        Number of epochs for MSE training.
    d : int
        Feature dimension for synthetic data.
    seed : int
        RNG seed.

    Returns
    -------
    mse_kare_grid : list of float
        MSE of kernel predictions after KARE training.
    mse_kernel_mse_grid : list of float
        MSE of kernel predictions after MSE-trained kernel.
    mse_nn_mse : float
        MSE of direct NN predictions after MSE training.
    """
    
    #Set device and seed
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    #Load the data
    train_loader, test_loader = create_train_test_dataloaders_cos(
        d = d, n = train_size, scale = scale, sigma = sigma, seed = seed
        )
    
    #Move dataset to right device
    train_loader.dataset.tensors = tuple(tensor.to(device) for tensor in train_loader.dataset.tensors)
    test_loader.dataset.tensors = tuple(tensor.to(device) for tensor in test_loader.dataset.tensors)
    
    # --- KARE training ---
    model_kare = NTKNN(
        in_features=train_loader.dataset.tensors[0].shape[1],
        num_hidden_layers = depth,
        hidden_dim=width,
        out_features = 1
    ).to(device)
    kare_criterion = KARELoss(lambda_reg = z_kare).to(device)
    optimizer_kare = torch.optim.SGD(model_kare.parameters(), lr = lr_kare)
    
    #Training loop for KARE
    for epoch in range(kare_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            
            #Move the tensors to the device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_kare.zero_grad()
            
            # Forward pass for KARE
            loss,_ = kare_criterion(model_kare, X_batch, y_batch)
            
            loss.backward()
            optimizer_kare.step()
            
            epoch_loss += loss.item()
            
        
        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"[KARE] Epoch [{epoch+1}/{kare_epochs}], Avg KARE Loss = {avg_loss:.6f}")  
        
    #Make the kernel prediction with KARE trained NTK
    K_test, K_train = model_kare.compute_kernel(
        train = train_loader.dataset.tensors[0],
        test = test_loader.dataset.tensors[0]
    )
    
    preds_kare,_ = model_kare.compute_kernel_regression(
        x_train = train_loader.dataset.tensors[0],
        x_test = test_loader.dataset.tensors[0],
        y_train = train_loader.dataset.tensors[1],
        ridge_penalty_grid = krr_ridge_penalty,
        k_train = K_train,
        k_test = K_test
        )

    #Compute the MSE on prediction   
    mse_kare_grid = [
        float(torch.mean((p - test_loader.dataset.tensors[1])**2).cpu())
        for p in preds_kare
    ]
        
    # --- MSE training ---
    model_mse = NTKNN(
        in_features=train_loader.dataset.tensors[0].shape[1],
        num_hidden_layers = depth,
        hidden_dim=width,
        out_features = 1
    ).to(device)
    
    mse_criterion = nn.MSELoss().to(device)
    optimizer_mse = torch.optim.SGD(model_mse.parameters(), lr=mse_lr)
    
    #Training loop for MSE
    for epoch in range(mse_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            
            #Move the tensors to the device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_mse.zero_grad()
            
            # Forward pass for MSE
            preds = model_mse(X_batch).view(-1,1)  # shape (batch_size,)
            loss = mse_criterion(preds, y_batch)
            loss.backward()
            optimizer_mse.step()
            
            epoch_loss += loss.item()
            
        
        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch+1) % 1000 == 0:
            print(f"[MSE]  Epoch [{epoch+1}/{mse_epochs}], Avg MSE Loss = {avg_loss:.6f}")
            
            
    #after-NTK kernel ridge regression
    K_test, K_train = model_mse.compute_kernel(
        train = train_loader.dataset.tensors[0],
        test = test_loader.dataset.tensors[0]
    )
    
    preds_mse_kernel,_ = model_mse.compute_kernel_regression(
        x_train = train_loader.dataset.tensors[0],
        x_test = test_loader.dataset.tensors[0],
        y_train = train_loader.dataset.tensors[1],
        ridge_penalty_grid = krr_ridge_penalty,
        k_train = K_train,
        k_test = K_test
        )

    mse_kernel_mse_grid = [
        float(torch.mean((p - test_loader.dataset.tensors[1])**2).cpu())
        for p in preds_mse_kernel
    ]
    
    # Direct NN MSE
    nn_preds = model_mse(test_loader.dataset.tensors[0])
    mse_nn_mse = float(torch.mean((nn_preds - test_loader.dataset.tensors[1])**2).cpu())
    
    
    return mse_kare_grid, mse_kernel_mse_grid, mse_nn_mse

if __name__ == "__main__":
    
    import json
    
    # Set the parameters for the experiment
    TRAIN_SIZE = 1000
    BATCH_SIZE = 1000
    DEPTH = 1
    WIDTH = 64
    Z_KARE = 0.1
    LR_KARE = 100
    KRR_RIDGE_PENALTY = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
    KARE_EPOCHS = 100
    MSE_LR = 0.1
    MSE_EPOCHS = 10000
    SCALE_LIST = [1, 1.25, 1.5]
    SIGMA_LIST = [0, 0.1]
    SAVE_PATH = "./results/sim/"
    
    LOOP_ITERATION = 10
    
    #Repeat the experiment multiple_times and store each of the results
    results_list = []
    for i in range(LOOP_ITERATION):
        results = {}
        for scale in SCALE_LIST:
            for sigma in SIGMA_LIST:
                print(f"Training model with scale = {scale}, width = {sigma}, loop = {i+1}/{LOOP_ITERATION}")
                results[str((scale, sigma))]= main(train_size = TRAIN_SIZE,
                                            depth = DEPTH,
                                            width = WIDTH,
                                            z_kare = Z_KARE,
                                            lr_kare = LR_KARE,
                                            krr_ridge_penalty = KRR_RIDGE_PENALTY,
                                            kare_epochs = KARE_EPOCHS,
                                            mse_lr = MSE_LR,
                                            mse_epochs = MSE_EPOCHS,
                                            scale = scale,
                                            sigma = sigma,
                                            seed = i)
                
                results_list.append(results)
    
    # Aggregate the results and save them
    results = aggregate_scores(results_list)

    to_json = {"results" : results}
    to_json['params'] = {"TRAIN_SIZE" : TRAIN_SIZE,
                        "BATCH_SIZE" : BATCH_SIZE,
                        "DEPTH" : DEPTH,
                        "WIDTH" : WIDTH,
                        "Z_KARE" : Z_KARE,
                        "LR_KARE" : LR_KARE,
                        "KRR_RIDGE_PENALTY" : KRR_RIDGE_PENALTY,
                        "KARE_EPOCHS" : KARE_EPOCHS,
                        "MSE_LR" : MSE_LR,
                        "MSE_EPOCHS" : MSE_EPOCHS,
                        "SCALE_LIST" : SCALE_LIST,
                        "SIGMA_LIST" : SIGMA_LIST}
    
    with open(SAVE_PATH + "v1_avg10.json", "w") as f:
        json.dump(to_json, f, indent=4)