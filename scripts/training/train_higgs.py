"""
train_higgs.py
-------------------------------------------------

Script to run Higgs dataset experiments with MSE or KARE objectives using RunnerDNN.

Loads preprocessed Higgs data from CSV, trains models over a parameter grid, and saves predictions.
"""

import argparse
import torch
import numpy as np

from sklearn.model_selection import ParameterGrid

from runners.RunnerDNN import RunnerDNN
from utils.datasets import load_higgs_data_kaggle


def run_train(
    train_size: int = 1000,
    test_size: int = None,
    batch_size: int = 1000,
    depth: int = 1,
    width:int = 32,
    z_kare: float = 0.1,
    lr: dict = None,
    seed:int = 0,
    optimizer: str = 'SGD',
    epochs: dict = None,
    name: str = 'MSE',
    folder = 'results/higgs'
) -> None:
    """
    Train and evaluate models on the Higgs dataset and save results.

    Parameters
    ----------
    train_size : int
        Number of training samples to load.
    test_size : int or None
        Number of test samples; None loads full set.
    batch_size : int
        Batch size for DataLoader.
    depth : int
        Number of hidden layers for SimpleNN.
    width : int
        Hidden layer width.
    z_kare : float
        Ridge penalty for KARE loss.
    lr : dict
        Learning rates mapping by model name and depth.
    seed : int
        Random seed for reproducibility.
    optimizer : str
        Optimizer type ('SGD' or 'Adam').
    epochs : dict
        Epoch counts mapping by model name.
    name : str
        Training objective ('MSE' or 'KARE').
    folder : str
        Directory to save checkpoint outputs.

    Returns
    -------
    None
    """
    
    # Ensure defaults
    if lr is None:
        lr = {'MSE': {1: 0.1, 2: 0.1, 4: 0.1}, 'KARE': {1: 10, 2: 10, 4: 1}}
    if epochs is None:
        epochs = {'MSE': 10000, 'KARE': 300}


    # Set the device for training and seed for reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Higgs data
    train_loader, test_loader = load_higgs_data_kaggle(
        train_samples=train_size,
        batch_size=batch_size,
        test_samples=test_size,
        seed=seed
    )

    # * 1.1 Move dataset to right device
    train_loader.dataset.tensors = tuple(tensor.to(device) for tensor in train_loader.dataset.tensors)
    test_loader.dataset.tensors = tuple(tensor.to(device) for tensor in test_loader.dataset.tensors)


    runner = RunnerDNN(
        train_loader=train_loader,
        model_name='SimpleNN',
        depth=depth,
        width=width,
        z_kare=z_kare,
        adaptative_kare=False,
        lr=lr[depth],
        device=device,
        optimizer=optimizer,
        epochs=epochs,
        seed=seed,
        folder=folder,
        name=name
    )
    
    # Run the training and evaluation
    runner.run(test_loader=test_loader)
    
def main():
    """
    main function to run the training script.
    """
    
    #Usage the parser to define which model to run
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='KARE', help='Model name (KARE or MSE)')
    
    args = parser.parse_args()
    name = args.model
    
    #Define the hyperparameters grid
    train_size = 1000
    test_size = 1000
    batch_size = 1000
    depth_list = [2,4]
    width_list = [32, 64, 128]
    z_kare = 0.1
    lr = {'MSE' : {1: 0.1, 2: 0.1, 4:0.1},
          'KARE' : {1 : 10, 2: 10, 4: 1}}
    epochs = {'MSE' : 10000,
              'KARE' : 300}
    folder = 'results/higgs'

    num_seeds = 10

    all_parameters = ParameterGrid({
        'train_size': [train_size],
        'test_size': [test_size],
        'batch_size': [batch_size],
        'depth': depth_list,
        'width': width_list,
        'z_kare': [z_kare],
        'lr': [lr[name]],
        'epochs': [epochs[name]],
        'seed': list(range(num_seeds)),
        'optimizer': ['SGD'],
        'name': [name],
        'folder': [folder],
    })

    #* Run the train-evaluation loop
    for parameters in all_parameters:
        run_train(**parameters)
    
if __name__ == "__main__":
    main()

