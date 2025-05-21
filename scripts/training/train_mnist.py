"""
train_mnist.py
-------------------------------------------------

Script to run MNIST experiments with MSE or KARE objectives using RunnerDNN.

Trains models over a parameter grid and saves predictions and metrics.
"""

import argparse
import torch
import numpy as np

from sklearn.model_selection import ParameterGrid

from runners.RunnerDNN import RunnerDNN
from utils.datasets import load_mnist_data

def run_train(
    train_size: int = 1000,
    test_size: int | None = None,
    batch_size: int = 1000,
    depth: int = 1,
    width: int = 32,
    z_kare: float = 0.1,
    lr: dict = None,
    seed: int = 0,
    optimizer : str = 'SGD',
    epochs: dict = None,
    name: str = 'MSE',
    folder = 'results/mnist'
)-> None:
    """
    Train and evaluate models on filtered MNIST (digits 7/9) and save results.

    Parameters
    ----------
    train_size : int
        Number of training samples.
    test_size : int or None
        Number of test samples; None uses full set.
    batch_size : int
        Batch size for DataLoader.
    depth : int
        Number of hidden layers.
    width : int
        Hidden layer width.
    z_kare : float
        KARE ridge penalty.
    lr : dict
        Learning rates mapping by model name and depth.
    seed : int
        RNG seed for reproducibility.
    optimizer : str
        Optimizer type ('SGD' or 'Adam').
    epochs : dict
        Number of epochs mapping by model name.
    name : str
        Training objective ('MSE' or 'KARE').
    folder : str
        Output directory for checkpoint files.
    """
    # Ensure default parameters are set
    if lr is None:
        lr = {'MSE': {1: 0.1, 2: 0.1, 4: 0.1}, 'KARE': {1: 10, 2: 10, 4: 1}}
    if epochs is None:
        epochs = {'MSE': 10000, 'KARE': 300}

    # Set the device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load MNIST filtered to digits 7 and 9
    train_loader, test_loader = load_mnist_data(
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
        name=name)
    
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
    
    #Create the hyperparameters grid
    train_size = 1000
    test_size = 1000
    batch_size = 1000
    depth_list = [2,4]
    width_list = [32,64,128]
    z_kare = 0.1
    lr = {'MSE' : {1: 0.1, 2: 0.1, 4:0.1},
          'KARE' : {1 : 10, 2: 10, 4: 1}}
    epochs = {'MSE' : 10000,
              'KARE' : 300}
    folder = 'results/mnist'

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

    # Run the train-evaluation loop
    for parameters in all_parameters:
        run_train(**parameters)

if __name__ == "__main__":
    main()
    