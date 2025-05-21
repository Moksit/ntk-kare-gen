"""
train_UCI.py
--------------------------------------------------

Command-line script to train neural networks on UCI repository datasets using MSE or KARE objectives.

Parses dataset metadata, loads and preprocesses data, constructs DataLoaders, and invokes the training routine.
"""

import argparse
import os

import numpy as np

from sklearn.preprocessing import RobustScaler

from scripts.training.train import train_on_data
from utils.config import RIDGES
from utils.data_transforms import wrap_data_in_dataloader

#* Parser settings
parser = argparse.ArgumentParser(
    description='Train UCI dataset models with MSE/KARE objectives'
    )
parser.add_argument('-dir', default="./dataset/UCI", type=str,
                    help="data directory")
parser.add_argument('-dataset', default = '', type = str,
                    help = "dataset on which to run the training")
parser.add_argument('-max_tot', default=1000, type=int,
                    help="Maximum number of data samples")
parser.add_argument('-min_tot', default=0, type=int,
                    help="Minimum number of data samples")
parser.add_argument('-lr_index', default=0, type=int,
                    help="Learning rate index for the grid of learning rates")
parser.add_argument('-opt', default='SGD', type=str,
                    help="Optimizer to use for the error minimization.")
parser.add_argument('-depth', default=1, type=int,
                    help="Depth of the neural network.")
parser.add_argument('-width', default=32, type=int,
                    help="Width of the neural network.")
parser.add_argument('-z_kare', default=0.1, type=float,
                    help="Regularization parameter for KARELoss.")
parser.add_argument('-model', default='KARE', type=str,
                    help="Model to use for the training.")
parser.add_argument('-output', default='./results/UCI/',
                    type=str, help="Path to which results should be saved.")

#* Get the info from the parser
args = parser.parse_args()
max_n_tot = args.max_tot
min_n_tot = args.min_tot
datadir = args.dir
dataset = args.dataset
lr_index = args.lr_index
optimizer = args.opt
depth = args.depth
width = args.width
z_kare = args.z_kare
model = args.model
output = args.output

#* Set the other hyperparameters
lr_grid = {'KARE' :{1: [100, 10, 1, 0.1],
               2: [100, 10, 1, 0.1],
               3: [ 10, 5, 1, 0.1],
               4: [10, 5, 1, 0.1],
               5: [10, 5, 1, 0.1]},
           'KARE-NNGP' :{1: [100, 10, 1, 0.1],
               2: [100, 10, 1, 0.1],
               3: [ 10, 5, 1, 0.1],
               4: [10, 5, 1, 0.1],
               5: [10, 5, 1, 0.1]},
            'MSE' : [0.1, 0.01, 0.001, 0.0001]}
epochs_grid = {'KARE' : 300, 'MSE' : 400, 'KARE-NNGP' : 300}
batch_size_grid = {'KARE' : None, 'MSE' : 32, 'KARE-NNGP' : None}

print(datadir)
print(dataset)

def main():
    
    """
    Load UCI dataset metadata, preprocess, and train model.

    Parameters
    ----------
    data_dir : str
        Base directory containing dataset subfolders.
    dataset_name : str
        Name of the dataset to load.
    min_tot : int
        Minimum total observations required to proceed.
    max_tot : int
        Maximum total observations allowed to proceed.
    lr_index : int
        Index into the learning rate grid for the chosen model.
    optimizer : str
        Optimizer name ('SGD' or 'Adam').
    depth : int
        Number of hidden layers for the network.
    width : int
        Hidden layer width.
    z_kare : float
        Ridge penalty parameter for KARE loss.
    model_type : str
        Model objective: 'MSE', 'KARE', or 'KARE-NNGP'.
    output_dir : str
        Directory to save training results and checkpoints.

    Returns
    -------
    None
    """

    # Construct dataset path and metadata file
    if not os.path.isdir(datadir + "/" + dataset):
        return
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        return
    print("Passing first check")
    dic = dict()
    for k, v in map(lambda x: x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v

    fold = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + "/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))

    # Get the dimension of the dataset and number of classes
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])

    # Get dataset size
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test

    # Drop datasets that have too many observations or that have some other test dataset
    #if(n_tot > max_n_tot or n_test > 0) or (n_tot < min_n_tot):
    #    print(str(dataset) + ' Dataset has too many/few observations')
    #    return
    
    # For large dataset and KARE model, we enforce a batch-size which is not full-batch
    if model == "KARE" and n_tot >= 5000:
        print("Using default large sample batch-size")
        batch_size_grid['KARE'] = 32

    # Load the data
    f = open(datadir + "/" + dataset + "/" + dic["fich1="], "rb").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))

    # Encode labels: multi-class one-hot for KARE, binary -1/1 otherwise
    if c > 2:
        if model == 'KARE':
            num_classes = np.max(y) + 1
            y = np.eye(num_classes)[y]
            y[y == 0] = -1
            
    else:
        y[y == 0] = -1
        y[y == 1] = 1

        #* Sanity check that all targets are correctly matched to -1 or 1
        if not np.all(np.isin(y, [-1, 1])):
            print(str(dataset) + ' Problem with the target mapping to -1, 1')
            return

    # We get the train-test indexes
    fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]
        
    print(f"Training {dataset} ,depth {depth}, width {width}")

    # Get the training and test data
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[val_fold], y[val_fold]
    
    # Some preprocessing
    if X.shape[0] >= 100:
        
        lower = np.percentile(X_train, 5, axis=0)    # shape: (n_features,)
        upper = np.percentile(X_train, 95, axis=0)
        
        # 2) Clip both train and test to those same bounds:
        X_train_clip = np.clip(X_train, lower, upper)  
        X_test_clip  = np.clip(X_test,  lower, upper)

        # (Optionally overwrite)
        X_train = X_train_clip
        X_test  = X_test_clip
            
        mean_X = np.mean(X_train, axis = 0)
        std_X = np.std(X_train, axis = 0)
        std_X[std_X == 0] = 1e-8 # Avoid division by zero
        X_train = (X_train - mean_X) / std_X
        X_test = (X_test - mean_X) / std_X
        
    else:
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Move the data to torch loaders
    train_loader, test_loader = wrap_data_in_dataloader(X_train, y_train, X_test, y_test)
    
    # Get the hyperparameters to be used
    lr = lr_grid[model][lr_index] if model == 'MSE' else lr_grid[model][depth][lr_index]
    epochs = epochs_grid[model]
    batch_size = batch_size_grid[model]

    # Run the training loop
    train_on_data(train_loader=train_loader,
                    test_loader=test_loader,
                    name = model,
                    folder=output + dataset,
                    depth=depth,
                    width=width,
                    z_kare=z_kare,
                    lr=lr,
                    krr_ridge_penalty=RIDGES,
                    epochs=epochs,
                    seed=0,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    parameter_specific_lr=False,
                    outdim_nngp = 50000)

if __name__ == "__main__":
    main()
