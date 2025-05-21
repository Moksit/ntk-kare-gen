# Training NTK to Generalize with KARE

# Setup programming environment

You can reproduce our python environment using 

```
conda env create -f environment.yml
conda activate BDGD
```


# Dataset Sources

The table below lists the datasets used in this project, along with their sources and direct download links. These datasets cover a variety of tasks, including classification, regression, and time-series analysis. MNIST dataset can directly be downloaded using ```load_mnist_data```from utils.datasets. The other ones can be downloaded using the link below :

| Dataset            | Source      | Link                                                                                          |
|--------------------|-------------|----------------------------------------------------------------------------------------------|
| Higgs              | Kaggle      | [Kaggle Higgs Boson Competition](https://www.kaggle.com/competitions/higgs-boson/data)        |
| MNIST              | PyTorch     | [torchvision.datasets.MNIST](https://pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#MNIST) |
| UCI                | UC Irvine   | [Delgado (2014) datasets](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr)  |

Please save the higgs dataset under : *./dataset/kaggle_higgs/training.csv*. For the UCI dataset, you can use:

```
wget http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz
mkdir UCI
tar -xvzf data.tar.gz -C UCI
```

Note: The UCI folder should be saved under */dataset*.

# Running the experiments

## Simulations
The simulation experiment can easily be run locally using :

```
python -m scripts.training.train_sim
```

The results will be saved to *results/sim*.

```
python -m scripts.results.simulations
```

The hyperparameters can be changed in the file.

## Higgs
The Higgs experiments can also be run locally using

```
python -m scripts.training.train_higgs -model KARE
```

The hyperparameters can be changed in the file.You can change the ```model``` argument to ```MSE``` to train the after-NTK and the DNN. The results will be saved to *results/higgs*.

## MNIST
You can run the higgs experiment running 

```
python -m scripts.training.train_mnist -model KARE
```

The hyperparameters can be changed in the file. You can change the ```model``` argument to ```MSE``` to train the after-NTK and the DNN. The results will be save to *results/mnist*.

## UCI 
The whole training-evaluation loop of the UCI datasets requires extensive compute power. For small dataset, you can run them locally using:

```
python -u scripts/training/train_UCI.py
    -dir ./dataset/UCI #path to the datasets
    -dataset"DATASET_NAME #try with balloons
    -lr_index 0 #This is the learning rate index on the grid
    -depth 1 #Depth of the network
    -width 32 #Width of the network
    -output ./results/UCI/
    -model KARE # Model to train : KARE or MSE
    -z_kare 0.1 #Ridge regularized to be used for KARELoss
```

If you want to run the whole loop on a compute cluster, we provide a template in *scripts/batch/train_grid_UCI*. Please note that you will need to adapt the script to the constraints of your cluster. Also note, that the training whole loop can be splited based on dataset sizes.

# Producing results
## Simulations

Once you have run the simulations train-test procedure, you can reproduce Figure 1 of the paper using 

```
python -m scripts.results.simulations
```

The results plot will be saved in *figures/simulations*

## MNIST + Higgs
Once you have run the Higgs and MNIST scripts for both KARE and DNN/after-NTK, you can reproduce Figure 2 of the paper using

```
python -m scripts.results.higgs_mnist
```

## UCI

Once you have run the training loop, you can recrate table 1 using:

```
python -m scripts.results.UCI
```

The latex table will be written to *resuts/UCI*. In particular, note that you can also produce a table for a subset of the 121 datasets from [Delgado (2014) datasets](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr). You need to run both DNN/after-NTK and NTK-KARE for the results script to work.
