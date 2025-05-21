"""
datasets.py

--------------------------------------------------------------
Data loading utilities for various datasets: Higgs (Kaggle), synthetic COS, MNIST (digits 7/9), and CelebA.

Provides functions to wrap datasets into TensorDatasets/DataLoaders with NumPy-style docstrings and inline comments.
"""

import os

import torch
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_higgs_data_kaggle(
    train_ratio=0.5,
    batch_size=None,
    train_samples=2000,
    test_samples=400,
    seed=0
)-> tuple[DataLoader, DataLoader]:
    """
    Load and preprocess the Kaggle Higgs dataset from CSV, filtering and splitting.
    The dataset is available at : https://www.kaggle.com/competitions/higgs-boson/data

    Parameters
    ----------
    csv_file : str
        Path to the Higgs CSV file.
    train_ratio : float, default=0.5
        Fraction of data to allocate to training set.
    batch_size : int or None, default=None
        Batch size for training DataLoader. If None, uses full training set.
    train_samples : int, default=2000
        Maximum number of training samples to randomly subsample.
    test_samples : int, default=400
        Maximum number of test samples to randomly subsample.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for training set.
    test_loader : DataLoader
        DataLoader for test set.
    """
    
    # Set random seed for NumPy operations
    np.random.seed(seed)

    # Determine batch size for training loader
    if batch_size is None:
        batch_size = train_samples

    # Load the CSV file into a DataFrame.
    df = pd.read_csv("./dataset/kaggle_higgs/training.csv")
    
    # Remove identifier column if present
    df.drop(columns=["EventId"], inplace=True)
    
    # Drop rows with missing values coded as -999
    df = df[~(df == -999).any(axis=1)]

    # Separate features and labels.
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    
    # Convert labels: 's' → -1.0, 'b' → 1.0
    y[y == "s"] = -1.0
    y[y == "b"] = 1.0
    y = y.astype(float)

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio
        , random_state=42, stratify=y
    )

    # Subsample training and test sets if larger than requested.
    if train_samples is not None and len(X_train) > train_samples:
        indices = np.random.choice(len(X_train), train_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        indices = np.random.choice(len(X_test), test_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]

    # Print class balance for training and test sets
    print("Number of y == 1 in train dataset :", np.sum(y_train == 1) / len(y_train))
    print("Number of y == -1 in train dataset :", np.sum(y_train == -1) / len(y_train))
    print("Number of y == 1 in test dataset :", np.sum(y_test == 1) / len(y_test))
    print("Number of y == -1 in test dataset :", np.sum(y_test == -1) / len(y_test))

    # Feature scaling: normalize by max of training features -> map to [-1,1]
    scale = np.max(X_train, axis=0)
    X_train = X_train / scale
    X_test = X_test / scale

    # Convert data to PyTorch tensors.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets and DataLoaders.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

    return train_loader, test_loader


def create_train_test_dataloaders_cos(
    d: int = 10,
    n: int = 500,
    scale: float = 1.5,
    sigma: float = 0.0,
    shuffle: bool = False,
    seed=0,
) -> tuple[DataLoader, DataLoader]:
    """
    Generate synthetic nonlinear COS dataset and return DataLoaders.

    Produces labels via a cosine-transformed linear function plus Gaussian noise.

    Parameters
    ----------
    d : int, default=10
        Number of features.
    n : int, default=500
        Total number of samples.
    scale : float, default=1.5
        Scaling factor for linear features.
    sigma : float, default=0.0
        Standard deviation of Gaussian noise.
    shuffle : bool, default=False
        If True, shuffle training data.
    seed : int, default=0
        Random seed.

    Returns
    -------
    train_loader, test_loader : tuple of DataLoader
        Full-dataset DataLoaders (batch size = dataset size).
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate raw features
    raw_features = np.random.randn(n, d)  # X_t, t = 1, ..., n

    # Generate true weights
    w = np.random.randn(d, 1)

    # Compute true linear features
    true_linear_features = scale * raw_features @ w  # X_t * w

    # Generate noise
    noise = np.random.randn(n, 1) * sigma  # N(0, sigma^2)

    # Compute true conditional expectation
    infeasible_nonlinear_predictions = np.cos(true_linear_features) / (
        1 + np.exp(true_linear_features)
    )

    # Compute labels
    labels = infeasible_nonlinear_predictions + noise

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(raw_features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    # Define train-test split sizes (80% train, 20% test)
    train_size = int(0.8 * n)
    test_size = n - train_size
    train_dataset, test_dataset = TensorDataset(
        X_tensor[:train_size], y_tensor[:train_size]
    ), TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    # Create DataLoaders with full-batch size
    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=shuffle)
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False
    )

    return train_loader, test_loader


def load_mnist_data(
    train_ratio=0.5,
    batch_size=None,
    train_samples=2000,
    test_samples=400,
    seed=0
)-> tuple[DataLoader, DataLoader]:
    """
    Load and filter MNIST to digits 7 and 9, returning DataLoaders. 

    Converts label 7→1.0 and 9→-1.0, crops center, and optionally subsamples.
    
    The code takes care of downloading the dataset if not already present.

    Parameters
    ----------
    batch_size : int or None, default=None
        Batch size for training loader; uses train_samples if None.
    train_samples : int, default=2000
        Number of training samples to subsample.
    test_samples : int, default=400
        Number of test samples to subsample.
    seed : int, default=0
        Random seed.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for filtered MNIST training data.
    test_loader : DataLoader
        DataLoader for filtered MNIST test data.
    """
    
    # Set default batch_size if not provided.
    if batch_size is None:
        batch_size = train_samples

    # Ensure reproducibility.
    np.random.seed(seed)

    # Define a basic transformation that converts images to tensors.
    transform = transforms.ToTensor()

    # Download and load the MNIST datasets.
    full_train_dataset = datasets.MNIST(
        root="./dataset/", train=True, download=True, transform=transform
    )
    full_test_dataset = datasets.MNIST(
        root="./dataset/", train=False, download=True, transform=transform
    )

    # Filter function: keep only digits 7 and 9, converting labels: 7 -> -1, 9 -> 1.
    def filter_and_convert(dataset):
        filtered = []
        for img, label in dataset:
            if label not in [7, 9]:
                continue

            # Crop 2 pixels from each side.
            if img.ndim == 2: 
                cropped_img = img[2:-2, 2:-2]
            elif img.ndim == 3:
                cropped_img = img[:, 2:-2, 2:-2]
            else:
                raise ValueError("Unexpected image shape: expected 2D or 3D tensor")

            # Flatten the cropped image
            flat_img = torch.flatten(cropped_img)
            flat_img = flat_img - flat_img.mean(dim=0)

            # Assign label: 1.0 for 7, -1.0 for 9
            new_label = 1.0 if label == 7 else -1.0
            filtered.append((flat_img, new_label))

        # Stack flattened images and convert labels to a tensor.
        images = torch.stack([img for img, lbl in filtered])
        labels = torch.tensor([lbl for img, lbl in filtered], dtype=torch.float32)
        return images, labels

    # Filter the training and testing datasets.
    train_images, train_labels = filter_and_convert(full_train_dataset)
    test_images, test_labels = filter_and_convert(full_test_dataset)

    # Optionally subsample the filtered training data.
    num_train = train_images.shape[0]
    if train_samples is not None and train_samples < num_train:
        indices = np.random.choice(num_train, train_samples, replace=False)
        train_images = train_images[indices]
        train_labels = train_labels[indices]

    # Optionally subsample the filtered test data.
    num_test = test_images.shape[0]
    if test_samples is not None and test_samples < num_test:
        indices = np.random.choice(num_test, test_samples, replace=False)
        test_images = test_images[indices]
        test_labels = test_labels[indices]

    # Print label distribution for training data.
    unique, counts = np.unique(train_labels.numpy(), return_counts=True)
    print("Training label distribution:")
    for label, count in zip(unique, counts):
        print(
            f"  Label {int(label)}: {count} samples, ratio: {count/len(train_labels):.4f}"
        )

    # Print label distribution for test data.
    unique, counts = np.unique(test_labels.numpy(), return_counts=True)
    print("Testing label distribution:")
    for label, count in zip(unique, counts):
        print(
            f"  Label {int(label)}: {count} samples, ratio: {count/len(test_labels):.4f}"
        )

    # Create TensorDatasets.
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, test_loader
