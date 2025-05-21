"""
Configuration constants for model hyperparameters.

RIDGES : list of float
    Grid of ridge penalty values for kernel ridge regression.
"""

# Predefined ridge penalty grid for kernel ridge regression
RIDGES: list[float] = [
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1e0,
    1e1,
    1e2,
    1e3,
    1e4,
]

