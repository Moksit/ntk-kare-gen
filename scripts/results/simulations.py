"""
simulations.py

----------------------------------------------
Plotting script to visualize MSE curves from simulation JSON results.

This script reads averaged MSE results for KRR (KARE vs standard NTK)
and DNN baselines from a JSON file, then produces and saves a grid of
plots across experimental conditions (scale, sigma).
"""

import ast  # Safely evaluates strings into Python literals
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Sequence, Tuple

def plot_mse_scale_sigma(
    results: Dict[Tuple[float, float], Sequence],
    ridge_grid: Sequence[float],
    z_kare: float,
    output_path: str
) -> None:
    """
    Plot MSE curves for KRR (KARE vs standard NTK) and DNN baseline.

    Parameters
    ----------
    results : dict
        Mapping (scale, sigma) → [mse_kare_list, mse_kernel_list, mse_nn_scalar].
    ridge_grid : sequence of float
        Ridge penalty values corresponding to the KRR curves.
    z_kare : float
        Reference λ value for a vertical marker in each subplot.
    output_path : str
        Path where the plot image will be saved.
    """
    
    # Global styling
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rc('mathtext', fontset='stix')  # Times-like math fonts
    plt.rc('font', size=11)
    plt.rc('axes', titlesize=12, labelsize=9)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('lines', linewidth=1, markersize=4)
    
    # Determine unique sorted scales and sigmas
    scales = sorted({key[0] for key in results.keys()})
    sigmas = sorted({key[1] for key in results.keys()})
    
    n_rows = len(sigmas)
    n_cols = len(scales)
    
    # Setup subplot grid dimensions
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False)
    
    #Loop over each subplot
    for i, scale in enumerate(scales):
        for j, sigma in enumerate(sigmas):
            ax = axes[j, i]
            
            # Unpack results for the current (scale, sigma) combination
            krr_mse1, krr_mse2, nn_mse = results[(scale, sigma)]
            
            # Plot the KRR MSE curves with enhanced markers and line widths
            ax.plot(ridge_grid, krr_mse1, marker='o', linewidth=1, 
                    label=r'NTK, KARE')
            ax.plot(ridge_grid, krr_mse2, color = 'red', marker='s', linewidth=1, 
                    label=r'after-NTK')
            
            # Plot baseline for Neural Net MSE and the vertical reference line at z_kare
            ax.plot(ridge_grid, nn_mse*np.ones(len(ridge_grid)), color='black', marker = '^', linewidth=1, 
                       label=r'DNN')
            
            # Set titles and axis labels with LaTeX formatting
            ax.set_title(f'$\gamma={scale},\sigma={sigma}$', fontsize=10)
            ax.set_xscale('log')
            ax.set_xlabel('Ridge Penalty $\\tilde \lambda$', fontsize=10)
            ax.set_ylabel('MSE', fontsize=10)
            #ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, alpha=0.7, linestyle='--')
            
            # Add legend with a frame; adjust fontsize as needed
            ax.legend(fontsize=8, frameon=True)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
def main(
    file_path : str = "./results/sim/v1_avg10.json",
    output_path: str = "./figures/simulations/sim_mse_curves.pdf"
    ) -> None:
    
    """
    Read simulation results and produce MSE plots. You first need to run scripts.training.train_sim.py

    Parameters
    ----------
    file_path : str, default='./results/sim/v1_avg10.json'
        Path to JSON file containing 'results' and 'params'.
    output_path : str, default='./figures/simulations/sim_mse_curves.png'
        File path to save the generated plot image.
    """

    # Load JSON into dict
    with open(file_path, "r") as f:
        str_key_dict = json.load(f)
        results = str_key_dict['results']
        params  = str_key_dict['params']

    # Convert string keys back to tuples
    tuple_key_dict = {
        ast.literal_eval(k): v 
        for k, v in results.items()}
    
    ridge_grid = params["KRR_RIDGE_PENALTY"]
    z_kare = params["Z_KARE"]
    
    plot_mse_scale_sigma(tuple_key_dict, ridge_grid, z_kare, output_path)
    
if __name__ == "__main__":
    main()