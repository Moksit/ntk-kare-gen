"""
higgs_mnist.py

-----------------------------------------------
Plotting utilities for NeurIPS-compliant figures of MSE across model depths and widths.

Includes functions to compute average MSE per architecture from saved predictions and
to generate a multi-panel plot comparing methods across datasets.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from typing import Dict, List, Sequence, Tuple, Any

from utils.config import RIDGES


def get_per_depth_average_mse(
    datapath : str
    ) -> Dict[str, np.ndarray]:
    
    """
    Aggregate average MSE per model configuration from saved predictions.

    Parameters
    ----------
    datapath : str
        Directory containing subdirectories for each run, with 'predictions.npy' and 'targets.npy'.

    Returns
    -------
    dict
        Mapping each model identifier (depth_width tag) to its average MSE array or scalar.
    """
    
    # List run directories
    all_models = os.listdir(datapath)
    
    mse_per_model = defaultdict(list)
    
    for run in all_models:
        
        model_save = run.split('_')[:-1]
        
        predictions = np.load(os.path.join(datapath, run, "predictions.npy"))
            
        # Load corresponding targets, handling 'NAIVE' suffix variants
        if 'NAIVE' in run:
            # try replacing suffix to find matching targets
            for suffix in ['KARE', 'MSENTK', 'NN']:
                try:
                    tgt = np.load(os.path.join(datapath, run.replace('NAIVE', suffix), 'targets.npy'))
                    break
                except FileNotFoundError:
                    continue
            else:
                continue
        else:
            tgt = np.load(os.path.join(datapath, run, 'targets.npy'))
        targets = tgt.reshape(-1, 1)
        
        
        #Compute MSE per ridge index
        mse = np.mean((predictions - targets)**2, axis = 0)

        mse_per_model['_'.join(model_save)].append(mse)

    avg_mse_per_model = {key : np.mean(val, axis = 0) if any(sub in key for sub in ['KARE', 'MSENTK']) else np.mean(val) for key, val in mse_per_model.items()}
    
    return avg_mse_per_model

def plot_neurips_results(
    dicts : List[Dict[str, Any]],
    first_args: Sequence[int],
    second_args: Sequence[int],
    methods: Sequence[str]=('MSENTK','KARE','NN'),
    names: Sequence[str] = None,
    gap_ratio: float = 0.05,
    output_path: str = './figures/higgs_mnist/higgs_mnist.pdf',
) -> None:
    """
    Generate a NeurIPS-style multi-panel figure comparing MSE across methods.

    Parameters
    ----------
    result_dicts : list of dict
        Two dicts mapping stringified keys '(depth, width)' to method→MSE values.
    depths : sequence of int
        List of model depths to layout rows.
    widths : sequence of int
        List of model widths to layout columns.
    methods : sequence of str
        Method keys to plot in order.
    dataset_names : sequence of str, length 2
        Titles for top of each block of subplots.
    gap_ratio : float, default=0.05
        Fractional height of gap row between dataset blocks.
    ridge_grid : sequence of float
        Ridge penalty values for x-axis of KRR curves.
    output_path : str
        File path to save the PDF figure.
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

    n_dicts = len(dicts)
    n_rows_per = len(first_args)
    n_cols = len(second_args)

    # Height ratios: block0, gap, block1
    height_ratios = [1] * n_rows_per + [gap_ratio] + [1] * n_rows_per
    total_height = 4 * n_rows_per * n_dicts + 4 * gap_ratio
    fig = plt.figure(figsize=(5 * n_cols, total_height))
    gs = GridSpec(nrows=len(height_ratios), ncols=n_cols,
                  height_ratios=height_ratios, figure=fig)

    # Prepare axes container
    axes = np.empty((n_dicts * n_rows_per, n_cols), dtype=object)
    
    method_to_names = {'NAIVE': '$Naive$',
                       'MSENTK': 'after-NTK',
                       'KARE': 'NTK-KARE',
                       'NN': 'DNN'}

    # Plot data blocks
    for d_idx, data in enumerate(dicts):
        for i, first in enumerate(first_args):
            # Compute grid row: shift second block by n_rows_per + 1
            grid_row = i if d_idx == 0 else i + n_rows_per + 1
            for j, second in enumerate(second_args):
                ax = fig.add_subplot(gs[grid_row, j])
                axes[d_idx * n_rows_per + i, j] = ax

                key = str((str(first), str(second)))
                if key not in data:
                    continue

                # Determine x-range
                y_lengths = [np.array(v).flatten().size for v in data[key].values()]
                x_vals = RIDGES[1:-1]

                # Plot each method with colors/markers
                color_dict = {"NAIVE": 'green', "MSENTK": 'red', "KARE": 'blue', "NN": 'black'}
                marker_dict = {"NAIVE": 'o', "MSENTK": 's', "KARE": '^', "NN": 'x'}
                for method in methods:
                    if method not in data[key]:
                        continue
                    y_arr = np.array(data[key][method]).flatten()
                    if y_arr.size > 1:
                        ax.plot(x_vals[:y_arr.size], y_arr[1:-1],
                                label=method_to_names[method],
                                color=color_dict.get(method),
                                marker=marker_dict.get(method))
                    else:
                        ax.plot(x_vals, y_arr.item() * np.ones_like(x_vals),
                                label=method_to_names[method],
                                color=color_dict.get(method),
                                marker=marker_dict.get(method))

                # Subplot title and labels
                ax.set_title(f"depth={first}, width={second}")
                if j == 0:
                    ax.set_ylabel('MSE', fontsize = 12)
                if i == n_rows_per - 1:
                    ax.set_xlabel("$\\tilde \\lambda$", fontsize =12)

                ax.set_xscale('log')
                ax.grid(True, alpha=0.7, linestyle='--')

    # Add dataset names above each block
    if names and len(names) == 2:
        mid_col = n_cols // 2
        # First block
        top_ax = axes[0, mid_col]
        top_ax.text(
            0.5, 1.1,
            names[0],
            transform=top_ax.transAxes,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
        # Second block (row index = n_rows_per)
        bottom_ax = axes[n_rows_per, mid_col]
        bottom_ax.text(
            0.5, 1.1,
            names[1],
            transform=bottom_ax.transAxes,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
        # Make room at top
        fig.subplots_adjust(top=0.9)

    # Shared legend in the first block's top-right
    axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300)
    
    
def get_result_dictionnary(
    head_datapath : str
) -> None:
    
    """
    Load and clean results for multiple datasets, then plot NeurIPS figure.

    Parameters
    ----------
    base_path : str
        Root directory containing subfolders for each dataset's results.
    dataset_dirs : list of str
        Subdirectory names under base_path for each dataset.
    depths : sequence of int
        Depth values used to index result dictionaries.
    widths : sequence of int
        Width values used to index result dictionaries.
    output_path : str
        File path for saving the combined plot PDF.
    """
    
    higgs_res = get_per_depth_average_mse(os.path.join(head_datapath, 'higgs'))
    mnist_res = get_per_depth_average_mse(os.path.join(head_datapath, 'mnist'))
    
    #Refactor the results
    clean_mnist_res = defaultdict(dict)
    for key, val in mnist_res.items():
        
        key_split = key.split('_')
        
        model_name = key_split[0]
        model_depth = key_split[2]
        model_width = key_split[3]
        
        clean_mnist_res[str((model_depth, model_width))][model_name] = val
        
    clean_higgs_res = defaultdict(dict)
    for key, val in higgs_res.items():
        
        key_split = key.split('_')
        
        model_name = key_split[0]
        model_depth = key_split[2]
        model_width = key_split[3]
        
        clean_higgs_res[str((model_depth, model_width))][model_name] = val
        
    plot_neurips_results([clean_mnist_res, clean_higgs_res],
                         [2, 4],
                         [32, 64, 128],
                         names = ['MNIST', 'Higgs'])

if __name__ == "__main__":
    
    get_result_dictionnary("./results/")