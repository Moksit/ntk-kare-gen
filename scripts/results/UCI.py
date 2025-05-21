"""
UCI.py
--------------------------------------------------

Analysis and reporting utilities for UCI experiment results.

Provides functions to compute per-model scores, aggregate best performances,
perform statistical analysis, and export results as a LaTeX table.
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def get_score_per_model_type(
    dataset_path : str,
    score_method : str,
    substep : str = ''
) -> Dict[str, List[np.ndarray]]:
    """
    Load predictions and compute per-run scores per model configuration.

    Parameters
    ----------
    dataset_path : str
        Directory containing run subdirectories.
    score_method : {'acc', 'mse'}
        Metric to compute: classification accuracy or MSE.
    substep : str, default=''
        Suffix for prediction filenames (e.g. 'epoch10').

    Returns
    -------
    dict
        Maps model prefix to list of score arrays from each seed.
    """
    
    #We store the accuracy for each seed in a dictionnary for each model and each hyperparametrization
    acc_per_model = defaultdict(list)
    
    for runs in os.listdir(dataset_path):
        
        #We skip the Adam experiments
        if "Adam" in runs or "NAIVE" in runs:
            continue
        
        model_save = '_'.join(runs.split('_')[:-1])
        
        #We kill the seed and keep the rest as an ID for the model
        pred_file = os.path.join(dataset_path, runs, f'predictions{substep}.npy')
        tgt_file = os.path.join(dataset_path, runs, 'targets.npy')
        if not os.path.exists(pred_file) or not os.path.exists(tgt_file):
            continue
        #Get the predictions
        predictions = np.load(pred_file)
        
        #Get the targets
        targets = np.load(tgt_file).reshape(-1,1)

        #Compute the score
        if score_method == 'acc':
            
            if np.min(targets) == -1:
                cpred = predictions.copy()
                cpred[cpred <= 0] = -1
                cpred[cpred > 0] = 1
                
            else:
                cpred = predictions
            score = np.mean((cpred == targets), axis = 0)
            
        elif score_method == 'mse':
            score = np.mean((predictions - targets)**2, axis = 0)
        else:
            raise ValueError(f"Unknown score_method {score_method}")
            break

        acc_per_model[model_save].append(score)
    
    return acc_per_model

def get_per_model_best_score(
    dataset_path : str,
    score_method : str, 
    substep : str = '',
    model_types : List[str] = ('KARE', 'MSENTK'),
) -> Tuple[Dict[str, float], Dict[str, List[np.ndarray]]]:
    """
    Determine best score per model type and collect all per-seed scores.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder.
    score_method : {'acc', 'mse'}
        Metric to compute: classification accuracy or MSE.
    substep : str
        Suffix for prediction filenames (e.g. 'epoch10').
    model_types : list of str
        Model prefixes to consider (e.g. 'KARE', 'MSENTK').

    Returns
    -------
    best_scores : dict
        Best score (max accuracy or min MSE) per method key.
    all_scores : dict
        All per-seed score arrays for each model prefix.
    """
    
    avg_acc_per_model = get_score_per_model_type(dataset_path, score_method, substep = substep)
    
    best_scores = {model : 0 for model in model_types}
    best_key    = {model : '' for model in model_types}
    
    for key, value in avg_acc_per_model.items():
        
        for mod in model_types:
             if key.startswith(mod):
                 
                if np.max(value) > best_scores[mod]:
                    best_scores[mod] = np.max(value)
                    best_key[mod] = key
                    
    targets = np.load(os.path.join(dataset_path, best_key['KARE'] + '_0', 'targets.npy')).reshape(-1,1)
    
    if np.min(targets) == -1:
        best_kare_pred = np.load(os.path.join(dataset_path, best_key['KARE'] + '_0', f'predictions{substep}.npy'))
        best_ntk_pred  = np.load(os.path.join(dataset_path, best_key['MSENTK'] + '_0', f'predictions{substep}.npy'))
        
    else:
        best_kare_pred = torch.load(os.path.join(dataset_path, best_key['KARE'] + '_0', f'probabilities.pt'), map_location = torch.device('cpu'))
        best_ntk_pred  = torch.load(os.path.join(dataset_path, best_key['KARE']+ '_0', f'probabilities.pt'), map_location = torch.device('cpu'))
        
        best_kare_pred = np.array([value.numpy() for key, value in best_kare_pred.items()])
        best_ntk_pred  = np.array([value.numpy() for key, value in best_ntk_pred.items()])
        
    ensemble_pred = (best_kare_pred + best_ntk_pred)/2
    
    if np.min(targets) == -1:
        cpred = ensemble_pred.copy()
        cpred[cpred <= 0] = -1
        cpred[cpred > 0] = 1
    else:
        cpred = np.argmax(ensemble_pred, axis = 2).T
        
    accuracy = np.mean((cpred == targets), axis = 0)
    best_scores['ensemble'] = np.max(accuracy)
                    
    return best_scores, avg_acc_per_model

def analyze_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical metrics and Friedman ranks across datasets.

    Parameters
    ----------
    df : DataFrame
        Rows=datasets, cols=models, values=accuracy.

    Returns
    -------
    DataFrame
        Columns: Friedman Rank, Mean ± 1σ, P90, P95, PMA ± 1σ.
    """
    
    N = df.shape[0]  # number of datasets
    K = df.shape[1]  # number of models
    
    #Read models from Delgado
    df_delgado = pd.read_fwf("./scripts/results/results_UCI.txt", header=0).T
    df_delgado.columns = df_delgado.loc['problema']
    df_delgado = df_delgado.drop('problema')
    df_delgado = df_delgado.iloc[:-1,:-3]
    for col in df_delgado.columns:
        df_delgado[col] = pd.to_numeric(df_delgado[col], errors = 'coerce')/100
        
    df = pd.merge(df, df_delgado.T, left_index = True, right_index=True, how = 'left')
    
    df = df[df['KARE'] != 0]
    df = df[df['MSENTK'] != 0]

    # 1. Friedman Rank: average rank per model (lower rank = better)
    ranks = df.rank(axis=1, ascending=False)  # higher accuracy = better rank
    friedman_rank = ranks.mean()

    # 2. Mean ± 1 sigma
    mean_accuracy = df.mean()*100
    std_accuracy = df.std()*100
    mean_acc_str = mean_accuracy.map('{:.3f}'.format) + ' ± ' + std_accuracy.map('{:.3f}'.format)

    # 3. P90 and P95
    best_acc = df.max(axis=1)
    p90 = ((df.ge(best_acc * 0.90, axis=0).sum() / len(df)).round(5)*100).map('{:.2f}'.format)
    p95 = ((df.ge(best_acc * 0.95, axis=0).sum() / len(df)).round(5)*100).map('{:.2f}'.format)

    # 4. PMA ± 1 sigma (accuracy / best per row)
    pma = (df.div(best_acc, axis=0) * 100)
    pma_mean = pma.mean()
    pma_std = pma.std()
    pma_str = pma_mean.map('{:.2f}'.format) + ' ± ' + pma_std.map('{:.2f}'.format)

    # Assemble final table
    result = pd.DataFrame({
        "Friedman Rank": friedman_rank,
        "Mean Acc ± 1σ": mean_acc_str,
        "P90": p90,
        "P95": p95,
        "PMA ± 1σ": pma_str
    })

    return result
                    
def get_dataset_folder_scores(
    data_path : str,
    score_method : str,
    substep : str = '',
    model_types : Tuple[str] = ('KARE', 'MSENTK'),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Get the scores for each dataset in the folder and return the score per dataset as well as Belkin's style table.
    
    Parameters
    ----------
    data_path : str
        Path to the folder containing the result folders.
    score_method : str
        Method to use for scoring ('acc' or 'mse').
    substep : str, default=''
        Substep to use for scoring (e.g. '10' is for 10% of the training).
    model_types : tuple of str, default=('KARE', 'MSENTK')
        Model types to consider for scoring.
        
        

    Returns
    -------
    df_res : DataFrame
        DataFrame containing the scores for each dataset.
        
    table : DataFrame
        DataFrame containing the scores for each dataset in Belkin's style.
    """
    
    results = {}
    results_all = {}
    all_datasets = os.listdir(data_path)
    
    failed_training = []#['glass', 'GesturePhaseSegmentationProcessed','eye_movements', 'gas-drit-different-concentrations', 'higgs', 'mnist', 'telco', 'churn']
    
    for dataset in all_datasets:
        
        if dataset in failed_training:
            continue

        results[dataset], results_all[dataset] = get_per_model_best_score(os.path.join(data_path, dataset),
                                                                            score_method = score_method,
                                                                            substep = substep,
                                                                            model_types = model_types)
        
    df_res = pd.DataFrame(results).T.sort_index()
    
    table = analyze_model_performance(df_res)
                    
    return df_res, table



def df_to_neurips_latex(
    df : pd.DataFrame,
    caption="Your caption here",
    label="tab:your_label",
    percent_cols=('Mean Acc ± 1σ', 'P90', 'P95'),
    decimals=2,
    bold_max=False
) -> str:
    
    """
    Convert DataFrame to a NeurIPS-style LaTeX table.

    Parameters
    ----------
    df : DataFrame
    caption : str
    label : str
    percent_cols : list of str
    decimals : int
    bold_max : bool

    Returns
    -------
    str
        LaTeX table.
    """

    df = df.copy()
    percent_cols = percent_cols or []

    # Track original (unformatted) values for max detection
    original_df = df.copy()
            
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_max = original_df[col].max()
            if col in percent_cols:
                df[col] = (df[col]*100).apply(
                    lambda x: f"\\textbf{{{x:.{decimals}f}}}" if bold_max and np.isclose(x, col_max) else f"{x:.{decimals}f}"
                )
                
            else:
                df[col] = df[col].apply(
                    lambda x: f"\\textbf{{{x:.{decimals}f}}}" if bold_max and np.isclose(x, col_max) else f"{x:.{decimals}f}"
                )
                
    # Format percentage columns
    for col in percent_cols:
        if col in df.columns:
            df[col] = (df[col]).str.replace(r'([0-9.]+)', lambda m: f"{m.group(1)}\\%", regex=True)

    # Convert to LaTeX
    col_format = 'l' + 'c' * (df.shape[1] - 1)
    table_latex = df.to_latex(index=False, escape=False, column_format=col_format,
                              header=True, bold_rows=False, longtable=False, na_rep='--',
                              multicolumn=False, multicolumn_format='c',
                              caption='', label='')

    wrapped = f"""
\\begin{{table}}[t]
  \\centering
  \\caption{{{caption}}}
  \\label{{{label}}}
{table_latex}
\\end{{table}}
""".strip()

    return wrapped

def export_table_to_tex(
    df: pd.DataFrame,
    output_dir: str,
    filename: str = 'uci_results_table.tex',
    **latex_kwargs: Any
) -> None:
    """
    Export a DataFrame as a LaTeX table file.

    Parameters
    ----------
    df : DataFrame
    output_dir : str
    filename : str
    latex_kwargs : dict
        Arguments forwarded to df_to_neurips_latex.
    """
    os.makedirs(output_dir, exist_ok=True)
    tex_str = df_to_neurips_latex(df, **latex_kwargs)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex_str)
    print(f"LaTeX table written to {path}")


        
if __name__ == "__main__":
    
    res, table = get_dataset_folder_scores('./results/UCI/', 'acc', substep = '', model_types = ('KARE', 'MSENTK'))
    
    export_table_to_tex(
        table.reset_index(),
        output_dir = './results/UCI/',
        filename = 'uci_results_table.tex'
    )
    
    
    
            
            
        
        
        