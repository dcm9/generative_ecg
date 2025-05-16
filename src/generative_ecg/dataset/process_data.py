import tqdm
import jax
import jax.numpy
import pandas
import ast
import wfdb

from pathlib import Path

from ..models.math_utils import compute_linproj_residual

def load_signals(filepath: str, sampling_rate: int, target: str='age') -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """
    Load the ECG signals from the PTB-XL dataset.
    Args:
        filepath (str): Path to the PTB-XL dataset.
        sampling_rate (int): Sampling rate of the ECG signals.
        target (str): Target variable for the dataset. Default is 'age'.

    Returns:
        X (ndarray): ECG signals.
        y (ndarray): Target variable.
    """
    if filepath is None:
        raise ValueError("ECG filepath must be provided.")

    df_annot = pandas.read_csv(
        Path(filepath, "ptbxl_database.csv"), 
        index_col="ecg_id"
    )

    df_annot.scp_codes = df_annot.scp_codes.apply(ast.literal_eval)
    # DEFAULT 500
    if sampling_rate == 500:
        file_paths: list = list(df_annot.filename_hr)

    elif sampling_rate == 100:
        file_paths: list = list(df_annot.filename_lr)

    data = []
    for i, f in enumerate(tqdm.tqdm(file_paths, desc="Loading data from records")):
        data.append(wfdb.rdsamp(Path(filepath, f)))

    X = jax.numpy.array([signal for signal, _ in data]).transpose(0, 2, 1)
    y = df_annot[target].values

    return X,y

def project(x_beats: list, y_beats:list, tol: float=1e-6) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """
    Load the ECG signals and target variables from the PTB-XL dataset.

    Args:
        filepath (str): Path to the PTB-XL dataset directory.
        sampling_rate (int): Sampling rate of the ECG signals (e.g., 500 or 100).
        target (str): Name of the target variable column in the annotation file. Default is 'age'.

    Returns:
        X (jax.numpy.ndarray): Array of ECG signals with shape (num_samples, num_channels, num_timesteps).
        y (jax.numpy.ndarray): Array of target variable values.
    """
    X_proj, y_proj = jax.numpy.array([]), jax.numpy.array([])

    processed_data = []
    processed_labels = []

    for i, x in enumerate(x_beats):
        x_transpose = jax.numpy.transpose(x, (1, 0))
        sol, res = jax.vmap(compute_linproj_residual)(x_transpose)
        if jax.numpy.mean(res) < tol:
            processed_data.append(sol.T)
            processed_labels.append(y_beats[i])
    
    if processed_data:
        X_proj = jax.numpy.stack(processed_data)
        y_proj = jax.numpy.array(processed_labels)

    return X_proj, y_proj
