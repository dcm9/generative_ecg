from pathlib import Path

import ast
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
import wfdb

from .segment_ecg import segment_and_filter_ecg

def load_data(ecg_filepath=None, sampling_rate=500, test_fold=10, 
              segmentation=False, target="age", processed=False):
    """
    Load ECG data from specified filepath, schema is ptb-xl.
    Args:
        ecg_filepath (str, optional): The filepath of the ECG data. Defaults to None.
        sampling_rate (int, optional): The sampling rate of the ECG data. Must be 100 or 500. Defaults to 500.
        test_fold (int, optional): The test fold number. Defaults to 10.
        segmentation (bool, optional): Whether to perform segmentation. Defaults to True.
        target (str, optional): The target variable. Defaults to "age".
        processed (bool, optional): Whether to load processed data. Defaults to False.
    Returns:
        tuple: A tuple containing the loaded ECG data in the following order:
            - X_train (numpy.ndarray): The training data.
            - y_train (numpy.ndarray): The training labels.
            - X_test (numpy.ndarray): The test data.
            - y_test (numpy.ndarray): The test labels.
    """
    # print("entered function")

    if ecg_filepath is None:
        raise ValueError("ecg_filepath must be provided.")
    
    if sampling_rate not in [100, 500]:
        raise ValueError("sampling rate is not in [100, 500]")

    if segmentation:
        X_name = "X_seg.npy" 
        y_name = f"y_{target}_seg.npy"
        X_te_name = "X_te_seg.npy"
        y_te_name = f"y_te_{target}_seg.npy"
    else: 
        X_name = "X.npy"
        y_name = f"y_{target}.npy"
        X_te_name = "X_te.npy"
        y_te_name = f"y_te_{target}.npy"
    
    processed_path = Path(ecg_filepath, "processed")
    # curr_path = Path(ecg_filepath, f"records{sampling_rate}")

    if processed:
        return _load_processed_data(processed_path, X_name, y_name, X_te_name, y_te_name)

    X_path = Path(processed_path, X_name)
    y_path = Path(processed_path, y_name)
    X_te_path = Path(processed_path, X_te_name)
    y_te_path = Path(processed_path, y_te_name)


    df_annot = pd.read_csv(
        Path(ecg_filepath, "ptbxl_database.csv"), 
        index_col="ecg_id"
    )

    df_annot.scp_codes = df_annot.scp_codes.apply(ast.literal_eval)
    if sampling_rate == 100:
        file_paths = list(df_annot.filename_lr)
    else:
        file_paths = list(df_annot.filename_hr)

    data = []

    for i, f in enumerate(tqdm.tqdm(file_paths, desc="Loading data from records")):
        data.append(wfdb.rdsamp(Path(ecg_filepath, f)))

    X = np.array([signal for signal, _ in data])
    if target in df_annot.columns:
        y = df_annot[target].values
        y_tr = y[np.where(df_annot.strat_fold != test_fold)]
        y_te = y[np.where(df_annot.strat_fold == test_fold)]
    else:
        if segmentation:
            y_tr, y_te = None, None
        else:
            y = _load_targets(X, target)
            y_tr = y[np.where(df_annot.strat_fold != test_fold)]
            y_te = y[np.where(df_annot.strat_fold == test_fold)]
    X_tr = X[np.where(df_annot.strat_fold != test_fold)]
    X_te = X[np.where(df_annot.strat_fold == test_fold)]
    X_tr = X_tr.transpose(0, 2, 1)
    X_te = X_te.transpose(0, 2, 1)

    _create_train_test_files(segmentation, sampling_rate, target,
                             X_tr, y_tr, X_path, y_path, X_te, 
                             y_te, X_te_path, y_te_path)
    
    return _load_processed_data(processed_path, X_name, y_name, X_te_name, y_te_name)

def _load_processed_data(processed_path, X_name, y_name, X_te_name, y_te_name):
    """
    Load processed data from the specified file paths.
    Parameters:
        X_path (str): File path for the input data.
        y_path (str): File path for the target labels.
        X_te_path (str): File path for the test input data.
        y_te_path (str): File path for the test target labels.
    Returns:
        tuple: A tuple containing the loaded input data, target labels, test input data, and test target labels.
    """
    X_path = Path(processed_path, X_name)
    y_path = Path(processed_path, y_name)
    X_te_path = Path(processed_path, X_te_name)
    y_te_path = Path(processed_path, y_te_name)

    try:
        X_tr = np.load(X_path)
        y_tr = np.load(y_path)
        X_te = np.load(X_te_path)
        y_te = np.load(y_te_path)
    except:
        raise ValueError("Processed data not found.")

    return X_tr, y_tr, X_te, y_te

def _load_targets(X, target):
    """
    Helper function to load the targets

    Args:
        X (ndarray): The input data.
        target (str): The target type.

    Returns:
        ndarray: The target variable.
    """
    if target == "range":
        def _compute_range(x):
            return jnp.max(x) - jnp.min(x)
        targets = jax.vmap(_compute_range)(X)
    elif target == "max":
        targets = jax.vmap(jnp.max)(X)
    elif target == "mean":
        targets = jax.vmap(jnp.mean)(X)
    elif target == "min-max-order":
        def _compute_min_max_order(x):
            min_idx, max_idx = jnp.argmin(x), jnp.argmax(x)
            return (min_idx < max_idx).astype(jnp.float32)
        targets = jax.vmap(_compute_min_max_order)(X)
    else:
        raise ValueError(f"Unknown target: {target}")

    return targets
    
def _create_train_test_files(segmentation, sampling_rate, target, X_tr, y_tr, 
                             X_path, y_path, X_te, y_te, X_te_path, y_te_path):
    """
    Create train and test files for machine learning model.
    Args:
        segmentation (bool): Flag indicating whether to perform segmentation.
        sampling_rate (int): Sampling rate of the ECG signal.
        target (str): Target variable for the machine learning model.
        X_tr (ndarray): Training data.
        y_tr (ndarray): Training labels.
        X_path (str): File path to save training data.
        y_path (str): File path to save training labels.
        X_te (ndarray): Test data.
        y_te (ndarray): Test labels.
        X_te_path (str): File path to save test data.
        y_te_path (str): File path to save test labels.
    Returns:
        None
    """
   
    for XX, yy, XX_path, yy_path in \
        [(X_tr, y_tr, X_path, y_path), (X_te, y_te, X_te_path, y_te_path)]:
        if segmentation:
            beats, targets = [], []
            for i, signal in enumerate(tqdm.tqdm(XX, desc="Segmenting ECGs")):
                try:
                    curr_beats = segment_and_filter_ecg(signal, sampling_rate)
                    beats.extend(curr_beats)
                    if yy is not None:
                        targets.extend([yy[i]] * len(curr_beats))
                    else:
                        curr_targets = _load_targets(curr_beats, target)
                        targets.extend(curr_targets)
                except:
                    continue
            X_curr, y_curr = np.array(beats), np.array(targets)
            print(X_curr.shape, y_curr.shape)
        else:
            X_curr, y_curr = XX, yy
        np.save(XX_path, X_curr)
        np.save(yy_path, y_curr)