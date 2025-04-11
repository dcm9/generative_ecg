import tqdm
import jax
import jax.numpy
import pandas
import ast
import wfdb

from pathlib import Path

from ..models.math_utils import compute_linproj_residual
from .segment_ecg import segment_and_filter_ecg

def load_signals(filepath, sampling_rate, target='age'):
    if filepath is None:
        raise ValueError("ECG filepath must be provided.")

    df_annot = pandas.read_csv(
        Path(filepath, "ptbxl_database.csv"), 
        index_col="ecg_id"
    )

    df_annot.scp_codes = df_annot.scp_codes.apply(ast.literal_eval)
    # DEFAULT 500
    if sampling_rate == 500:
        file_paths = list(df_annot.filename_hr)

    elif sampling_rate == 100:
        file_paths = list(df_annot.filename_lr)

    data = []
    for i, f in enumerate(tqdm.tqdm(file_paths, desc="Loading data from records")):
        data.append(wfdb.rdsamp(Path(filepath, f)))

    X = jax.numpy.array([signal for signal, _ in data]).transpose(0, 2, 1)
    y = df_annot[target].values

    return X,y

def project(x_beats,y_beats,tol=1e-6):
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
