import tqdm
import jax
import jax.numpy

from pathlib import Path

from .process_data import load_data
from ..models.math_utils import compute_linproj_residual
from .segment_ecg import segment_and_filter_ecg

def load_signals(filepath, target='age'):
    if filepath is None:
        raise ValueError("ECG filepath must be provided.")

    df_annot = pandas.read_csv(
        Path(filepath, "ptbxl_database.csv"), 
        index_col="ecg_id"
    )

    df_annot.scp_codes = df_annot.scp_codes.apply(ast.literal_eval)
    # DEFAULT 500
    file_paths = list(df_annot.filename_hr)

    data = []
    for i, f in enumerate(tqdm.tqdm(file_paths, desc="Loading data from records")):
        data.append(wfdb.rdsamp(Path(filepath, f)))

    X = numpy.array([signal for signal, _ in data])
    y = df_annot[target].values

    return X,y

# def load_processed_dataset(ecg_filepath=None, beat_segment=False, target="age"):
def load_unprocessed_dataset(X, y, targets, beat_segment=False, n_channels=12, 
                             x_len=400, atol=1e-6):
    #TODO: remove n_channels, put in examples
    X = X[:, :n_channels, :]

    if beat_segment:
        sampling_rate = 500
        beats, targets = jax.numpy.array([]), jax.numpy.array([])
        for i, signal in enumerate(tqdm.tqdm(X, desc="Segmenting ECGs")):
            try:
                # Note: segment_and_filter_ecg may need to be updated to return JAX arrays
                curr_beats = segment_and_filter_ecg(signal, sampling_rate)
                beats = jax.numpy.concatenate((beats, curr_beats))
                if y is not None:
                    targets = jax.numpy.concatenate((targets, jax.numpy.array([y[i]] * len(curr_beats))))
                # else:
                #     curr_targets = _load_targets(curr_beats, target)
                #     targets.extend(curr_targets)
            except:
                continue
        X_curr, y_curr = beats, targets
    else:
        X = X[:, :, :x_len]
        X_curr, y_curr = X, y

    X_proc, y_proc = jax.numpy.array([]), jax.numpy.array([])

    processed_data = []
    processed_labels = []
    
    for i, x in enumerate(tqdm.tqdm(X_curr, desc="Processing by linproj")):
        x_transpose = jax.numpy.transpose(x, (1, 0))
        sol, res = jax.vmap(compute_linproj_residual)(x_transpose)
        if jax.numpy.mean(res) < atol:
            processed_data.append(sol.T)
            processed_labels.append(y_curr[i])
    
    if processed_data:
        X_proc = jax.numpy.stack(processed_data)
        y_proc = jax.numpy.array(processed_labels)

    return X_proc, y_proc

def project(x_beats,y_beats,tol=1e-6):
    X_proj, y_proj = jax.numpy.array([]), jax.numpy.array([])

    processed_data = []
    processed_labels = []

    for i, x in enumerate(tqdm.tqdm(x_beats, desc="Processing by linproj")):
        x_transpose = jax.numpy.transpose(x, (1, 0))
        sol, res = jax.vmap(compute_linproj_residual)(x_transpose)
        if jax.numpy.mean(res) < tol:
            processed_data.append(sol.T)
            processed_labels.append(y_beats[i])
    
    if processed_data:
        X_proj = jax.numpy.stack(processed_data)
        y_proj = jax.numpy.array(processed_labels)

    return X_proj, y_proj
