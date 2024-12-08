import tqdm
import jax.numpy as jnp
import jax

from pathlib import Path

from .process_data import load_data
from ..models.math_utils import compute_linproj_residual

def load_dataset(ecg_filepath=None, beat_segment=False, processed=False, 
                 n_channels=12, verbose=True, target="age", x_len=400, atol=1e-6):
    if beat_segment:
        sampling_rate = 500
    else:
        sampling_rate = 100

    X_tr, y_tr, X_te, y_te = load_data(
        ecg_filepath=ecg_filepath,
        segmentation=beat_segment, 
        sampling_rate=sampling_rate,
        target=target,
        processed=processed,
    )
    # Take only the first n channels
    if not processed:
        X_tr = X_tr[:, :n_channels, :]
        X_te = X_te[:, :n_channels, :]
        if not beat_segment:
            X_tr = X_tr[:, :, :x_len]
            X_te = X_te[:, :, :x_len]
    
    if verbose:
        print(
            f"Loaded dataset from {ecg_filepath} with shape X_tr: {X_tr.shape},"
            f"y_tr: {y_tr.shape}, X_te: {X_te.shape}, y_te: {y_te.shape}"
        )

    if processed:
        return X_tr, y_tr, X_te, y_te, target

    processed_path = Path(ecg_filepath, "processed")
    X_proc_tr, X_proc_te = [], []
    y_proc_tr, y_proc_te = [], []
    
    # Process and filter dataset
    for X, X_proc, y, y_proc in zip(
        [X_tr, X_te], [X_proc_tr, X_proc_te], 
        [y_tr, y_te], [y_proc_tr, y_proc_te]
    ):
        for i, x in enumerate(tqdm.tqdm(X, desc="Processing by linproj")):
            x_tr = jnp.transpose(x, (1, 0))
            sol, res = jax.vmap(compute_linproj_residual)(x_tr)
            if jnp.mean(res) < atol:
                X_proc.append(sol.T)
                y_proc.append(y[i])
    
    # Save the processed and filtered dataset
    if beat_segment:
        X_name = "X_seg.npy"
        y_name = f"y_{target}_seg.npy"
        X_te_name = "X_te_seg.npy"
        y_te_name = f"y_te_{target}_seg.npy"
    else:
        X_name = "X.npy"
        y_name = f"y_{target}.npy"
        X_te_name = "X_te.npy"
        y_te_name = f"y_te_{target}.npy"
    jnp.save(Path(processed_path, X_name), jnp.array(X_proc_tr))
    jnp.save(Path(processed_path, y_name), jnp.array(y_proc_tr))
    jnp.save(Path(processed_path, X_te_name), jnp.array(X_proc_te))
    jnp.save(Path(processed_path, y_te_name), jnp.array(y_proc_te))    
    
    return X_proc_tr, y_proc_tr, X_proc_te, y_proc_te, target