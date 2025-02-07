import tqdm
import torch
import jax
import jax.numpy

from pathlib import Path

from .process_data import load_data
from ..models.math_utils import compute_linproj_residual
from .segment_ecg import segment_and_filter_ecg

# def load_processed_dataset(ecg_filepath=None, beat_segment=False, target="age"):
def load_unprocessed_dataset(X, y, targets, beat_segment=False, n_channels=12, 
                             x_len=400, atol=1e-6):
    #TODO: remove n_channels, put in examples
    X = X[:, :n_channels, :]

    if beat_segment:
        sampling_rate = 500
        beats, targets = torch.tensor([]), torch.tensor([])
        for i, signal in enumerate(tqdm.tqdm(X, desc="Segmenting ECGs")):
            try:
                curr_beats = segment_and_filter_ecg(signal, sampling_rate)
                beats = torch.cat((beats, curr_beats))
                if y is not None:
                    targets = torch.cat((targets, torch.tensor([y[i]] * len(curr_beats))))
                # else:
                #     curr_targets = _load_targets(curr_beats, target)
                #     targets.extend(curr_targets)
            except:
                continue
        X_curr, y_curr = beats, targets
    else:
        X = X[:, :, :x_len]
        X_curr, y_curr = X, y

    X_proc, y_proc = torch.tensor([]), torch.tensor([])

    # before, the first layer of loop ensures that the residual is computed for both training and test data
    # t
    # for X, X_proc, y, y_proc in zip(
    #     [X_tr, X_te], [X_proc_tr, X_proc_te], 
    #     [y_tr, y_te], [y_proc_tr, y_proc_te]
    # ):

    for i, x in enumerate(tqdm.tqdm(X_curr, desc="Processing by linproj")):
        x_transpose = jax.numpy.transpose(x, (1, 0))
        sol, res = jax.vmap(compute_linproj_residual)(x_transpose)
        if jax.numpy.mean(res) < atol:
            X_proc = torch.cat((X_proc, sol.T))
            y_proc = torch.cat((y_proc, y_curr[i]))

    return X_proc, y_proc
