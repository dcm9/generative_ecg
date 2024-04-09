import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import tqdm

from mlscience_ekgs.Code.experiments.s02_train_and_generate_ecgs import (
    load_dataset,
)
from mlscience_ekgs.Code.experiments.s05_check_linproj_constraint import (
    compute_linproj_residual,
)
from mlscience_ekgs.settings import ptb_xl_processed_path


def main(args):
    X_tr, y_tr, X_te, y_te, target = load_dataset(
        args.dataset, args.beat_segment, False, 12
    )
    ptb_xl_processed_path.mkdir(parents=True, exist_ok=True)
    X_proc_tr, X_proc_te = [], []
    y_proc_tr, y_proc_te = [], []
    
    # Process and filter dataset
    for X, X_proc, y, y_proc in zip(
        [X_tr, X_te], [X_proc_tr, X_proc_te], 
        [y_tr, y_te], [y_proc_tr, y_proc_te]
    ):
        for i, x in enumerate(tqdm.tqdm(X)):
            x_tr = jnp.transpose(x, (1, 0))
            sol, res = jax.vmap(compute_linproj_residual)(x_tr)
            if jnp.mean(res) < args.atol:
                X_proc.append(sol.T)
                y_proc.append(y[i])
    
    # Save the processed and filtered dataset
    if args.beat_segment:
        X_name = "X_seg.npy"
        y_name = f"y_{target}_seg.npy"
        X_te_name = "X_te_seg.npy"
        y_te_name = f"y_te_{target}_seg.npy"
    else:
        X_name = "X.npy"
        y_name = f"y_{target}.npy"
        X_te_name = "X_te.npy"
        y_te_name = f"y_te_{target}.npy"
    jnp.save(Path(ptb_xl_processed_path, X_name), jnp.array(X_proc_tr))
    jnp.save(Path(ptb_xl_processed_path, y_name), jnp.array(y_proc_tr))
    jnp.save(Path(ptb_xl_processed_path, X_te_name), jnp.array(X_proc_te))
    jnp.save(Path(ptb_xl_processed_path, y_te_name), jnp.array(y_proc_te))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Specify dataset
    parser.add_argument("--dataset", type=str, default="ptb-xl",
                        choices=["mimic-iv", "ptb-xl"],)
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    
    # Filtering average (across timestep) tolerance
    parser.add_argument("--atol", type=float, default=1e-6)
    
    args = parser.parse_args()
    
    main(args)