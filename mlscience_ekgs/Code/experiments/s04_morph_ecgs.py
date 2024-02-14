import argparse
from pathlib import Path

import flax.linen as nn
from flax.training import train_state, orbax_utils
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as orbax_ckpt
from tqdm import tqdm

from mlscience_ekgs.Code.experiments.s02_train_and_generate_ecgs import (
    CHANNELS,
    load_dataset,
    load_model
)
from mlscience_ekgs.Code.experiments.s03_train_discriminator import CNN
from mlscience_ekgs.Code.src.s03_dr_vae import gaussian_sample
from mlscience_ekgs.Code.src.s06_utils import plot_ecg
from mlscience_ekgs.settings import result_path


def create_cnn_train_state(X, key=0):
    """Creates initial `TrainState`."""
    # Initialize NN model
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    nn_model = CNN(output_dim=1)
    params = nn_model.init(key, X[0])
    apply_fn = lambda p, x: nn_model.apply(p, x)

    # Create trainstate
    tx = optax.adam(1e-3)
    opt_state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=tx
    )
    
    return opt_state, apply_fn


def naive_morph(x, params, apply_fn, n_steps=100, lr=5e-2):
    _pred_fn = lambda x: apply_fn(params, x)[0]
    x_curr = x.copy()
    xs = [x_curr]
    pbar = tqdm(range(n_steps), desc="Prediction: "
                f"{jax.nn.sigmoid(_pred_fn(x_curr)).item():.2f}%")
    for _ in pbar:
        x_delta = jax.grad(_pred_fn)(x_curr)
        x_curr = x_curr + lr * x_delta
        xs.append(x_curr)
        pbar.set_description(
            f"Prediction: {jax.nn.sigmoid(_pred_fn(x_curr)).item():.2f}%"
        )
    
    return xs
    

def get_latent_var(x, result, key):
    mu, sigmasq = result["apply_fn_enc"](
        result["params_enc"], x
    )
    z_pred = gaussian_sample(key, mu, sigmasq)

    return z_pred


def subspace_morph(x, params, apply_fn, result, n_steps=100, lr=5e-2):
    _pred_fn = lambda x: apply_fn(params, x)[0]
    decode_fn = lambda z: result["apply_fn_dec"](
        result["params_dec"], z
    ).reshape(x.shape)
    x_curr = x.copy()
    z_curr = get_latent_var(x_curr, result, jr.PRNGKey(0))
    xs = [x_curr]
    pbar = tqdm(range(n_steps), desc="Prediction: "
                f"{jax.nn.sigmoid(_pred_fn(x_curr)).item():.2f}%")
    for _ in pbar:
        _pred_fn_induced = lambda z: _pred_fn(decode_fn(z))
        z_delta = jax.grad(_pred_fn_induced)(z_curr)
        z_curr = z_curr + lr * z_delta
        x_curr = decode_fn(z_curr)
        xs.append(x_curr)
        pbar.set_description(
            f"Prediction: {jax.nn.sigmoid(_pred_fn(x_curr)).item():.2f}%"
        )
    
    return xs


def main(args):
    # Load dataset
    X, y = load_dataset("ptb-xl", args.beat_segment, None, args.n_channels,
                        target=args.target)
    assert args.target == "sex"
    
    # Load discriminative model
    target_dir = Path(result_path, f"{args.target}")
    if args.beat_segment:
        ckpt_dir = Path(target_dir, f"cnn_bs_{args.n_channels}_ckpt")
    else:
        ckpt_dir = Path(target_dir, f"cnn_{args.n_channels}_ckpt")
    state, apply_fn = create_cnn_train_state(X)
    ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
    state = ckptr.restore(ckpt_dir, state)
    
    # load generative model
    drvae_result_path = Path(result_path, "s03_ptbxl", "s01_dr_vae")
    gen_ckpt_dir = Path(drvae_result_path, "dr_vae_ckpt")
    result = load_model(X, gen_ckpt_dir)
    
    # Morph
    key = jr.PRNGKey(args.seed)
    idx = jr.choice(key, len(X), shape=(args.n_ecgs,), replace=False)
    X_m = X[idx]
    for i, x in enumerate(X_m):
        pred = jax.nn.sigmoid(apply_fn(state.params, x)).item()
        print(f"Morphing ECG {i+1}/{args.n_ecgs} (pred: {pred})...")
        lr = args.lr if pred < 0.5 else -args.lr
        
        # Naive morph
        xs_nm = naive_morph(x, state.params, apply_fn, lr=lr)
        x_nm = xs_nm[-1]
        
        # Subspace morph
        xs_sm = subspace_morph(x, state.params, apply_fn, result, lr=lr)
        x_sm = xs_sm[-1]
        
        ecg_dir = Path(target_dir, "morphed_ecgs")
        ecg_dir.mkdir(exist_ok=True)
        
        fig, _ = plot_ecg(x, CHANNELS, args.n_channels,
                          (2*args.n_channels, 6))
        fig.savefig(Path(ecg_dir, f"ecg_{i}.png"))
        
        fig, _ = plot_ecg(x_nm, CHANNELS, args.n_channels,
                          (2*args.n_channels, 6))
        fig.savefig(Path(ecg_dir, f"ecg_nm_{i}.png"))
        
        fig, _ = plot_ecg(x_sm, CHANNELS, args.n_channels,
                          (2*args.n_channels, 6))
        fig.savefig(Path(ecg_dir, f"ecg_sm_{i}_steps.png"))    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=0) # random seed
    
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--target", type=str, default="sex")
    parser.add_argument("--n_channels", type=int, default=3) # number of channels to use
    
    parser.add_argument("--lr", type=float, default=1e-2) # learning rate
    parser.add_argument("--n_ecgs", type=int, default=10) # number of epochs to morph
    
    args = parser.parse_args()
    
    main(args)