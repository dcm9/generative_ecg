import argparse
from pathlib import Path

import flax.linen as nn
from flax.training import train_state, orbax_utils
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
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
from mlscience_ekgs.settings import (
    disc_path,
    ecg_path,
    gen_path,
)


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


def naive_morph(x, params, apply_fn, pred_fn, n_steps=100, lr=5e-2):
    _pred_fn = lambda x: apply_fn(params, x)[0]
    x_curr = x.copy()
    xs = [x_curr]
    pbar = tqdm(range(n_steps), desc=f"Prediction: {pred_fn(x_curr):.2f}")
    for _ in pbar:
        x_delta = jax.grad(_pred_fn)(x_curr)
        x_curr = x_curr + lr * x_delta
        xs.append(x_curr)
        pbar.set_description(
            f"Prediction: {pred_fn(x_curr):.2f}"
        )
    
    return xs
    

def get_latent_var(x, result, key):
    mu, sigmasq = result["apply_fn_enc"](
        result["params_enc"], x
    )
    z_pred = gaussian_sample(key, mu, sigmasq)

    return z_pred


def subspace_morph(x, params, apply_fn, pred_fn, result, n_steps=100, lr=5e-2):
    _pred_fn = lambda x: apply_fn(params, x)[0]
    decode_fn = lambda z: result["apply_fn_dec"](
        result["params_dec"], z
    ).reshape(x.shape)
    x_curr = x.copy()
    z_curr = get_latent_var(x_curr, result, jr.PRNGKey(0))
    xs = [x_curr]
    pbar = tqdm(range(n_steps), desc=f"Prediction: {pred_fn(x_curr):.2f}")
    for _ in pbar:
        _pred_fn_induced = lambda z: _pred_fn(decode_fn(z))
        z_delta = jax.grad(_pred_fn_induced)(z_curr)
        z_curr = z_curr + lr * z_delta
        x_curr = decode_fn(z_curr)
        xs.append(x_curr)
        pbar.set_description(
            f"Prediction: {pred_fn(x_curr):.2f}"
        )
    
    return xs


def main(args):
    # Load dataset
    X_tr, *_ = load_dataset(
        args.dataset, args.beat_segment, args.n_channels, target=args.target
    )
    ecg_dir = Path(ecg_path, args.dataset, args.target)
    
    # Load discriminative and generative model
    target_dir = Path(disc_path, args.dataset, args.target)
    gen_result_path = Path(gen_path, args.dataset, args.gen_model)
    if args.beat_segment:
        ecg_dir = Path(ecg_dir, f"bs_ch{args.n_channels}")
        ckpt_dir = Path(target_dir, f"cnn_bs_{args.n_channels}_ckpt")
        gen_ckpt_dir = Path(gen_result_path, f"bs_ch{args.n_channels}",
                            f"beta1_{args.beta1_scheduler}", 
                            f"beta1_{args.beta1}", f"bs_{args.n_channels}_ckpt")
    else:
        ecg_dir = Path(ecg_dir, f"ch{args.n_channels}")
        ckpt_dir = Path(target_dir, f"cnn_{args.n_channels}_ckpt")
        gen_ckpt_dir = Path(gen_result_path, f"ch{args.n_channels}",
                            f"beta1_{args.beta1_scheduler}", 
                            f"beta1_{args.beta1}", f"{args.n_channels}_ckpt")
    ecg_dir.mkdir(parents=True, exist_ok=True)
    state, apply_fn = create_cnn_train_state(X_tr)
    ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
    state = ckptr.restore(ckpt_dir, state)
    
    kwargs = None
    if args.target in ("age", "range", "max", "mean"):
        pred_fn = lambda x: apply_fn(state.params, x).item()
        if args.target == "range":
            kwargs = {"Range": lambda x: jnp.max(x) - jnp.min(x)}
        elif args.target == "max":
            kwargs = {"Max": lambda x: jnp.max(x)}
        elif args.target == "mean":
            kwargs = {"Mean": lambda x: jnp.mean(x)}
    elif args.target in ("sex", "min-max-order"):
        pred_fn = lambda x: jax.nn.sigmoid(apply_fn(state.params, x)).item()
        if args.target == "min-max-order":
            kwargs = {"Min idx": lambda x: jnp.argmin(x), 
                      "Max idx": lambda x: jnp.argmax(x)}
    
    # load generative model
    result = load_model(X_tr, gen_ckpt_dir, args)
    
    # Morph
    key = jr.PRNGKey(args.seed)
    idx = jr.choice(key, len(X_tr), shape=(args.n_ecgs,), replace=False)
    X_m = X_tr[idx]
    for i, x in enumerate(X_m):
        pred = pred_fn(x)
        print(f"Morphing ECG {i}/{args.n_ecgs} (pred: {pred})...")
        
        # Naive morph
        xs_nm_p = naive_morph(x, state.params, apply_fn, pred_fn,
                              n_steps=args.n_steps, lr=args.lr_nm)
        xs_nm_m = naive_morph(x, state.params, apply_fn, pred_fn,
                              n_steps=args.n_steps, lr=-args.lr_nm)
        x_nm_p = xs_nm_p[-1]
        x_nm_m = xs_nm_m[-1]
        
        # Subspace morph
        xs_sm_p = subspace_morph(x, state.params, apply_fn, pred_fn, 
                                 result, n_steps=args.n_steps, lr=args.lr_sm)
        xs_sm_m = subspace_morph(x, state.params, apply_fn, pred_fn, 
                                 result, n_steps=args.n_steps, lr=-args.lr_sm)
        x_sm_p = xs_sm_p[-1]
        x_sm_m = xs_sm_m[-1]
        
        for (xx, x_str) in (
            (x, ""), (x_nm_p, "_nm_p"), (x_nm_m, "_nm_m"), 
            (x_sm_p, "_sm_p"), (x_sm_m, "_sm_m"),
        ):
            if x_str == "":
                title = f"Original ECG {i}"
            else:
                title = f"Morphed ECG {i}"
            fig, _ = plot_ecg(
                xx, CHANNELS, args.n_channels, (6, args.n_channels+1),
                title=title, ylim=(x.min()-0.1, x.max()+0.1), **kwargs
            )
            fig.savefig(Path(ecg_dir, f"ecg_{i}{x_str}.png"))
            plt.close(fig)   
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=0) # random seed
    
    parser.add_argument("--dataset", type=str, default="ptb-xl",
                        choices=["mimic-iv", "ptb-xl"],)
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--target", type=str, default="sex")
    parser.add_argument("--n_channels", type=int, default=3) # number of channels to use
    
    parser.add_argument("--gen_model", type=str, default="dr-vae",
                        choices=["dr-vae", "baseline", "dsm", "real"],)
    
    
    # Specify model parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512) # latent dim
    parser.add_argument("--beta1", type=float, default=0.01) # KL-div reg. weight
    parser.add_argument("--beta2", type=float, default=0.0) # disc. reg. weight
    parser.add_argument("--hidden_width", type=int, default=100) # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4) # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7) # initial learning rate
    parser.add_argument("--lr_peak", type=float, default=1e-4) # peak learning rate
    parser.add_argument("--lr_end", type=float, default=1e-7) # end learning rate
    parser.add_argument("--beta1_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine",
                                 "warmup_cosine", "cyclical"],)
    
    # Morphing parameters
    parser.add_argument("--lr_nm", type=float, default=1e-3) # naive morphing
    parser.add_argument("--lr_sm", type=float, default=5e-2) # subspace morphing
    parser.add_argument("--n_steps", type=int, default=300) # number of steps to morph
    
    parser.add_argument("--n_ecgs", type=int, default=10) # number of epochs to morph
    
    args = parser.parse_args()
    
    main(args)