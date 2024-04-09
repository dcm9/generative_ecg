import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm

from mlscience_ekgs.Code.experiments.s02_train_and_generate_ecgs import (
    load_dataset,
    load_model,
    generate_vae_ecg,
)
from mlscience_ekgs.settings import gen_path, residual_path
from mlscience_ekgs.Code.src.s02_dipole_model import OMAT


def compute_linproj_residual(x):
    """Compute the residual of the linear projection.

    Args:
        x (jnp.array): 12d lead observation.

    Returns:
        res (jnp.array): residual of the linear projection.
    """    
    sol, res, *_ = jnp.linalg.lstsq(OMAT, x, rcond=None)
    
    return sol, res


def check_linproj_constraint(x, atol=1e-6):
    """Check the linear projection constraint.

    Args:
        x (jnp.array): 12d lead observation.
        atol (float): absolute tolerance.

    Returns:
        is_valid (bool): whether the linear projection constraint is satisfied.
    """
    _, res = compute_linproj_residual(x)
    is_valid = jnp.abs(res) < atol
    
    return is_valid


def main(args):
    X_tr, *_ = load_dataset(args.dataset, args.beat_segment,
                            args.processed, 12)
    residual_np_path = Path(
        residual_path, f"{args.dataset}_{args.gen_model}_residuals.npy"
    )
    if residual_np_path.exists():
        residuals = jnp.load(residual_np_path)
    else:
        residuals = []
        if args.gen_model == "real":
            for x in tqdm.tqdm(X_tr):
                if args.processed:
                    x = OMAT @ x
                x_curr = jnp.transpose(x, (1, 0))
                _, res = jax.vmap(compute_linproj_residual)(x_curr)
                residuals.append(res)
        elif args.gen_model == "dr-vae":
            gen_result_path = Path(gen_path, args.dataset, args.gen_model)
            if args.beat_segment:
                gen_ckpt_dir = Path(
                    gen_result_path, f"bs_ch12",
                    f"beta1_{args.beta1_scheduler}", f"beta1_{args.beta1}",
                    "bs_12_ckpt"
                )
            else:
                gen_ckpt_dir = Path(
                    gen_result_path, f"ch12",
                    f"beta1_{args.beta1_scheduler}", f"beta1_{args.beta1}",
                    "12_ckpt"
                )
            result = load_model(X_tr, gen_ckpt_dir, args)
            for i in tqdm.tqdm(range(args.n_examples)):
                x = generate_vae_ecg(result, 12, i)
                if args.processed:
                    x = OMAT @ x
                x_curr = jnp.transpose(x, (1, 0))
                _, res = jax.vmap(compute_linproj_residual)(x_curr)
                residuals.append(res)
        else:
            raise ValueError(f"Invalid generative model: {args.gen_model}")
        residuals = jnp.array(residuals)
        jnp.save(residual_np_path, residuals)
    if args.compute_type == "residual":
        # Exclude max 1% residuals in terms of mean across all axes
        res_per_ex = jnp.mean(residuals, axis=1).squeeze()
        # Print quartiles
        res_per_ex_sorted = jnp.sort(res_per_ex)
        print(f"Top 10: {res_per_ex_sorted[-10:]}")
        residuals = residuals[
            res_per_ex < jnp.percentile(res_per_ex, 99, axis=0)
        ]
        res_per_ts = jnp.mean(residuals, axis=0).squeeze()
        fig, ax = plt.subplots()
        ax.plot(res_per_ts)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean residual")
        ax.set_title(f"Mean residual per time step ({args.gen_model})")
        plt.savefig(
            Path(residual_path, 
                 f"{args.dataset}_{args.gen_model}_residual_ts.png")
        )

        # Plot the histogram of residuals
        res_per_ex = jnp.mean(residuals, axis=1).squeeze()
        fig, ax = plt.subplots()
        ax.hist(res_per_ex, bins=50, density=True, alpha=0.6, color="g")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of residuals ({args.gen_model})")
        plt.savefig(
            Path(residual_path, 
                 f"{args.dataset}_{args.gen_model}_residual_hist.png")
        )
    elif args.compute_type == "fraction":
        _compute_fracs = lambda res: jnp.mean(res < args.atol)
        fracs = jax.vmap(_compute_fracs)(residuals)
        n = len(fracs)
        print(f"For tolerance {args.atol}:")
        print(f"\tTotal examples: {n} ({n*100/n:.1f}%)")
        print(f"\tValid linear projections: {jnp.sum(fracs == 1.0)} "
            f"({jnp.sum(fracs == 1.0)*100/n:.1f}%)")
        for i in range(10, 0, -1):
            num = jnp.sum(fracs <= i/10) - jnp.sum(fracs < (i-1)/10)
            print(f"\t{(i-1)*10}-{i*10}% Valid : {num} ({num*100/n:.1f}%)")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Specify dataset
    parser.add_argument("--dataset", type=str, default="ptb-xl",
                        choices=["mimic-iv", "ptb-xl"],)
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--processed", action="store_true") # use processed dataset
    
    # Use generative model
    parser.add_argument("--gen_model", type=str, default="real",
                        choices=["real", "dr-vae"])
    # For generative model, specify the number of ECGs to generate
    parser.add_argument("--n_examples", default=10_000)
    
    # Compute type
    parser.add_argument("--compute_type", type=str, default="residual",
                        choices=["residual", "fraction"])
    
    # Constraing tolerance
    parser.add_argument("--atol", type=float, default=1e-6)
    
    # Specify model parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512) # latent dim
    parser.add_argument("--hidden_width", type=int, default=100) # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4) # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7) # initial learning rate
    parser.add_argument("--beta1", type=float, default=0.01) # KL-div reg. weight
    parser.add_argument("--beta2", type=float, default=0.0) # disc. reg. weight
    parser.add_argument("--target", type=str, default="sex") # target for discriminator
    parser.add_argument("--beta1_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine",
                                 "warmup_cosine", "cyclical"],)
    
    parser.add_argument("--seed", type=int, default=0) # random seed
    
    args = parser.parse_args()
    
    main(args)