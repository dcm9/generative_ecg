from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tqdm

from .plot_utils import plot_ecg, find_closest_real_ecg
from ..models.math_utils import OMAT

CHANNELS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 
            'V4', 'V5', 'V6']

def generate_and_save_ecgs(X, result, gen_result_path, **kwargs):
    
    generate_params = {
        "z_dim": 512,
        "seed": 0,
        "n_ecgs": 5,
        "processed": False,
        "find_closest_real": False,
        "n_channels": 12,
    }

    for key in generate_params:
        if key not in kwargs.keys():
            kwargs[key] = generate_params[key]

    gen_result_path = Path(gen_result_path, "generated_ecgs")
    gen_result_path.mkdir(parents=True, exist_ok=True)
    key = jr.PRNGKey(kwargs["seed"])
    key, subkey = jr.split(key)

    fn_dec, params_dec = result["apply_fn_dec"], result["params_dec"]
    mu_mean, mu_std = result["mu_mean"], result["mu_std"]
    sigmasq_mean, sigmasq_std = \
        result["sigmasq_mean"], result["sigmasq_std"]
    ecgs, rmses = [], []
    for i in tqdm.trange(kwargs["n_ecgs"]):
        key1, key2, key3, key  = jr.split(key, 4)
        mu_curr = mu_mean + mu_std*jr.normal(key1, shape=(kwargs["z_dim"],))
        sigmasq_curr = sigmasq_mean + \
            sigmasq_std*jr.normal(key2, shape=(kwargs["z_dim"],))
        z = mu_curr + \
            jnp.sqrt(sigmasq_curr)*jr.normal(key3, shape=(kwargs["z_dim"],))
        x = fn_dec(params_dec, z).reshape(X.shape[1], -1)
        if kwargs["processed"]:
            x = OMAT @ x
        ecgs.append(x)
        fig, _ = plot_ecg(x, CHANNELS, kwargs["n_channels"], 
                            (6, kwargs["n_channels"]+1), 
                            title=f"ECG {i+1}")
        fig.savefig(Path(gen_result_path, f"ecg_{i+1}.png"))
        plt.close("all")
        
        if kwargs["find_closest_real"]:
            ecg_c, dist = find_closest_real_ecg(X, x, kwargs["processed"])
            rmses.append(dist)
            fig, _ = plot_ecg(
                ecg_c, CHANNELS, kwargs["n_channels"], 
                (6, kwargs["n_channels"]+1),
                title=f"Closest real ECG {i+1}, dist: {dist:.3f}"
            )
            fig.savefig(Path(gen_result_path, f"ecg_{i+1}_closest.png"))
        plt.close("all")
    if rmses:
        rmses = jnp.array(rmses)
        print(f"\nClosest real ECG:")
        print(f"\tmean: {jnp.mean(rmses):.3f}")
        print(f"\tstd: {jnp.std(rmses):.3f}")
        print(f"\tmax: {jnp.max(rmses):.3f}")
        print(f"\tmin: {jnp.min(rmses):.3f}")