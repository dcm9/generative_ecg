from pathlib import Path

import jax.numpy
import jax.random
import matplotlib.pyplot 
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
    key = jax.random.PRNGKey(kwargs["seed"])
    key, subkey = jax.random.split(key)

    fn_dec, params_dec = result["apply_fn_dec"], result["params_dec"]
    mu_mean, mu_std = result["mu_mean"], result["mu_std"]
    sigmasq_mean, sigmasq_std = \
        result["sigmasq_mean"], result["sigmasq_std"]
    ecgs, rmses = [], []
    for i in tqdm.trange(kwargs["n_ecgs"]):
        key1, key2, key3, key  = jax.random.split(key, 4)
        mu_curr = mu_mean + mu_std*jax.random.normal(key1, shape=(kwargs["z_dim"],))
        sigmasq_curr = sigmasq_mean + \
            sigmasq_std*jax.random.normal(key2, shape=(kwargs["z_dim"],))
        z = mu_curr + \
            jax.numpy.sqrt(sigmasq_curr)*jax.random.normal(key3, shape=(kwargs["z_dim"],))
        x = fn_dec(params_dec, z).reshape(X.shape[1], -1)
        if kwargs["processed"]:
            x = OMAT @ x
        ecgs.append(x)
        fig, _ = plot_ecg(x, CHANNELS, kwargs["n_channels"], 
                            (6, kwargs["n_channels"]+1), 
                            title=f"ECG {i+1}")
        fig.savefig(Path(gen_result_path, f"ecg_{i+1}.png"))
        matplotlib.pyplot.close("all")
        
        if kwargs["find_closest_real"]:
            ecg_c, dist = find_closest_real_ecg(X, x, kwargs["processed"])
            rmses.append(dist)
            fig, _ = plot_ecg(
                ecg_c, CHANNELS, kwargs["n_channels"], 
                (6, kwargs["n_channels"]+1),
                title=f"Closest real ECG {i+1}, dist: {dist:.3f}"
            )
            fig.savefig(Path(gen_result_path, f"ecg_{i+1}_closest.png"))
        matplotlib.pyplot.close("all")
    if rmses:
        rmses = jax.numpy.array(rmses)
        print(f"\nClosest real ECG:")
        print(f"\tmean: {jax.numpy.mean(rmses):.3f}")
        print(f"\tstd: {jax.numpy.std(rmses):.3f}")
        print(f"\tmax: {jax.numpy.max(rmses):.3f}")
        print(f"\tmin: {jax.numpy.min(rmses):.3f}")