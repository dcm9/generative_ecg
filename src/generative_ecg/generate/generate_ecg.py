from pathlib import Path

import jax.numpy
import jax.random
import matplotlib.pyplot 
import tqdm

from .plot_utils import plot_ecg, find_closest_real_ecg, plot_gaussian_noise
from ..models.math_utils import OMAT

CHANNELS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 
            'V4', 'V5', 'V6']

def generate_ecgs(X, result, gen_params, save_dir):
    key = jax.random.PRNGKey(gen_params["seed"])

    for i in tqdm.trange(gen_params["n_ecgs"]):
        key1, key2, key3 = jax.random.split(key, 3)
        mu = plot_gaussian_noise(
            result["mu_mean"], result["mu_std"], key1, shape=(gen_params["z_dim"],)
        )
        sigmasq = plot_gaussian_noise(
            result["sigmasq_mean"], result["sigmasq_std"], key2, shape=(gen_params["z_dim"],)
        )
        z = plot_gaussian_noise(
            mu, jax.numpy.sqrt(sigmasq), key3, shape=(gen_params["z_dim"],)
        )

        x = result["apply_fn_dec"](result["params_dec"], z).reshape(X.shape[1], -1)

        if gen_params["processed"]:
            x = OMAT @ x

        fig, _ = plot_ecg(x, CHANNELS, gen_params["n_channels"], 
                            (6, gen_params["n_channels"]+1), 
                            gen_params['std'], 
                            title=f"{gen_params['title']} {i+1}",
                            ylim=gen_params["ylim"])
        fig.savefig(Path(save_dir, f"ecg_{i+1}.png"))
        matplotlib.pyplot.close("all")