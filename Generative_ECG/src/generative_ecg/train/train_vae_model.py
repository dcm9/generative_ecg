from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as orbax_ckpt

from Generative_ECG.src.generative_ecg.train.train_utils.cnn_utils import create_cnn_train_state
from Generative_ECG.src.generative_ecg.train.train_utils.dr_vae_utils import train_dr_vae
from Generative_ECG.src.generative_ecg.models.nn_models import CNN
from Generative_ECG.src.generative_ecg.models.math_utils import OMAT


def train_vae(X, ckpt_dir, gen_ckpt_dir, **kwargs):
    # Load discriminative model

    model_params = {
        "beta1": 1.0,
        "beta2": 0.0,
        "z_dim": 512,
        "hidden_width": 100,
        "hidden_depth": 4,
        "lr_init": 1e-7,
        "lr_peak": 1e-4,
        "lr_end": 1e-7,
        "beta1_scheduler": "constant",
        "target": "age",
    }

    data_params = {
        "n_channels": 12,
        "beat_segment": False,
        "processed": False,
    }

    train_params = {
        "seed": 0,
        "batch_size": 512,
        "n_epochs": 100,
    }

    for dict in [model_params, data_params, train_params]:
        for key in dict:
            if key not in kwargs.keys():
                kwargs[key] = dict[key]

    state_disc = create_cnn_train_state(X)
    ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
    state_disc = ckptr.restore(
        ckpt_dir, item=state_disc
    )
    model = CNN(output_dim=1)
    vae_pred_fn = lambda x: model.apply(state_disc.params, x)
    
    gen_ckpt_dir.mkdir(parents=True, exist_ok=True)
    result = train_dr_vae(vae_pred_fn, X, kwargs["beta1"], kwargs["beta2"],
                            kwargs["z_dim"], kwargs["seed"], kwargs["n_epochs"],
                            kwargs["batch_size"], kwargs["hidden_width"],
                            kwargs["hidden_depth"], kwargs["lr_init"],
                            kwargs["lr_peak"], kwargs["lr_end"],
                            encoder_type="cnn", use_bias=False,
                            beta1_scheduler_type=kwargs["beta1_scheduler"],)

    with open(Path(gen_ckpt_dir, "params_enc.npy"), "wb") as f:
        jnp.save(f, result["params_enc"])
    with open(Path(gen_ckpt_dir, "params_dec.npy"), "wb") as f:
        jnp.save(f, result["params_dec"])
    with open(Path(gen_ckpt_dir, "mu_mean.npy"), "wb") as f:
        jnp.save(f, result["mu_mean"])
    with open(Path(gen_ckpt_dir, "mu_std.npy"), "wb") as f:
        jnp.save(f, result["mu_std"])
    with open(Path(gen_ckpt_dir, "sigmasq_mean.npy"), "wb") as f:
        jnp.save(f, result["sigmasq_mean"])
    with open(Path(gen_ckpt_dir, "sigmasq_std.npy"), "wb") as f:
        jnp.save(f, result["sigmasq_std"])
    with open(Path(gen_ckpt_dir, "losses.npy"), "wb") as f:
        jnp.save(f, result["losses"])
    with open(Path(gen_ckpt_dir, "losses_rec.npy"), "wb") as f:
        jnp.save(f, result["losses_rec"])
    with open(Path(gen_ckpt_dir, "losses_kl.npy"), "wb") as f:
        jnp.save(f, result["losses_kl"])
    with open(Path(gen_ckpt_dir, "losses_dr.npy"), "wb") as f:
        jnp.save(f, result["losses_dr"])

    return result

