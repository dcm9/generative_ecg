from pathlib import Path

import jax.numpy
import jax.random
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from flax.training import train_state, orbax_utils
import orbax.checkpoint
import optax
import tqdm

from typing import Any, Callable, Dict, Optional, Tuple

from ..models.nn_models import Encoder, Decoder, CNNEncoder
from ..models.loss_utils import binary_loss
from ..train.dr_vae_utils import create_vae_base

def train_vae(
    X: jax.numpy.ndarray,
    y: jax.numpy.ndarray,
    pred_fn: Callable,
    hyp_params: Dict[str, Any],
    lr_schedule: Any,
    ckpt_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a Variational Autoencoder (VAE) using JAX and Optax.

    Args:
        X (jax.numpy.ndarray): Input data array.
        y (jax.numpy.ndarray): Target labels or auxiliary data.
        pred_fn (Callable): Prediction or discriminator function for regularization.
        hyp_params (dict): Dictionary of hyperparameters for the model and training.
        lr_schedule (Any): Learning rate schedule or optimizer.
        ckpt_dir (Optional[str]): Directory to save checkpoints. If None, no checkpointing.

    Returns:
        dict: Dictionary containing latent statistics and trained encoder/decoder parameters.
    """
    model_key = jax.random.PRNGKey(0)
    key_enc, key_dec = jax.random.split(model_key)
    _, *x_dim = X.shape
    x_dim = jax.numpy.array(x_dim)
    n = len(X)

    apply_fn_enc, apply_fn_dec, params_enc, params_dec = create_vae_base(
        X, hyp_params
    )

    params = jax.numpy.array([*params_enc, *params_dec])
    split_idx = len(params_enc)

    # Train state
    n_steps = hyp_params['n_epochs'] * (n // hyp_params['batch_size'])
    optimizer = optax.adam(lr_schedule)
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=optimizer
    )

    # TODO: ALLOW FOR DIFFERENT BETA SCHEDULERS, CURRENTLY ONLY WARMUP_COSINE
    beta1_scheduler = optax.join_schedules(
            [optax.linear_schedule(1.0, 1.0-hyp_params['beta1'], n_steps // 4),
             optax.cosine_decay_schedule(1.0, n_steps // 2, alpha=1.0-hyp_params['beta1'])],
            [n_steps // 4]
        )
    
    pbar = tqdm.tqdm(range(hyp_params['n_epochs']))
    # losses = []

    for epoch in pbar:
        # Extract epoch-dependent info
        epoch_key = jax.random.PRNGKey(epoch)
        epoch_idxs = jax.random.permutation(epoch_key, n)
        batch_count = n // hyp_params['batch_size']

        for i in range(batch_count):
            # Extract batch-dependent info
            batch_key = jax.random.PRNGKey(i)
            batch_idxs = epoch_idxs[i * hyp_params['batch_size'] : (i + 1) * hyp_params['batch_size']]
            X_batch, y_batch = X[batch_idxs], y[batch_idxs]

            beta1 = 1 - beta1_scheduler(i)
            binary_loss_fn = lambda params, batch_key, input: binary_loss(
                batch_key, params, split_idx, input, apply_fn_enc, apply_fn_dec,
                pred_fn, beta1, hyp_params['beta2']
            )

            keys = jax.random.split(batch_key, len(X_batch))

            # Core JAX training step
            batch_loss_fn = lambda params: tree_map(
                lambda x: jax.numpy.mean(x),
                jax.vmap(binary_loss_fn, (None, 0, 0))(params, keys, X_batch)
            )
            grad_fn = jax.value_and_grad(batch_loss_fn)
            loss_value, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

        tqdm.tqdm.set_description(pbar, f"Epoch {epoch} average loss {loss_value:.4f}")
    
    def _step(carry: Any, x: jax.numpy.ndarray) -> Tuple[Any, Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]:
        """
        Helper function for scanning over the dataset to collect latent statistics.

        Args:
            carry (Any): Carry-over state (not used here).
            x (jax.numpy.ndarray): Input data sample.

        Returns:
            Tuple containing updated carry and a tuple of (mu, sigmasq).
        """
        mu, sigmasq = apply_fn_enc(state.params[:split_idx], x)
        return (mu, sigmasq), (mu, sigmasq)

    carry_init = apply_fn_enc(state.params[:split_idx], X[0])
    _, (mus, sigmasqs) = jax.lax.scan(_step, carry_init, X)
    mu_mean, mu_std = jax.numpy.mean(mus, axis=0), jax.numpy.std(mus, axis=0)
    sigmasq_mean, sigmasq_std = jax.numpy.mean(sigmasqs, axis=0), \
        jax.numpy.std(sigmasqs, axis=0)
        
    params_enc, params_dec = state.params[:split_idx], state.params[split_idx:]

    result = {
        'mu_mean': mu_mean,
        'mu_std': mu_std,
        'sigmasq_mean': sigmasq_mean,
        'sigmasq_std': sigmasq_std,
        'params_enc': params_enc,
        'params_dec': params_dec,
    }

    if ckpt_dir:
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        save_args = orbax_utils.save_args_from_target(result)
        ckptr.save(ckpt_dir, result, force=True, save_args=save_args)

    return result

