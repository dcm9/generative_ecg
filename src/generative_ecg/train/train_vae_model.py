from pathlib import Path

import jax.numpy
import jax.random
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from flax.training import train_state, orbax_utils
import orbax.checkpoint
import optax
import tqdm

from .cnn_utils import create_cnn_train_state
from .dr_vae_utils import train_dr_vae
from ..models.nn_models import Encoder, Decoder, CNNEncoder
from ..models.math_utils import OMAT
from ..models.loss_utils import binary_loss


def train_vae(X_tr, X_te, y_tr, y_te, pred_fn, params, lr_schedule, ckpt_dir:None):
    model_key = jax.random.PRNGKey(0)
    key_enc, key_dec = jax.random.split(model_key)
    _, *x_dim = X_tr.shape
    x_dim = jax.numpy.array(x_dim)
    n = len(X_tr)

    hidden_feats = [params['hidden_width']] * params['hidden_depth']
    encoder_feats = [*hidden_feats, params['z_dim']]
    decoder_feats = [*hidden_feats, jax.numpy.prod(params['z_dim'])]

    if params['encoder_type'] == "mlp":
        encoder = Encoder(encoder_feats)
    elif params['encoder_type'] == "cnn":
        encoder = CNNEncoder(params['z_dim'])

    params_enc = encoder.init(key_enc, jax.numpy.ones(x_dim,))['params']
    params_enc, unflatten_fn_enc = ravel_pytree(params_enc)
    print(f"Encoder params size: {params_enc.shape}")
    apply_fn_enc = lambda params, x: encoder.apply(
        {'params': unflatten_fn_enc(params)}, x
    )

    decoder = Decoder(decoder_feats, use_bias=params['use_bias'])
    params_dec = decoder.init(key_dec, jax.numpy.ones(params['z_dim'],))['params']
    params_dec, unflatten_fn_dec = ravel_pytree(params_dec)
    print(f"Decoder params size: {params_dec.shape}")
    apply_fn_dec = lambda params, x: decoder.apply(
        {'params': unflatten_fn_dec(params)}, x
    )
    params = jax.numpy.array([*params_enc, *params_dec])
    split_idx = len(params_enc)

    # Train state
    n_steps = params['n_epochs'] * (n // params['batch_size'])
    optimizer = optax.adam(lr_schedule)
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=optimizer
    )

    # TODO: ALLOW FOR DIFFERENT BETA SCHEDULERS, CURRENTLY ONLY WARMUP_COSINE
    beta1_scheduler = optax.join_schedules(
            [optax.linear_schedule(1.0, 1.0-params['beta1'], n_steps // 4),
             optax.cosine_decay_schedule(1.0, n_steps // 2, alpha=1.0-params['beta1'])],
            [n_steps // 4]
        )
    
    pbar = tqdm.tqdm(range(params['n_epochs']), desc=f"Epoch 0 average loss: 0.0")
    # losses = []

    for epoch in pbar:
        # Extract epoch-dependent info
        epoch_key = jax.random.PRNGKey(epoch)
        epoch_idxs = jax.random.permutation(epoch_key, n)
        batch_count = n // params['batch_size']

        for i in range(batch_count):
            # Extract batch-dependent info
            batch_key = jax.random.PRNGKey(i)
            batch_idxs = epoch_idxs[i * params['batch_size'] : (i + 1) * params['batch_size']]
            X_batch, y_batch = X_tr[batch_idxs], y_tr[batch_idxs]

            beta1 = 1 - beta1_scheduler(i)
            binary_loss_fn = lambda params, batch_key, input: binary_loss(
                batch_key, params, split_idx, input, apply_fn_enc, apply_fn_dec,
                pred_fn, beta1, params['beta2']
            )

            keys = jax.random.split(batch_key, len(batch_idxs))

            # Core JAX training step
            batch_loss_fn = lambda params: tree_map(
                lambda x: jax.numpy.mean(x),
                jax.vmap(binary_loss_fn, (None, 0, 0))(params, keys, X_batch)
            )
            grad_fn = jax.value_and_grad(batch_loss_fn)
            loss_value, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            tqdm.tqdm.set_description(pbar, f"Epoch {epoch} average loss: {loss_value:.4f}")
    
    def _step(carry, x):
        mu, sigmasq = apply_fn_enc(state.params[:split_idx], x)
        return (mu, sigmasq), (mu, sigmasq)

    carry_init = apply_fn_enc(state.params[:split_idx], X_tr[0])
    _, (mus, sigmasqs) = jax.lax.scan(_step, carry_init, X_tr)
    mu_mean, mu_std = jax.numpy.mean(mus, axis=0), jax.numpy.std(mus, axis=0)
    sigmasq_mean, sigmasq_std = jax.numpy.mean(sigmasqs, axis=0), \
        jax.numpy.std(sigmasqs, axis=0)
    
    # losses= jax.numpy.array(losses)
    
    params_enc, params_dec = state.params[:split_idx], state.params[split_idx:]

    result = {
        'apply_fn_enc': apply_fn_enc,
        'apply_fn_dec': apply_fn_dec,
        'mu_mean': mu_mean,
        'mu_std': mu_std,
        'sigmasq_mean': sigmasq_mean,
        'sigmasq_std': sigmasq_std,
        'params_enc': params_enc,
        'params_dec': params_dec,
    }

    return result

