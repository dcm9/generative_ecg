from functools import partial
from typing import Sequence
from tqdm import tqdm

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy
import jax.random
from flax.training import train_state, orbax_utils
import orbax.checkpoint

from ..models.nn_models import Encoder, Decoder, CNNEncoder
from ..models.math_utils import OMAT

def create_vae_base(X, model_params):
    model_key = jax.random.PRNGKey(0)
    key_enc, key_dec = jax.random.split(model_key)
    _, *x_dim = X.shape
    x_dim = jax.numpy.array(x_dim)

    hidden_feats = [model_params['hidden_width']] * model_params['hidden_depth']
    encoder_feats = [*hidden_feats, model_params['z_dim']]
    decoder_feats = [*hidden_feats, int(jax.numpy.prod(x_dim))]

    if model_params['encoder_type'] == "mlp":
        encoder = Encoder(encoder_feats)
    elif model_params['encoder_type'] == "cnn":
        encoder = CNNEncoder(model_params['z_dim'])

    params_enc = encoder.init(key_enc, jax.numpy.ones(x_dim,))['params']
    params_enc, unflatten_fn_enc = ravel_pytree(params_enc)
    apply_fn_enc = lambda params, x: encoder.apply(
        {'params': unflatten_fn_enc(params)}, x
    )

    decoder = Decoder(decoder_feats, use_bias=model_params['use_bias'])
    params_dec = decoder.init(key_dec, jax.numpy.ones(model_params['z_dim'],))['params']
    params_dec, unflatten_fn_dec = ravel_pytree(params_dec)

    apply_fn_dec = lambda params, x: decoder.apply(
        {'params': unflatten_fn_dec(params)}, x
    )

    return apply_fn_enc, apply_fn_dec, params_enc, params_dec

def load_vae_from_ckpt(X, model_params, ckpt_dir):
    """Load VAE model from checkpoint."""
    # Load model
    fn_enc, fn_dec, _, _ = create_vae_base(X, model_params)

    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    result = ckptr.restore(ckpt_dir)

    result['apply_fn_enc'] = fn_enc
    result['apply_fn_dec'] = fn_dec

    # mu, sigmasq = fn_enc(result['params_enc'], X)
    # z = mu + jax.numpy.sqrt(sigmasq) * jax.random.normal(jax.random.PRNGKey(0), mu.shape)
    # x_rec = fn_dec(result['params_dec'], z).reshape(X.shape)

    # if model_params['processed']:
    #     x_rec = OMAT @ x_rec
    #     X = OMAT @ X

    # jax.numpy.sqrt(jax.numpy.mean((X - x_rec)**2))

    return result