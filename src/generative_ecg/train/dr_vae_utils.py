from functools import partial
from typing import Sequence, Callable, Tuple, Any, Dict
from tqdm import tqdm

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy
import jax.random
from flax.training import train_state, orbax_utils
import orbax.checkpoint

from ..models.nn_models import Encoder, Decoder, CNNEncoder
from ..models.math_utils import OMAT

def create_vae_base(
    X: jax.numpy.ndarray,
    model_params: Dict[str, Any]
) -> Tuple[
    Callable[[Any, jax.numpy.ndarray], Any],
    Callable[[Any, jax.numpy.ndarray], Any],
    Any,
    Any
]:
    """
    Create VAE encoder and decoder functions and their initial parameters.

    Args:
        X (jax.numpy.ndarray): Example input data, used to determine input shape.
        model_params (dict): Dictionary of model hyperparameters, including:
            - 'hidden_width': Width of hidden layers.
            - 'hidden_depth': Number of hidden layers.
            - 'z_dim': Latent dimension.
            - 'encoder_type': 'mlp' or 'cnn'.
            - 'use_bias': Whether to use bias in decoder layers.

    Returns:
        Tuple containing:
            - apply_fn_enc: Function to apply the encoder.
            - apply_fn_dec: Function to apply the decoder.
            - params_enc: Flattened encoder parameters.
            - params_dec: Flattened decoder parameters.
    """
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

def load_vae_from_ckpt(
    X: jax.numpy.ndarray,
    model_params: Dict[str, Any],
    ckpt_dir: str
) -> Dict[str, Any]:
    """
    Load a VAE model and its parameters from a checkpoint directory.

    Args:
        X (jax.numpy.ndarray): Example input data, used to determine input shape.
        model_params (dict): Dictionary of model hyperparameters.
        ckpt_dir (str): Path to the checkpoint directory.

    Returns:
        dict: Dictionary containing the loaded model parameters and encoder/decoder apply functions.
    """
    fn_enc, fn_dec, _, _ = create_vae_base(X, model_params)

    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    result = ckptr.restore(ckpt_dir)

    result['apply_fn_enc'] = fn_enc
    result['apply_fn_dec'] = fn_dec

    return result