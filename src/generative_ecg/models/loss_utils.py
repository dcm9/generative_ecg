import jax
import jax.numpy
import optax
from typing import Any, Callable, Tuple

from .math_utils import gaussian_kl, gaussian_logpdf, gaussian_sample

def binary_ce_loss(
    params: Any,
    apply_fn: Callable,
    X_batch: jax.numpy.ndarray,
    y_batch: jax.numpy.ndarray
) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """
    Compute the binary cross-entropy loss and accuracy for a batch.

    Args:
        params (Any): Model parameters.
        apply_fn (Callable): Model apply function.
        X_batch (jax.numpy.ndarray): Input batch of data.
        y_batch (jax.numpy.ndarray): Ground truth labels for the batch.

    Returns:
        Tuple[jax.numpy.ndarray, jax.numpy.ndarray]: Tuple containing the loss and accuracy.
    """
    y_preds = jax.vmap(apply_fn, (None, 0))(params, X_batch).ravel()
    y_preds_labels = jax.nn.sigmoid(y_preds) > 0.5
    accuracy = jax.numpy.mean(y_preds_labels == y_batch)
    loss = optax.sigmoid_binary_cross_entropy(y_preds, y_batch).mean()
    return loss, accuracy

def rmse_loss(
    params: Any,
    apply_fn: Callable,
    X_batch: jax.numpy.ndarray,
    y_batch: jax.numpy.ndarray
) -> jax.numpy.ndarray:
    """
    Compute the root mean squared error (RMSE) loss for a batch.

    Args:
        params (Any): Model parameters.
        apply_fn (Callable): Model apply function.
        X_batch (jax.numpy.ndarray): Input batch of data.
        y_batch (jax.numpy.ndarray): Ground truth labels for the batch.

    Returns:
        jax.numpy.ndarray: The RMSE loss value.
    """
    y_preds = jax.vmap(apply_fn, (None, 0))(params, X_batch).ravel()
    loss = jax.numpy.sqrt(jax.numpy.mean((y_preds - y_batch)**2))
    return loss

def dr_reg(
    input: jax.numpy.ndarray,
    x_pred: jax.numpy.ndarray,
    pred_fn: Callable
) -> jax.numpy.ndarray:
    """
    Compute the discriminator regularization value.

    Args:
        input (jax.numpy.ndarray): Original input data.
        x_pred (jax.numpy.ndarray): Predicted/reconstructed data.
        pred_fn (Callable): Discriminator or prediction function.

    Returns:
        jax.numpy.ndarray: The regularization value.
    """
    x_disc = pred_fn(input)
    x_pred_disc = pred_fn(x_pred)
    dr_reg_val = (x_disc - x_pred_disc) ** 2
    return dr_reg_val

def losses(
    key: jax.random.PRNGKey,
    params: Any,
    split_idx: int,
    input: jax.numpy.ndarray,
    encoder_apply: Callable,
    decoder_apply: Callable
) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]:
    """
    Monte Carlo estimate of the negative evidence lower bound (ELBO) for a VAE.

    Args:
        key (jax.random.PRNGKey): Random key for sampling.
        params (Any): Combined encoder and decoder parameters.
        split_idx (int): Index to split encoder and decoder parameters.
        input (jax.numpy.ndarray): Input data sample.
        encoder_apply (Callable): Encoder apply function.
        decoder_apply (Callable): Decoder apply function.

    Returns:
        Tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]:
            - Reconstruction loss
            - KL divergence loss
            - Reconstructed input
    """
    enc_params, dec_params = params[:split_idx], params[split_idx:]
    mu, sigmasq = encoder_apply(enc_params, input)
    z_pred = gaussian_sample(key, mu, sigmasq)
    x_pred = decoder_apply(dec_params, z_pred).reshape(input.shape)
    loss_rec = -gaussian_logpdf(x_pred, input)
    loss_kl = gaussian_kl(mu, sigmasq)
    return loss_rec, loss_kl, x_pred

def binary_loss(
    key: jax.random.PRNGKey,
    params: Any,
    split_idx: int,
    input: jax.numpy.ndarray,
    encoder_apply: Callable,
    decoder_apply: Callable,
    pred_fn: Callable,
    beta1: float,
    beta2: float
) -> jax.numpy.ndarray:
    """
    Compute the total binary loss for a VAE, including reconstruction, KL, and regularization terms.

    Args:
        key (jax.random.PRNGKey): Random key for sampling.
        params (Any): Combined encoder and decoder parameters.
        split_idx (int): Index to split encoder and decoder parameters.
        input (jax.numpy.ndarray): Input data sample.
        encoder_apply (Callable): Encoder apply function.
        decoder_apply (Callable): Decoder apply function.
        pred_fn (Callable): Discriminator or prediction function for regularization.
        beta1 (float): Weight for KL divergence term.
        beta2 (float): Weight for discriminator regularization term.

    Returns:
        jax.numpy.ndarray: The total loss value.
    """
    loss_rec, loss_kl, x_pred = losses(
        key, params, split_idx, input, encoder_apply, decoder_apply
    )
    dr_reg_val = dr_reg(input, x_pred, pred_fn)
    result = loss_rec + beta1 * loss_kl + beta2 * dr_reg_val
    return result
