import jax
import jax.numpy
import optax

from .math_utils import gaussian_kl, gaussian_logpdf, gaussian_sample

def binary_ce_loss(params, apply_fn, X_batch, y_batch):
    y_preds = jax.vmap(apply_fn, (None, 0))(
        params, X_batch
    ).ravel()
    y_preds_labels = jax.nn.sigmoid(y_preds) > 0.5
    accuracy = jax.numpy.mean(y_preds_labels == y_batch)
    loss = optax.sigmoid_binary_cross_entropy(y_preds, y_batch).mean()
    
    return loss, accuracy

def rmse_loss(params, apply_fn, X_batch, y_batch):
    y_preds = jax.vmap(apply_fn, (None, 0))(
        params, X_batch
    ).ravel()
    loss = jax.numpy.sqrt(jax.numpy.mean((y_preds - y_batch)**2))
    
    return loss, loss

def dr_reg(input, x_pred, pred_fn):
    x_disc = pred_fn(input)
    x_pred_disc = pred_fn(x_pred)
    dr_reg_val = (x_disc - x_pred_disc) ** 2

    return dr_reg_val

def losses(key, params, split_idx, input, encoder_apply, decoder_apply):
    """Monte Carlo estimate of the negative evidence lower bound."""
    enc_params, dec_params = params[:split_idx], params[split_idx:]
    mu, sigmasq = encoder_apply(enc_params, input)
    z_pred = gaussian_sample(key, mu, sigmasq)
    x_pred = decoder_apply(dec_params, z_pred).reshape(input.shape)
    loss_rec = -gaussian_logpdf(x_pred, input)
    loss_kl = gaussian_kl(mu, sigmasq)

    return loss_rec, loss_kl, x_pred

def binary_loss(key, params, split_idx, input, encoder_apply, decoder_apply,
                pred_fn, beta1, beta2):
    """Binary cross-entropy loss."""
    loss_rec, loss_kl, x_pred = losses(
        key, params, split_idx, input, encoder_apply, decoder_apply
    )
    dr_reg_val = dr_reg(input, x_pred, pred_fn)
    result = loss_rec + beta1 * loss_kl + beta2 * dr_reg_val

    return result, (loss_rec, loss_kl, beta2 * dr_reg_val)
