from functools import partial
from pathlib import Path

import flax
from flax.training import train_state, orbax_utils
import jax
import optax
import jax.numpy
import jax.random
import orbax.checkpoint 
import tqdm

from typing import Callable, Optional, Any

from ..models.loss_utils import (
    rmse_loss,
    binary_ce_loss
)

def train_cnn(
    X_tr: jax.numpy.ndarray,
    X_te: jax.numpy.ndarray,
    y_tr: jax.numpy.ndarray,
    y_te: jax.numpy.ndarray,
    model: Any,
    loss_fn: Callable[[Any, Callable, jax.numpy.ndarray, jax.numpy.ndarray], Any],
    lr_schedule: Any,
    ckpt_dir: Optional[str] = None,
    batch_size: int = 64,
    n_epochs: int = 1000
) -> train_state.TrainState:
    """
    Train a neural network model (e.g., CNN) using JAX and Optax.

    Args:
        X_tr (jax.numpy.ndarray): Training input data.
        X_te (jax.numpy.ndarray): Validation input data.
        y_tr (jax.numpy.ndarray): Training labels/targets.
        y_te (jax.numpy.ndarray): Validation labels/targets.
        model (Any): Flax model to be trained.
        loss_fn (Callable): Loss function to use for training.
        lr_schedule (Any): Learning rate schedule or optimizer.
        ckpt_dir (Optional[str]): Directory to save checkpoints. If None, no checkpointing.
        batch_size (int, optional): Batch size for training. Default is 64.
        n_epochs (int, optional): Number of training epochs. Default is 1000.

    Returns:
        train_state.TrainState: The final training state containing model parameters and optimizer state.
    """
    optimizer = optax.adam(lr_schedule)
    model_key = jax.random.PRNGKey(0)
    params = model.init(model_key, X_tr[0])
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    batch_count = X_tr.shape[0] // batch_size

    for epoch_n in tqdm.tqdm(range(n_epochs), desc="Training", unit="epoch"):
        # Extract epoch-dependent info
        epoch_key = jax.random.PRNGKey(epoch_n)
        epoch_idxs = jax.random.permutation(epoch_key, X_tr.shape[0])
        loss_hist = []

        for i in range(batch_count):
            # Extract batch-dependent info
            batch_idxs = epoch_idxs[i * batch_size : (i + 1) * batch_size]
            X_batch, y_batch = X_tr[batch_idxs], y_tr[batch_idxs]

            # Core JAX training step
            batch_loss_fn = lambda params: loss_fn(params, state.apply_fn, X_batch, y_batch)
            grad_fn = jax.value_and_grad(batch_loss_fn)
            loss_value, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            # Store metadata
            loss_hist.append(loss_value)
    
    # Extract validation metrics
    val_loss = loss_fn(state.params, model.apply, X_te, y_te)
    print(f"Validation Loss: {val_loss:.4f}")

    # Checkpointing 
    if ckpt_dir:
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        save_args = orbax_utils.save_args_from_target(state)
        ckptr.save(ckpt_dir, state, force=True, save_args=save_args)

    return state