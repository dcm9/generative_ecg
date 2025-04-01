from functools import partial
from pathlib import Path

from flax.training import train_state, orbax_utils
import jax
import optax
import jax.numpy
import jax.random
import orbax.checkpoint 
import tqdm

from ..models.loss_utils import (
    rmse_loss,
    binary_ce_loss
)

from ..models.nn_models import (
    CNN,

)

def train_cnn(X_beats, y_beats, model, loss_fn, lr_schedule, ckpt_dir:None,  
              batch_size=64):

    optimizer = optax.adam(lr_schedule)
    model_key = jax.random.PRNGKey(0)
    params = model.init(model_key, X_beats[0])
    state = flax.training.train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    # QUESTION: DO WE SPLIT TRAIN/TEST, OR DOES THE USER IF WE'RE CALCULATING TR/TE values
    batch_count = len(train_data[0]) // batch_size

    # Run training loops
    epoch_loss_hist = []
    for epoch_n in range(epoch_count):
        # Extract epoch-dependent info
        epoch_key = jax.random.PRNGKey(epoch_n)
        epoch_idxs = jax.random.permutation(epoch_key, len(X_beats[0]))
        loss_hist = []
        for batch_n in range(batch_count):
            # Extract batch-dependent info
            batch_idxs = epoch_idxs[i * batch_size : (i + 1) * batch_size]
            X_batch, y_batch = X_beats[batch_idxs], y_beats[batch_idxs]

            # Core JAX training step
            batch_loss_fn = lambda params: loss_fn(params, state.apply_fn, X_batch, y_batch)
            grad_fn = jax.value_and_grad(batch_loss_fn)
            loss_value, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            # Store metadata
            loss_hist.append(loss_value)
        
        epoch_loss_hist.append(loss_hist)

    # ANOTHER QUESTION: WHY DO WE NEED TO SAVE EPOCH LOSS HIST

    # Extract validation metrics
    loss, aux = loss_fn(state.params, model.apply, X_beats, y_beats)
    print(f"Loss: {loss:.4f}")

    # Checkpointing 
    if ckpt_dir:
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        save_args = orbax_utils.save_args_from_target(state)
        ckptr.save(ckpt_dir, state, force=True, save_args=save_args)

    return state