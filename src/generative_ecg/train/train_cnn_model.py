from functools import partial
from pathlib import Path

from flax.training import train_state, orbax_utils
import jax
import optax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as orbax_ckpt
import tqdm


from Generative_ECG.src.generative_ecg.models.loss_utils import (
    rmse_loss,
    binary_ce_loss
)

from Generative_ECG.src.generative_ecg.models.nn_models import (
    CNN,

)

@partial(jax.jit, static_argnums=(1, 4))
def update_step(state, apply_fn, X_batch, y_batch, problem="classification"):
    loss_fn = binary_ce_loss if problem == "classification" else rmse_loss
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, apply_fn, X_batch, y_batch
    )
    state = state.apply_gradients(grads=grads)

    return state, loss, aux


def train_epoch(X, y, state, apply_fn, batch_size=128, key=0, 
                problem="classification"):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    n_train = len(X)
    idx = jr.permutation(key, n_train)
    X, y = X[idx], y[idx]
    n_batches = n_train // batch_size
    rem = n_train % batch_size
    if rem:
        n_batches += 1
    losses, auxes = [], []
    for batch in range(n_batches):
        lb, ub = batch*batch_size, (batch+1)*batch_size
        X_batch, y_batch = X[lb:ub], y[lb:ub]
        state, loss, aux = update_step(state, apply_fn, X_batch, y_batch,
                                  problem=problem)
        losses.append(loss)
        auxes.append(aux)
    loss = jnp.array(losses).mean()
    aux = jnp.array(auxes).mean()

    return state, loss, aux


def train_cnn(X, y, lr_init=1e-3, lr_peak=1e-2, lr_end=1e-3, n_epochs=10, 
          batch_size=64, key=0, problem="classification"):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    model = CNN(
        output_dim=1
    )
    apply_fn = lambda p, x: model.apply(
        p, x
    )
    params = model.init(
        key, X[0]
    )
    n_train = len(X)
    n_batches = n_train // batch_size
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_init,
        peak_value=lr_peak,
        warmup_steps=(n_batches*n_epochs) // 50,
        decay_steps=(n_batches*n_epochs) // 2,
        end_value=lr_end
    )
    optimizer = optax.adam(lr_schedule)
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=optimizer
    )
    if problem == "classification":
        prange = tqdm.tqdm(
            range(n_epochs), desc=f"Epoch {0:>6} | "
            f"Loss: {0.:>10.5f} | Accuracy: {0.:>10.2f}%"
        )
    else:
        prange = tqdm.tqdm(
            range(n_epochs), desc=f"Epoch {0:>6} | RMSE: {0.:>10.5f}"
        )
    for epoch in prange:
        state, loss, aux = train_epoch(
            X, y, state, apply_fn, batch_size, epoch, problem
        )
        if problem == "classification":
            prange.set_description(
                f"Epoch {epoch+1: >6} | Loss: {loss:>10.5f}" 
                f" | Accuracy: {aux*100:>10.2f}%"
            )
        else:    
            prange.set_description(
                f"Epoch {epoch+1: >6} | RMSE: {loss:>10.5f}"
            )
   
    return state


def train_discriminator(X_tr, y_tr, X_te, y_te, result_path, seed:int=0, 
                         dataset:str="ptb-xl", beat_segment:bool=False, 
                         processed:bool=False, target:str="age", 
                         n_channels:int=12,n_epochs:int=100):
    if target in ("age", "range", "max", "mean"):
        problem = "regression"
        loss_fn = rmse_loss
    elif target in ("sex", "min-max-order"):
        problem = "classification"
        loss_fn = binary_ce_loss
    else:
        raise ValueError(f"Unknown target: {target}")
    
    # Shuffle
    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    idx_tr = jr.permutation(key, len(X_tr))
    X_tr, y_tr = X_tr[idx_tr], y_tr[idx_tr]
    idx_te = jr.permutation(subkey, len(X_te))
    X_te, y_te = X_te[idx_te], y_te[idx_te]
    
    # Train
    state = train_cnn(X_tr, y_tr, n_epochs=n_epochs, problem=problem)
    
    # Evaluate
    model = CNN(output_dim=1)
    loss_te, aux_te = loss_fn(state.params, model.apply, X_te, y_te)
    print(f"Test loss: {loss_te:.4f}")
    if problem == "classification":
        print(f"Test accuracy: {aux_te:.4f}")
    
    # Save checkpoint
    dataset_name = dataset
    if processed:
        dataset_name += "_processed"
    ckpt_dir = Path(result_path, dataset_name, f"{target}")
    if beat_segment:
        ckpt_dir = Path(ckpt_dir, f"cnn_bs_{n_channels}_ckpt")
    else:
        ckpt_dir = Path(ckpt_dir, f"cnn_{n_channels}_ckpt")
    ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
    save_args = orbax_utils.save_args_from_target(state)
    ckptr.save(ckpt_dir, state, force=True, save_args=save_args)

    return ckpt_dir
