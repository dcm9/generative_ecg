import argparse
from functools import partial
from pathlib import Path

import flax.linen as nn
from flax.training import train_state, orbax_utils
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as orbax_ckpt
import tqdm

from mlscience_ekgs.Code.experiments.s02_train_and_generate_ecgs import load_dataset
from mlscience_ekgs.settings import result_path


class CNN(nn.Module):
    output_dim: int
    activation: nn.Module = nn.relu
    
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (1, 0)) # to (batch_size, time, channel)
        x = nn.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Conv(features=12, kernel_size=(10,))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.ravel()
        x = nn.Dense(features=128)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.output_dim)(x).ravel()
        
        return x


def binary_ce_loss(params, apply_fn, X_batch, y_batch):
    y_preds = jax.vmap(apply_fn, (None, 0))(
        params, X_batch
    ).ravel()
    y_preds_labels = jax.nn.sigmoid(y_preds) > 0.5
    accuracy = jnp.mean(y_preds_labels == y_batch)
    loss = optax.sigmoid_binary_cross_entropy(y_preds, y_batch).mean()
    
    return loss, accuracy


def rmse_loss(params, apply_fn, X_batch, y_batch):
    y_preds = jax.vmap(apply_fn, (None, 0))(
        params, X_batch
    ).ravel()
    loss = jnp.sqrt(jnp.mean((y_preds - y_batch)**2))
    
    return loss, loss


@partial(jax.jit, static_argnums=(1, 4))
def update_step(state, apply_fn, X_batch, y_batch, problem="classification"):
    loss_fn = binary_ce_loss if problem == "classification" else rmse_loss
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, apply_fn, X_batch, y_batch
    )
    state = state.apply_gradients(grads=grads)

    return state, loss 


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
    losses = []
    for batch in range(n_batches):
        lb, ub = batch*batch_size, (batch+1)*batch_size
        X_batch, y_batch = X[lb:ub], y[lb:ub]
        state, loss = update_step(state, apply_fn, X_batch, y_batch,
                                  problem=problem)
        losses.append(loss)
    loss = jnp.array(losses).mean()

    return state, loss


def train(X, y, lr_init=1e-3, lr_peak=5e-2, lr_end=1e-3, n_epochs=10, 
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
   
    prange = tqdm.tqdm(range(n_epochs),
                       desc=f"Epoch {0: >6} | Loss: {0.:>10.9f}")
    for epoch in prange:
        state, loss = train_epoch(
            X, y, state, apply_fn, batch_size, epoch, problem
        )
        prange.set_description(
            f"Epoch {epoch+1: >6} | Loss: {loss:>10.9f}"
        )
   
    return state


def main(args):
    # Load dataset
    X, y = load_dataset("ptb-xl", args.beat_segment, None, args.n_channels,
                        target=args.target)
    if args.target == "age":
        problem = "regression"
        loss_fn = rmse_loss
    elif args.target == "sex":
        problem = "classification"
        loss_fn = binary_ce_loss
    else:
        raise ValueError(f"Unknown target: {args.target}")
    
    # Shuffle
    key = jr.PRNGKey(args.seed)
    idx = jr.permutation(key, len(X))
    X, y = X[idx], y[idx]
    
    n_train = int(0.8*len(X))
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    
    # Train
    state = train(X_tr, y_tr, n_epochs=args.n_epochs, problem=problem)
    
    # Evaluate
    model = CNN(output_dim=1)
    loss_te, aux_te = loss_fn(state.params, model.apply, X_te, y_te)
    print(f"Test loss: {loss_te:.4f}")
    if problem == "classification":
        print(f"Test accuracy: {aux_te:.4f}")
    
    # Save checkpoint
    ckpt_dir = Path(result_path, f"{args.target}")
    if args.beat_segment:
        ckpt_dir = Path(ckpt_dir, f"cnn_bs_{args.n_channels}_ckpt")
    else:
        ckpt_dir = Path(ckpt_dir, f"cnn_{args.n_channels}_ckpt")
    ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
    save_args = orbax_utils.save_args_from_target(state)
    ckptr.save(ckpt_dir, state, force=True, save_args=save_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=0) # random seed
    
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--target", type=str, default="age")
    parser.add_argument("--n_channels", type=int, default=3) # number of channels to use
    
    parser.add_argument("--n_epochs", type=int, default=100) # number of epochs to train
    
    args = parser.parse_args()
    
    main(args)