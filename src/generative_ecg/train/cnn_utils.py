from typing import Callable, Sequence
from flax.training import train_state
import jax.random as jr
import optax

from ..models.nn_models import CNN

def create_cnn_train_state(X, key=0):
    """Creates initial `TrainState`."""
    # Initialize NN model
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    nn_model = CNN(output_dim=1)
    params = nn_model.init(key, X[0])

    # Create trainstate
    tx = optax.adam(1e-3)
    opt_state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=tx
    )
    
    return opt_state
