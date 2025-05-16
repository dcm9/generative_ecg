from typing import Callable, Sequence
from flax.training import train_state
import jax.random
import optax

from ..models.nn_models import ECGConv

def create_cnn_train_state(X: jax.numpy.ndarray, key: jax.random.PRNGKey) -> train_state.TrainState:
    """
    Create a train state for the CNN model.
    Args:
        X (jax.numpy.ndarray): Input data for the model.
        key (jax.random.PRNGKey): Random key for initialization.
    Returns:
        train_state.TrainState: Initialized train state.
    """
    key, subkey = jax.random.split(key)
    nn_model = ECGConv(output_dim=1)
    params = nn_model.init(key, X[0])

    # Create trainstate
    tx = optax.adam(1e-3)
    opt_state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=tx
    )
    
    return opt_state
