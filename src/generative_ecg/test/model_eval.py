import jax.numpy as jnp
import jax.random as jnr
from ..models.loss_utils import rmse, fd

def evaluate_generated_ecgs(true, gen, sample=10):
    mean_rmses = []
    mean_fds = []
    key = jnr.key(0)
    
    for ecg in gen:
        rmses = []
        fds = []
        samples = jnr.choice(key, true, shape=(sample,))

        for sample in samples:
            rmses.append(rmse(sample, ecg))
            fds.append(fd(sample, ecg))
        
    print(f"Mean of Frechet Distance of {len(gen)} synthetic ECGS and {sample}-sample true ECGS: {jnp.mean(mean_fds)}")
    print(f"Mean of RMSE of {len(gen)} synthetic ECGS and {sample}-sample true ECGS: {jnp.mean(mean_rmses)}")

    return jnp.mean(mean_fds), jnp.mean(mean_fds)
    
