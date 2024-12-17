import jax.numpy
import jax.random
from ..models.loss_utils import rmse, fd

def evaluate_generated_ecgs(true, gen, sample=10):
    mean_rmses = []
    mean_fds = []
    key = jax.random.key(0)
    
    for ecg in gen:
        rmses = []
        fds = []
        samples = jax.random.choice(key, true, shape=(sample,))

        for sample in samples:
            rmses.append(rmse(sample, ecg))
            fds.append(fd(sample, ecg))
        
    print(f"Mean of Frechet Distance of {len(gen)} synthetic ECGS and {sample}-sample true ECGS: {jax.numpy.mean(mean_fds)}")
    print(f"Mean of RMSE of {len(gen)} synthetic ECGS and {sample}-sample true ECGS: {jax.numpy.mean(mean_rmses)}")

    return jax.numpy.mean(mean_fds), jax.numpy.mean(mean_fds)
    
