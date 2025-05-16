import jax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
import jax.numpy
import jax.random
import tensorflow_probability.substrates.jax 


tfd = tensorflow_probability.substrates.jax.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag
Normal = tfd.Normal


OE = jax.numpy.array([
    [1., -1., 0.],
    [0., -1., 1.],
    [-1., 0., 1.]
])
OA = jax.numpy.array([
    [-0.5, 1., -0.5],
    [1., -0.5, -0.5],
    [-0.5, -0.5, 1.]
])
OV = -1/3 * jax.numpy.ones((6, 3))
OH = jax.numpy.vstack((jax.numpy.zeros((6, 6)), jax.numpy.eye(6)))
OMAT = jax.numpy.hstack((jax.numpy.vstack((OE, OA, OV)), OH))
    
def gaussian_kl(mu: jax.numpy.ndarray, sigmasq: jax.numpy.ndarray):
    """KL divergence from a diagonal Gaussian to the standard Gaussian.
    Args:
        mu (jax.numpy.array): mean of the Gaussian.
        sigmasq (jax.numpy.array): variance of the Gaussian.
    Returns:
        jax.numpy.array: KL divergence.
    """
    return -0.5 * jax.numpy.sum(1. + jax.numpy.log(sigmasq) - mu**2. - sigmasq)

def gaussian_sample(key: jax.random.PRNGKey, mu: jax.numpy.ndarray, sigmasq:jax.numpy.ndarray):
    """Sample a diagonal Gaussian.
    Args:
        key (jax.random.PRNGKey): random key.
        mu (jax.numpy.array): mean of the Gaussian.
        sigmasq (jax.numpy.array): variance of the Gaussian.
    Returns:
        jax.numpy.array: sample from the Gaussian.
    """
    return mu + jax.numpy.sqrt(sigmasq) * jax.random.normal(key, mu.shape)

def gaussian_logpdf(x_pred: jax.numpy.ndarray, x: jax.numpy.ndarray):
    """Gaussian log pdf of data x given x_pred.
    Args:
        x_pred (jax.numpy.array): predicted data.
        x (jax.numpy.array): data.
    Returns:
        jax.numpy.array: log pdf of x given x_pred.
    """
    return -0.5 * jax.numpy.sum((x - x_pred)**2., axis=-1)

def compute_linproj_residual(x: jax.numpy.ndarray) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """Compute the residual of the linear projection.

    Args:
        x (jax.numpy.array): 12d lead observation.

    Returns:
        res (jax.numpy.array): residual of the linear projection.
    """    
    sol, res, *_ = jax.numpy.linalg.lstsq(OMAT, x, rcond=None)
    
    return sol, res