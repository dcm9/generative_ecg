from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag
Normal = tfd.Normal


def compute_electrode_electric_potential(s, p, r, kappa=0.2):
    """Compute the electric potential due to a single dipole.
    
    Args:
        s: 3-d location of the dipole
        p: 3-d moment of the dipole
        r: 3-d location of the electrode
        kappa: electrical conductivity of the torso
        
    Returns:
        e_p: electric potential at the electrode
    """
    d = r - s # 3-d displacement vector
    e_p = 1/(4*jnp.pi*kappa) * (d @ p)/jnp.linalg.norm(d)**3

    return e_p


def ndipole_compute_electrode_electric_potential(s, p, r, kappa=0.2):
    """Compute the electric potential due to n dipoles.
    
    Args:
        s (n, 3): locations of the n dipoles
        p (n, 3): moments of the n dipole
        r: 3-d location of the electrode
        kappa: electrical conductivity of the torso
        
    Returns:
        e_p: electric potential at the electrode
    """
    eps = jax.vmap(
        compute_electrode_electric_potential, (0, 0, None, None)
    )(s, p, r, kappa)
    e_p = jnp.sum(eps, axis=0)

    return e_p


def compute_lead_observations(eps):
    """Compute the 12d lead observations given 
    electric potentials for the 9 electrodes.

    Args:
        eps: 9-d electric potential values.
    
    Returns:
        leads: 12-d lead observations.
    """
    Oe = jnp.array([
        [1., -1., 0.],
        [0., -1., 1.],
        [-1., 0., 1.]
    ])
    Oa = jnp.array([
        [-0.5, 1., -0.5],
        [1., -0.5, -0.5],
        [-0.5, -0.5, 1.]
    ])
    Ov = -1/3 * jnp.ones((6, 3))
    Oh = jnp.vstack((jnp.zeros((6, 6)), jnp.eye(6)))
    Omat = jnp.hstack((jnp.vstack((Oe, Oa, Ov)), Oh))
    leads = Omat @ eps

    return leads


def predict_lead_obs(params):
    """Predict the 12d lead observations given
    the parameters of the model.

    Args:
        params: parameters of the n-dipole model.

    Returns:
        leads_pred: predicted 12d lead observations.
    """
    r, s, p = params["r"], params["s"], params["p"]
    eps = jax.vmap(
        jax.vmap(ndipole_compute_electrode_electric_potential, (0, 0, None)), 
        (None, None, 0)
    )(s, p, r)
    leads_pred = compute_lead_observations(eps)
    
    return leads_pred


# Log joint probability --------------------------------------------------------

def compute_log_likelihood(params, obs, obs_mask, obs_scale=0.1):
    """Compute the log likelihood of the model.
    
    Args:
        params: parameters of the model.
        obs: observed lead observations.
        obs_mask: mask for the observed lead observations.
        obs_scale: scale of the observation noise.
        
    Returns:
        log_likelihood: log likelihood of the model.
    """
    r, s, p = params["r"], params["s"], params["p"]
    eps = jax.vmap(
        jax.vmap(compute_electrode_electric_potential, (0, 0, None)), (None, None, 0)
    )(s, p, r)
    leads_pred = compute_lead_observations(eps)
    ll_fn = lambda pred, obs: Normal(loc=pred, scale=obs_scale).log_prob(obs)
    log_likelihood = jax.vmap(ll_fn)(leads_pred, obs)
    log_likelihood = jnp.sum(log_likelihood * obs_mask)
    
    return log_likelihood


def compute_log_joint_prob(params, obs, obs_mask, 
                           r_prior, s_prior, p_prior, obs_scale=0.01):
    """Compute the log joint probability of the model.

    Args:
        params: parameters of the model.
        obs: observed lead observations.
        obs_mask: mask for the observed lead observations.
        r_prior: prior distribution for the location of the dipole.
        s_prior: prior distribution for the moment of the dipole.
        p_prior: prior distribution for the location of the electrode.
        obs_scale: scale of the observation noise.

    Returns:
        log_joint: log joint probability of the model.
    """
    r, s, p = params["r"], params["s"], params["p"]
    log_likelihood = compute_log_likelihood(params, obs, obs_mask, obs_scale)
    # Assume Gaussian prior
    log_prior_r = r_prior.log_prob(r).sum()
    log_prior_s = s_prior.log_prob(s).sum()
    log_prior_p = p_prior.log_prob(p).sum()
    log_prior = log_prior_r + log_prior_s + log_prior_p
    log_joint = log_likelihood + log_prior
    
    return log_joint


@jax.jit
def update_step_logjoint(state, obs, obs_mask,
                         r_prior, s_prior, p_prior,  obs_scale=2e-2):
    log_joint, grads = jax.value_and_grad(compute_log_joint_prob)(
        state.params, obs, obs_mask, obs_scale, r_prior, s_prior, p_prior
    )
    state = state.apply_gradients(grads=grads)
    
    return state, log_joint


def train_log_joint(params, obs, r_prior, s_prior, p_prior,
                    obs_scale=1e-2, lr_peak=1e-1, lr_end=1e-7, n_epochs=1_000):
    """Train the model using the log joint probability as the loss function.

    Args:
        params: parameters of the model.
        obs: observed lead observations.
        r_prior: prior distribution for the location of the dipole.
        s_prior: prior distribution for the moment of the dipole.
        p_prior: prior distribution for the location of the electrode.
        obs_scale: scale of the observation noise.
        lr_peak: peak learning rate.
        lr_end: end learning rate.
        n_epochs: number of epochs to train.

    Returns:
        state: trained state of the model.
    """
    mask = jnp.where(jnp.isnan(obs), 0., 1.)
    obs = jnp.where(jnp.isnan(obs), 0., obs)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_peak,
        warmup_steps=n_epochs//10,
        decay_steps=n_epochs,
        end_value=lr_end
    )
    optimizer = optax.adam(lr_schedule)
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=optimizer
    )
    prange = tqdm(range(n_epochs),
                    desc=f"Epoch {0: >6} | Loss: {0:>10.9f}")
    
    for epoch in prange:    
        state, loss = update_step_logjoint(
            state, obs, mask, r_prior, s_prior, p_prior, obs_scale
        )
        prange.set_description(f"Epoch {epoch+1: >6} | Loss: {loss:>10.9f}")
            
    return state


# RMSE -------------------------------------------------------------------------

def rmse_loss(params, obs, obs_mask):
    leads_pred = predict_lead_obs(params)
    loss = (leads_pred - obs)**2
    loss = loss * obs_mask
    loss = jnp.sqrt(loss.mean())
    
    return loss

@jax.jit
def update_step_rmse(state, obs, obs_mask):
    loss, grads = jax.value_and_grad(rmse_loss)(state.params, obs, obs_mask)
    state = state.apply_gradients(grads=grads)
    
    return state, loss

def train_rmse(params, obs, lr_peak=1e-1, lr_end=1e-7, n_epochs=1_000):
    """Train the model using the RMSE as the loss function.

    Args:
        params: parameters of the model.
        obs: observed lead observations.
        lr_peak: peak learning rate.
        lr_end: end learning rate.
        n_epochs: number of epochs to train.

    Returns:
        state: trained state of the model.
    """
    mask = jnp.where(jnp.isnan(obs), 0., 1.)
    obs = jnp.where(jnp.isnan(obs), 0., obs)
    
    # Train state
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_peak,
        warmup_steps=n_epochs//10,
        decay_steps=n_epochs,
        end_value=lr_end
    )
    optimizer = optax.adam(lr_schedule)
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=optimizer
    )
    
    prange = tqdm(range(n_epochs), 
                  desc=f"Epoch {0: >6} | Loss: {0.:>10.9f}")
    for epoch in prange:
        state, loss = update_step_rmse(state, obs, mask)
        prange.set_description(f"Epoch {epoch+1: >6} | Loss: {loss:>10.9f}")
    
    return state