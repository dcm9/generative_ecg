from functools import partial

# import cvxpy as cp
from flax.training import train_state
import jax
import jax.numpy as jnp
from jaxopt import ProjectedGradient
import optax
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag
Normal = tfd.Normal


OE = jnp.array([
    [1., -1., 0.],
    [0., -1., 1.],
    [-1., 0., 1.]
])
OA = jnp.array([
    [-0.5, 1., -0.5],
    [1., -0.5, -0.5],
    [-0.5, -0.5, 1.]
])
OV = -1/3 * jnp.ones((6, 3))
OH = jnp.vstack((jnp.zeros((6, 6)), jnp.eye(6)))
OMAT = jnp.hstack((jnp.vstack((OE, OA, OV)), OH))

R_PRIOR = jnp.array([
    [0.125, 0., 0.125],
    [-0.125, 0., 0.125],
    [0.125, 0., -0.125],
    [0.125*jnp.cos(100/180*jnp.pi), -0.125/2.75*jnp.sin(100/180*jnp.pi), 0.],
    [0.125*jnp.cos(80/180*jnp.pi), -0.125/2.75*jnp.sin(80/180*jnp.pi), 0.],
    [0.125*jnp.cos(60/180*jnp.pi), -0.125/2.75*jnp.sin(60/180*jnp.pi), 0.],
    [0.125*jnp.cos(40/180*jnp.pi), -0.125/2.75*jnp.sin(40/180*jnp.pi), 0.],
    [0.125*jnp.cos(20/180*jnp.pi), -0.125/2.75*jnp.sin(20/180*jnp.pi), 0.],
    [0.125*jnp.cos(0/180*jnp.pi), -0.125/2.75*jnp.sin(0/180*jnp.pi), 0.],
])


def compute_electrode_electric_potential(s, p, r, kappa=0.2):
    """Compute the electric potential due to a single dipole.
    
    Args:
        s: 3-d location of the dipole
        p: 3-d moment of the dipole
        r: 3-d location of the electrode
        kappa: electrical conductivity of the torso
        
    Returns:
        e_p: electric potential at the electrode, in millivolts
    """
    d = r - s # 3-d displacement vector
    e_p = 1/(4*jnp.pi*kappa) * (d @ p)/jnp.linalg.norm(d)**3
    e_p *= 1e3 # Convert to millivolts

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
    leads = OMAT @ eps

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
        jax.vmap(
            compute_electrode_electric_potential, (0, 0, None)
        ), (None, None, 0)
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
        r_prior: prior distribution for the location of the electrode.
        s_prior: prior distribution for the location of the dipole.
        p_prior: prior distribution for the moment of the dipole.
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
        r_prior: prior distribution for the location of the electrode.
        s_prior: prior distribution for the location of the dipole.
        p_prior: prior distribution for the moment of the dipole.
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

def rmse_loss(params, obs, obs_mask, s_smooth=0.0, p_smooth=0.0):
    if 's' not in params:
        params = {
            's': jnp.zeros_like(params['p']),
            **params
        }
    if 'r' not in params:
        params = {
            'r': R_PRIOR,
            **params
        }
    leads_pred = predict_lead_obs(params)
    rmse = (leads_pred - obs)**2
    rmse = rmse * obs_mask
    rmse = jnp.sqrt(jnp.nanmean(rmse))
    s_sm_reg = jnp.nanmean(
        jnp.linalg.norm(jnp.diff(params["s"], axis=0), axis=2)
    )
    p_sm_reg = jnp.nanmean(
        jnp.linalg.norm(jnp.diff(params["p"], axis=0), axis=2)
    )
    s_sm_reg = 0
    p_sm_reg = 0
    loss = rmse + s_smooth * s_sm_reg + p_smooth * p_sm_reg

    return loss


def _impose_hard_constraint(param, tol, constrain_dipoles_to_cuboid=False):
    limits = jnp.array([
        [-0.0425, 0.0425],
        [-0.03, 0.03],
        [-0.06, 0.06]
    ])
    
    def _project_step(carry, x_t, tol, cuboid_constraint=False):
        lb = jnp.array([limits[:, 0]] * carry.shape[0])
        ub = jnp.array([limits[:, 1]] * carry.shape[0])
        lower_bound = jnp.where(
            cuboid_constraint, 
            jnp.max(jnp.array([carry - tol, lb]), axis=0),
            carry - tol
        )
        upper_bound = jnp.where(
            cuboid_constraint, 
            jnp.min(jnp.array([carry + tol, ub]), axis=0),
            carry + tol
        )
        projected = jnp.clip(x_t, lower_bound, upper_bound)
        
        return projected, projected

    def _project(param, tol, cuboid_constraint=False):
        # The first timestep remains the same
        initial = param[0, :, :]
        if cuboid_constraint:
            initial = jnp.clip(initial, limits[:, 0], limits[:, 1])
        
        # Project all subsequent timesteps based on the previous timestep
        _, projected = jax.lax.scan(
            partial(_project_step, tol=tol, cuboid_constraint=cuboid_constraint),
            initial, 
            param[1:, :, :]
        )
        
        # Concatenate the initial timestep with the projected timesteps
        return jnp.concatenate([initial[None, :, :], projected], axis=0)
    
    def _objective_fun(curr_param):
        abs_diff =  jnp.mean(jnp.abs(curr_param - param))
        
        return abs_diff
    
    if tol < 0.0:
        tol = 1e6
        
    pg = ProjectedGradient(
        fun=_objective_fun, 
        projection=lambda x, *_: _project(x, tol, constrain_dipoles_to_cuboid)
    )
    optimized_param = pg.run(param).params
    
    return optimized_param


# def _impose_hard_constraint(param, tol, constrain_dipoles_to_cuboid=False):
#     n_steps, n_dipoles, n_dim = param.shape
#     param_reshaped = param.reshape(param.shape[0], -1)
#     param_var = cp.Variable(param_reshaped.shape)
    
#     # mse = cp.sum_squares(param_reshaped - param_var) / \
#     #     np.prod(param_reshaped.shape)
#     abs_diff = cp.norm(param_reshaped - param_var, 1)
#     objective = cp.Minimize(abs_diff)
    
#     constraints = []
#     for t in range(n_steps): # Time steps
#         for d in range(n_dipoles): # Dipole components
#             start_idx = d * n_dim
#             end_idx = (d+1) * n_dim
#             # if constrain_dipoles_to_cuboid:
#             #     x_min, x_max = -0.0425, 0.0425
#             #     y_min, y_max = -0.03, 0.03
#             #     z_min, z_max = -0.06, 0.06

#             #     constraints.append(param_var[t, start_idx] >= x_min)
#             #     constraints.append(param_var[t, start_idx] <= x_max)
#             #     constraints.append(param_var[t, start_idx+1] >= y_min)
#             #     constraints.append(param_var[t, start_idx+1] <= y_max)
#             #     constraints.append(param_var[t, start_idx+2] >= z_min)
#             #     constraints.append(param_var[t, start_idx+2] <= z_max)
#             if t == 0:
#                 continue
#             if tol > 0.0:
#                 # # Norm constraint - slower!
#                 # constraints.append(
#                 #     cp.norm(param_var[t, start_idx:end_idx] - 
#                 #             param_var[t-1, start_idx:end_idx], 2) <= tol
#                 # )
#                 # Elementwise linear bounds - faster!
#                 for i in range(3):
#                     constraints.append(
#                         cp.abs(param_var[t, start_idx+i] - 
#                                param_var[t-1, start_idx+i])
#                         <= tol
#                     )
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver = cp.SCS)
#     optimized_param = param_var.value.reshape(param.shape)
    
#     if constrain_dipoles_to_cuboid:
#         limits = jnp.array([
#             [-0.0425, 0.0425],
#             [-0.03, 0.03],
#             [-0.06, 0.06]
#         ])
#         for i in range(3):
#             optimized_param[:, :, i] = jnp.clip(
#                 optimized_param[:, :, i], limits[i, 0], limits[i, 1]
#             )
    
#     return optimized_param
    
    
@partial(jax.jit, static_argnums=(5,6,7,8,9))
def update_step_rmse(state, obs, obs_mask, s_smooth=0.0, p_smooth=0.0, 
                     fix_electrodes=False, fix_dipoles=False,
                     constrain_dipoles_to_cuboid=False,
                     s_hard=-1.0, p_hard=-1.0):
    loss, grads = jax.value_and_grad(rmse_loss)(state.params, obs, obs_mask,
                                                s_smooth, p_smooth)
    if fix_electrodes:
        grads["r"] = jnp.zeros_like(grads["r"])
    if fix_dipoles:
        grads["s"] = jnp.zeros_like(grads["s"])
    def replace_nans_with_zeros(x):
        return jnp.where(jnp.isnan(x), 0.0, x)
    grads_no_nans = jax.tree_map(replace_nans_with_zeros, grads)
    state = state.apply_gradients(grads=grads_no_nans) 
    if s_hard > 0.0 or constrain_dipoles_to_cuboid:
        state.params["s"] = _impose_hard_constraint(
            state.params["s"], s_hard, constrain_dipoles_to_cuboid
        )
    if p_hard > 0.0:
        state.params["p"] = _impose_hard_constraint(state.params["p"], p_hard)
   
    return state, loss
    

def train_proj_grad_rmse(params, obs, s_smooth=0.0, p_smooth=0.0,
                         lr_peak=1e-1, lr_end=1e-7, n_epochs=1_000, 
                         fix_electrodes=False, fix_dipoles=False,
                         constrain_dipoles_to_cuboid=False,
                         s_hard=-1.0, p_hard=-1.0):
    mask = jnp.where(jnp.isnan(obs), 0., 1.)
    obs = jnp.where(jnp.isnan(obs), 0., obs)
    
    limits = jnp.array([
        [-0.0425, 0.0425],
        [-0.03, 0.03],
        [-0.06, 0.06]
    ])
    
    def _objective_fun(curr_param):
        loss = rmse_loss(curr_param, obs, mask, s_smooth, p_smooth)
        
        return loss
    
    def _project_step(carry, x_t, tol):
        projected = jnp.clip(x_t, carry-tol, carry+tol)
        
        return projected, projected

    def _project(param, tol, cuboid_constraint=False):
        # The first timestep remains the same
        s_param = param["s"]
        if cuboid_constraint:
            s_param = jnp.clip(s_param, limits[:, 0], limits[:, 1])
        initial = s_param[0, :, :]
        # Project all subsequent timesteps based on the previous timestep
        _, projected = jax.lax.scan(
            partial(
                _project_step, 
                tol=tol,
            ),
            initial, 
            s_param[1:, :, :],
        )
        # Concatenate the initial timestep with the projected timesteps
        s_param = jnp.concatenate([initial[None, :, :], projected], axis=0)
        param.update({"s": s_param})
        
        return param
    
    if fix_electrodes:
        params.pop("r")
    if fix_dipoles:
        params.pop("s")
        project = lambda x, *_: x
    else:
        project = _project
    
    if s_hard < 0.0:
        s_hard = 1e9
        
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_end,
        peak_value=lr_peak,
        warmup_steps=n_epochs//10,
        decay_steps=n_epochs,
        end_value=lr_end
    )
    optimizer = optax.adam(lr_schedule)
    pg = ProjectedGradient(
        fun=_objective_fun, 
        projection=lambda x, *_: project(x, s_hard, constrain_dipoles_to_cuboid),
        maxiter=n_epochs,
        stepsize=lr_schedule,
    )
    params = pg.run(params).params
    if fix_electrodes:
        params["r"] = R_PRIOR
    if fix_dipoles:
        params["s"] = jnp.zeros_like(params["p"])
    # print(f"loss: {rmse_loss(params, obs, mask, s_smooth, p_smooth)}")
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=optimizer
    )
    
    return state


def train_rmse(params, obs, s_smooth=0.0, p_smooth=0.0,
               lr_peak=1e-1, lr_end=1e-7, n_epochs=1_000,
               fix_electrodes=False, fix_dipoles=False,
               constrain_dipoles_to_cuboid=False,
               s_hard=-1.0, p_hard=-1.0):
    """Train the model using the RMSE as the loss function.

    Args:
        params: parameters of the model.
        obs: observed lead observations.
        s_smooth: smoothness regularization for the dipole locations.
        p_smooth: smoothness regularization for the dipole moments.
        lr_peak: peak learning rate.
        lr_end: end learning rate.
        n_epochs: number of epochs to train.
        fix_electrodes: whether to fix the electrodes or not.
        fix_dipoles: whether to fix the dipole locations or not.
        constrain_dipoles_to_cuboid: whether to constrain the dipole
            locations to a cuboid or not.
        s_hard: hard constraint on the dipole locations.
        p_hard: hard constraint on the dipole moments.

    Returns:
        state: trained state of the model.
    """
    mask = jnp.where(jnp.isnan(obs), 0., 1.)
    obs = jnp.where(jnp.isnan(obs), 0., obs)
    
    if fix_electrodes:
        params["r"] = R_PRIOR
    if fix_dipoles:
        params["s"] = jnp.zeros_like(params["s"])
    
    # Train state
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_end,
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
        state, loss = update_step_rmse(state, obs, mask, s_smooth, p_smooth,
                                       fix_electrodes, fix_dipoles,
                                       constrain_dipoles_to_cuboid,
                                       s_hard, p_hard)
        prange.set_description(f"Epoch {epoch+1: >6} | Loss: {loss:>10.9f}")
    
    return state