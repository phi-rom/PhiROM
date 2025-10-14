"""
Inference and evaluation functions for trained models.
"""

from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import PyTree

from ..modules.inference import find_latent_descent, find_latent_descent_XLB
from ..modules.models import NodeROM
from ..training.metrics import mean_squared_error, normalized_mse


def unroll_descent(
    model: NodeROM,
    model_state: PyTree,
    batch,
    t1,
    t0=0,
    subsample_rate=1.0,
    n_steps=500,
    loss="nmse",
    optimizer: optax.GradientTransformation = None,
    ode_solver=None,
    adaptive=None,
    return_latent=False,
    ode_steps=None,
    loss_inversion="mse",
):
    """
    Inference function for unrolling a batch of initial conditions by first performing inversion using a gradient descent method, then integrating the dynamics network, and finally reconstructing all the trajectory.

    Args:
        model (NodeROM): The trained NodeROM model.
        model_state (PyTree): The state of the model (e.g., for batch normalization).
        batch (dict): A batch of data containing 'data', 'coords', and optionally 'node_args'.
        t1 (int): The end time index for unrolling.
        t0 (int, optional): The start time index for unrolling. Defaults to 0.
        subsample_rate (float, optional): The rate at which to randomly subsample coordinates for inversion. Defaults to 1.0 (no subsampling).
        n_steps (int, optional): Number of gradient descent steps for inversion. Defaults to 500.
        loss (str, optional): The loss function to use for error calculation ('mse' or 'nmse'). Defaults to 'nmse'.
        optimizer (optax.GradientTransformation, optional): The optimizer to use for inversion. Defaults to None (uses Adam with lr=1e-1).
        ode_solver (callable, optional): The ODE solver to use for integration. Defaults to None.
        adaptive (bool, optional): Whether to use adaptive step sizing in the ODE solver. Defaults to None.
        return_latent (bool, optional): Whether to return the integrated latent states along with unrolls and errors. Defaults to False.
        ode_steps (int, optional): Number of steps for the ODE solver if not adaptive. Defaults to None.
        loss_inversion (str, optional): The loss function to use for inversion ('mse' or 'nmse'). Defaults to 'mse'.

    Returns:
        unrolls (np.ndarray): The reconstructed trajectories of shape (batch_size, t1-t0, *data_shape).
        errors (np.ndarray): The error for each trajectory in the batch.
        ls (np.ndarray, optional): The integrated latent states if return_latent is True.
    """

    model = eqx.nn.inference_mode(model, True)
    trajs = batch["data"]
    node_args = batch.get("node_args", None)
    coords = batch["coords"]
    ts = batch["t"][:, t0:t1]
    errors = []
    traj_ics = trajs[:, t0].reshape(
        trajs.shape[0], trajs.shape[2], np.prod(trajs.shape[3:])
    )
    find_latent = partial(find_latent_descent, loss=loss_inversion)
    if optimizer is None:
        optimizer = optax.adam(1e-1)
    if subsample_rate < 1.0:  # Subsampled inversion
        key = jr.PRNGKey(0)
        sub_coords, indices = eqx.filter_vmap(
            subsample_coords, in_axes=(0, None, None, None)
        )(coords, subsample_rate, True, key)
        print(indices.shape)
        traj_ics = traj_ics[:, :, indices[0]]
        print(sub_coords.shape)
        latents_ic = eqx.filter_vmap(
            find_latent, in_axes=(None, 0, 0, None, None, None, None)
        )(
            model.decoder,
            traj_ics,
            sub_coords,
            model.decoder.latent_dim,
            optimizer,
            None,
            n_steps,
        )[
            0
        ]
    else:  # Full inversion
        latents_ic = eqx.filter_vmap(
            find_latent, in_axes=(None, 0, 0, None, None, None, None)
        )(
            model.decoder,
            traj_ics,
            coords,
            model.decoder.latent_dim,
            optimizer,
            None,
            n_steps,
        )[
            0
        ]
    if model.node.param_size == 0:  # Integrate latent ODE (no PDE parameters)
        ls, _, _ = eqx.filter_vmap(
            model.node,
            in_axes=(0, 0, None, None, None, None, None, None),
            out_axes=(0, 0, None),
            axis_name="batch",
        )(latents_ic, ts, None, None, ode_solver, adaptive, ode_steps, model_state)
    else:  # Integrate latent ODE (parameterized PDE)
        ls, _, _ = eqx.filter_vmap(
            model.node,
            in_axes=(0, 0, None, 0, None, None, None, None),
            out_axes=(0, 0, None),
            axis_name="batch",
        )(latents_ic, ts, None, node_args, ode_solver, adaptive, ode_steps, model_state)
    unrolls = (
        eqx.filter_jit(
            eqx.filter_vmap(
                eqx.filter_vmap(
                    eqx.filter_vmap(
                        model.decoder.call_coords_latent, in_axes=(0, None)
                    ),
                    in_axes=(None, 0),
                ),
                in_axes=(0, 0),
            )
        )(coords, ls)
        .transpose(0, 1, 3, 2)
        .reshape(trajs.shape[0], t1 - t0, *trajs.shape[2:])
    )  # Reconstruct the trajectory
    if loss == "mse":
        errors = eqx.filter_jit(eqx.filter_vmap(mean_squared_error))(
            unrolls, trajs[:, t0:t1]
        )
    elif loss == "nmse":
        errors = eqx.filter_jit(eqx.filter_vmap(normalized_mse))(
            unrolls, trajs[:, t0:t1]
        )
    unrolls = np.array(unrolls)
    errors = np.array(errors)
    if return_latent:
        return unrolls, errors, ls
    return unrolls, errors


def unroll_descent_XLB(
    model: NodeROM,
    model_state: PyTree,
    macro,
    batch,
    t1,
    t0=0,
    subsample_rate=1.0,
    n_steps=500,
    loss="nmse",
    optimizer: optax.GradientTransformation = None,
    ode_solver=None,
    adaptive=None,
    return_latent=False,
    ode_steps=None,
):
    """
    Inference function for unrolling a batch of initial conditions by first performing inversion using a gradient descent method, then integrating the dynamics network, and finally reconstructing all the trajectory.
    Modified for LBM.
    """

    model = eqx.nn.inference_mode(model, True)
    trajs = batch["data"]
    node_args = batch.get("node_args", None)
    coords = batch["coords"]
    ts = batch["t"][:, t0:t1]
    errors = []
    _, trajs = eqx.filter_vmap(eqx.filter_vmap(macro))(trajs)
    traj_ics = trajs[:, t0].reshape(
        trajs.shape[0], trajs.shape[2], np.prod(trajs.shape[3:])
    )
    if optimizer is None:
        optimizer = optax.adam(1e-1)
    if subsample_rate < 1.0:
        key = jr.PRNGKey(0)
        sub_coords, indices = eqx.filter_vmap(
            subsample_coords, in_axes=(0, None, None, None)
        )(coords, subsample_rate, True, key)
        print(indices.shape)
        traj_ics = traj_ics[:, :, indices[0]]
        print(sub_coords.shape)
        latents_ic = eqx.filter_vmap(
            find_latent_descent_XLB, in_axes=(None, None, 0, 0, None, None, None, None)
        )(
            model.decoder,
            macro,
            traj_ics,
            sub_coords,
            model.decoder.latent_dim,
            optimizer,
            None,
            n_steps,
        )[
            0
        ]
    else:
        latents_ic = eqx.filter_vmap(
            find_latent_descent_XLB, in_axes=(None, None, 0, 0, None, None, None, None)
        )(
            model.decoder,
            macro,
            traj_ics,
            coords,
            model.decoder.latent_dim,
            optimizer,
            None,
            n_steps,
        )[
            0
        ]
    if model.node.param_size == 0:
        ls, _, _ = eqx.filter_vmap(
            model.node,
            in_axes=(0, 0, None, None, None, None, None, None),
            out_axes=(0, 0, None),
            axis_name="batch",
        )(latents_ic, ts, None, None, ode_solver, adaptive, ode_steps, model_state)
    else:
        ls, _, _ = eqx.filter_vmap(
            model.node,
            in_axes=(0, 0, None, 0, None, None, None, None),
            out_axes=(0, 0, None),
            axis_name="batch",
        )(latents_ic, ts, None, node_args, ode_solver, adaptive, ode_steps, model_state)
    unrolls = (
        eqx.filter_jit(
            eqx.filter_vmap(
                eqx.filter_vmap(
                    eqx.filter_vmap(
                        model.decoder.call_coords_latent, in_axes=(0, None)
                    ),
                    in_axes=(None, 0),
                ),
                in_axes=(0, 0),
            )
        )(coords, ls)
        .transpose(0, 1, 3, 2)
        .reshape(trajs.shape[0], t1 - t0, 9, *trajs.shape[3:])
    )
    _, unrolls = eqx.filter_vmap(eqx.filter_vmap(macro))(unrolls)

    if loss == "mse":
        errors = eqx.filter_jit(eqx.filter_vmap(mean_squared_error))(
            unrolls, trajs[:, t0:t1]
        )
    elif loss == "nmse":
        errors = eqx.filter_jit(eqx.filter_vmap(normalized_mse))(
            unrolls, trajs[:, t0:t1]
        )
    unrolls = np.array(unrolls)
    errors = np.array(errors)
    if return_latent:
        return unrolls, errors, ls
    return unrolls, errors


def subsample_coords(full_coords, res_ratio, random=False, key=jr.PRNGKey(0)):
    ratio = res_ratio ** full_coords.shape[1]
    print(ratio)
    if random:
        key, subkey = jr.split(key)
        indices = jr.permutation(
            subkey, jnp.arange(full_coords.shape[0]), independent=True
        )[: int(full_coords.shape[0] * ratio)]
    else:
        indices = jnp.arange(0, full_coords.shape[0], int(1 / ratio))
    return full_coords[indices], indices
