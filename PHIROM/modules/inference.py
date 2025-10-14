"""
This file contains basic first-order inversion routines for the auto-decoders.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
import optimistix as optimix
from equinox import filter_jit, filter_vmap
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation, OptState

from .base import BaseDecoder, EncoderMLP, NeuralODE


def find_latent_descent(
    decoder: BaseDecoder,
    field: Array,
    coords: Array,
    latent_dim: int,
    optimizer: GradientTransformation,
    opt_state: OptState = None,
    n_steps=300,
    init_guess: Array = None,
    return_loss=False,
    loss: str = "mse",
):
    """
    Find the latent representation of a field given the field, the coordinates, and the decoder.

    Args:
        decoder: BaseDecoder
            The decoder to use.
        field: Array
            The field to match.
        coords: Array
            The coordinates of the field.
        latent_dim: int
            The latent dimension.
        optimizer: GradientTransformation
            The optimizer to use.
        opt_state: OptState
            The optimizer state.
        n_steps: int
            The number of optimization steps.
        init_guess: Array
            The initial guess for the latent representation.

    Returns:
        latent: Array
            The latent representation.
        opt_state: OptState
            The optimizer state.
        loss: float
            The loss if `return_loss` is `True`.
    """
    if init_guess is None:
        init_guess = jnp.zeros((latent_dim,))
    latent = init_guess
    if opt_state is None:
        opt_state = optimizer.init(latent)

    latents_all = jnp.zeros((n_steps, latent_dim))
    losses = jnp.zeros((n_steps,))

    def loss_fn(decoder, coords, latent, field):
        field_p = filter_vmap(decoder.call_coords_latent, in_axes=(0, None))(
            coords, latent
        ).T
        if loss == "mse":
            err = jnp.mean((field - field_p) ** 2)
        elif loss == "nmse":
            err = jnp.mean(jnp.linalg.norm(field - field_p, axis=1) ** 2) / jnp.mean(
                jnp.linalg.norm(field, axis=1) ** 2
            )
        return err

    @filter_jit
    def update(decoder, coords, latent, field, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn, argnums=2)(
            decoder, coords, latent, field
        )
        updates, opt_state = optimizer.update(grads, opt_state, latent)
        latent = optax.apply_updates(latent, updates)
        return latent, opt_state, loss_val

    for i in range(n_steps):
        latent, opt_state, loss_val = update(decoder, coords, latent, field, opt_state)
        latents_all = latents_all.at[i].set(latent)
        losses = losses.at[i].set(loss_val)
    best_latent = latents_all[jnp.argmin(losses)]
    loss_val = jnp.min(losses)
    # best_latent = latent

    if return_loss:
        return best_latent, opt_state, loss_val
    return best_latent, opt_state


def find_latent_descent_XLB(
    decoder: BaseDecoder,
    macro,
    field: Array,
    coords: Array,
    latent_dim: int,
    optimizer: GradientTransformation,
    opt_state: OptState = None,
    n_steps=300,
    init_guess: Array = None,
    return_loss=False,
):
    """
    Find the latent representation of a field given the field, the coordinates, and the decoder.

    Args:
        decoder: BaseDecoder
            The decoder to use.
        field: Array
            The field to match.
        coords: Array
            The coordinates of the field.
        latent_dim: int
            The latent dimension.
        optimizer: GradientTransformation
            The optimizer to use.
        opt_state: OptState
            The optimizer state.
        n_steps: int
            The number of optimization steps.
        init_guess: Array
            The initial guess for the latent representation.

    Returns:
        latent: Array
            The latent representation.
        opt_state: OptState
            The optimizer state.
        loss: float
            The loss if `return_loss` is `True`.
    """
    if init_guess is None:
        init_guess = jnp.zeros((latent_dim,))
    latent = init_guess
    if opt_state is None:
        opt_state = optimizer.init(latent)

    latents_all = jnp.zeros((n_steps, latent_dim))
    losses = jnp.zeros((n_steps,))

    def loss(decoder, coords, latent, field):
        field_p = filter_vmap(decoder.call_coords_latent, in_axes=(0, None))(
            coords, latent
        ).T
        _, field_p = macro(field_p)
        return jnp.mean((field - field_p) ** 2)

    @filter_jit
    def update(decoder, coords, latent, field, opt_state):
        loss_val, grads = jax.value_and_grad(loss, argnums=2)(
            decoder, coords, latent, field
        )
        updates, opt_state = optimizer.update(grads, opt_state, latent)
        latent = optax.apply_updates(latent, updates)
        return latent, opt_state, loss_val

    for i in range(n_steps):
        latent, opt_state, loss_val = update(decoder, coords, latent, field, opt_state)
        latents_all = latents_all.at[i].set(latent)
        losses = losses.at[i].set(loss_val)
    best_latent = latents_all[jnp.argmin(losses)]
    loss_val = jnp.min(losses)

    if return_loss:
        return best_latent, opt_state, loss_val
    return best_latent, opt_state
