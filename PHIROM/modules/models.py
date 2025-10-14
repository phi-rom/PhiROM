"""
This file contains the models for Phi-ROM (and other Neural ODE based baselines) along with the training and inference models for the CROM framework. 
The models here bring the blocks in the base.py file together to form the complete model.
"""

from enum import Enum
from typing import Callable, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optis
from equinox import field
from jaxtyping import Array, PRNGKeyArray

from .base import DecoderMLP, EncoderMLP, HyperDecoder, NeuralODE


class DiffMode(str, Enum):
    FINITE_DIFF = "fd"
    AUTOMATIC = "ad"


class InversionMode(str, Enum):
    ENCODER = "encoder"
    ENCODER_INVERSION = "encoder_inversion"
    INVERSION = "inversion"
    LATENT_MEM = "latent_memory"


class DecoderArchEnum(str, Enum):
    MLP = "mlp"
    HYPER = "hyper"


class CROMOffline(eqx.Module):
    """
    Offline (training) model for CROM.
    """

    encoder: eqx.Module
    decoder: DecoderMLP
    autodecoder: bool = field(static=True)
    latent_dim: int = field(static=True)
    num_sensors: int = field(static=True)
    field_dim: int = field(static=True)
    spatial_dim: int = field(static=True)

    def __init__(
        self,
        latent_dim: int,
        num_sensors: int,
        field_dim: int,
        spatial_dim: int,
        autodecoder: bool = True,
        encoder: eqx.Module = None,
        decoder: eqx.Module = None,
        decoder_arch: DecoderArchEnum = DecoderArchEnum.MLP,
        *,
        key: PRNGKeyArray,
        **kwargs
    ):
        self.latent_dim = latent_dim
        self.num_sensors = num_sensors
        self.field_dim = field_dim
        self.spatial_dim = spatial_dim
        # if encoder is None:
        #     key, subkey = jax.random.split(key)
        #     self.encoder = EncoderMLP(field_dim, num_sensors, latent_dim, **kwargs, key=subkey)
        # else:
        #     self.encoder = encoder
        if autodecoder:
            encoder = None
        else:
            if encoder is None:
                key, subkey = jax.random.split(key)
                encoder = EncoderMLP(
                    field_dim, num_sensors, latent_dim, **kwargs, key=subkey
                )
        self.encoder = encoder
        self.autodecoder = autodecoder
        if decoder is None:
            key, subkey = jax.random.split(key)
            if decoder_arch == DecoderArchEnum.MLP:
                self.decoder = DecoderMLP(
                    spatial_dim, latent_dim, out_dim=field_dim, **kwargs, key=subkey
                )
            elif decoder_arch == DecoderArchEnum.HYPER:
                self.decoder = HyperDecoder(
                    spatial_dim, latent_dim, field_dim, **kwargs, key=subkey
                )
            else:
                raise ValueError("Invalid decoder architecture")
        else:
            self.decoder = decoder

    def __call__(self, x: Array, sensor_coords) -> Array:
        """
        Forward pass. x should be in channel-first format (field_dim, num_sensors).

        Args:
            x: Array, input tensor in channel-first format (field_dim, num_sensors)

        Returns:
            Array, reconstructed output tensor in channel-first format (field_dim, num_sensors)
        """
        x = self.encoder(x)
        x = eqx.filter_vmap(self.decoder.call_coords_latent, in_axes=(0, None))(
            sensor_coords, x
        )
        return x


class CROMOnline(eqx.Module):
    """
    Online (inference) model for CROM.
    """

    encoder: eqx.Module
    decoder: eqx.Module
    latent_dim: int = field(static=True)
    num_integration_points: int = field(static=True)
    field_dim: int = field(static=True)
    spatial_dim: int = field(static=True)
    solver: optis.AbstractGradientDescent
    max_iterations: int = field(static=True)
    autodecoder: bool = field(static=True)
    evolve_fn: Callable

    def __init__(
        self,
        evolve_fn: Callable,
        latent_dim: int,
        num_integration_points: int,
        field_dim: int,
        spatial_dim: int,
        encoder: eqx.Module,
        decoder: eqx.Module,
        solver=optis.GradientDescent(rtol=1e-6, atol=1e-6, learning_rate=1e-3),
        max_iterations: int = 20,
        autodecoder: bool = False,
        **kwargs
    ):
        self.latent_dim = latent_dim
        self.num_integration_points = num_integration_points
        self.field_dim = field_dim
        self.spatial_dim = spatial_dim
        self.encoder = encoder
        self.decoder = decoder
        self.solver = solver
        self.max_iterations = max_iterations
        self.autodecoder = autodecoder
        self.evolve_fn = evolve_fn

    def __call__(self, coord: Array, latent: Array):
        """
        Forward pass. latent should be in channel-first format (latent_dim, ).

        Args:
            latent: Array, latent tensor in channel-first format (latent_dim, )
            coords: Array, integration coordinates (num_integration_points, spatial_dim)

        Returns:
            Array, reconstructed output tensor in channel-first format (field_dim, num_integration_points)
        """
        x = jnp.concatenate([coord, latent], axis=-1)
        x = self.decoder(x)
        return x

    def inversion_to_latent(
        self,
        field: Array,
        init_guess: Array,
        integration_coords: Array,
        decoder,
        max_steps=None,
    ) -> Array:
        """
        Second-order inversion step for CROM.

        Args:
            field: Array, solution at a time step n (num_integration_points, field_dim)
            init_guess: initial guess for the latent tensor (latent_dim, )
            integration_coords: Array, integration coordinates (num_integration_points, spatial_dim)

        Returns:
            Array, latent tensor for time step n (latent_dim, )
        """
        if max_steps is None:
            max_steps = self.max_iterations
        latent = optis.least_squares(
            self.inversion_reconstruction_residual,
            self.solver,
            init_guess,
            args=[integration_coords, field, decoder],
            max_steps=max_steps,
            throw=False,
        ).value
        return latent

    @staticmethod
    def inversion_reconstruction_residual(latent, args):
        integration_coords, field, decoder = args
        field_reconstructed = decoder(integration_coords, latent).T.reshape(field.shape)
        return field_reconstructed - field

    def solve_full_order(
        self,
        field_initial: Array,
        integration_coords: Array,
        delta_t: float,
        delta_x: Union[float, Array],
        args_solver,
        num_steps: int,
        return_latents: bool = False,
        inversion_mode: InversionMode = InversionMode.INVERSION,
        diff_mode: DiffMode = DiffMode.FINITE_DIFF,
        init_guess=None,
    ) -> Array:
        """
        Full-order inference model for CROM.
        """
        if not self.autodecoder:
            encoder = eqx.filter_jit(self.encoder)
        decoder = eqx.filter_jit(
            eqx.filter_vmap(self.decoder.call_coords_latent, in_axes=(0, None))
        )
        evolve = eqx.filter_jit(self.evolve_fn)
        sols = []
        if not self.autodecoder:
            latent = encoder(field_initial.reshape((self.field_dim, -1)))
        else:
            if init_guess is None:
                latent = jnp.zeros((self.latent_dim,))
            else:
                latent = init_guess
        latent = self.inversion_to_latent(
            field_initial, latent, integration_coords, decoder, max_steps=None
        )
        field = decoder(integration_coords, latent).T.reshape(field_initial.shape)
        sols.append(field)
        if return_latents:
            latents = [latent]
        for _ in range(num_steps - 1):
            if diff_mode == DiffMode.AUTOMATIC:
                field = evolve(
                    self.decoder, integration_coords, latent, delta_t, *args_solver
                )
            elif diff_mode == DiffMode.FINITE_DIFF:
                field = evolve(field, delta_t, delta_x, *args_solver)
            else:
                raise ValueError("Invalid diff mode")

            if inversion_mode == InversionMode.ENCODER:
                latent = encoder(field.reshape((self.field_dim, -1)))
            elif inversion_mode == InversionMode.ENCODER_INVERSION:
                latent = encoder(field.reshape((self.field_dim, -1)))
                latent = self.inversion_to_latent(
                    field, latent, integration_coords, decoder
                )
            elif inversion_mode == InversionMode.INVERSION:
                latent = self.inversion_to_latent(
                    field, latent, integration_coords, decoder
                )
            elif inversion_mode == InversionMode.LATENT_MEM:
                raise NotImplementedError(
                    "Latent memory not implemented for full order model"
                )
            else:
                raise ValueError("Invalid inversion mode")

            field = decoder(integration_coords, latent).T.reshape(field_initial.shape)
            sols.append(field)
            if return_latents:
                latents.append(latent)
        if return_latents:
            return jnp.stack(sols, axis=0), latents
        return jnp.stack(sols, axis=0)

    def solver_reduced_order(
        self,
        field_initial: Array,
        integration_coords: Array,
        delta_t: float,
        solver_args,
        num_steps: int,
        return_latents: bool = False,
        inversion_mode: InversionMode = InversionMode.INVERSION,
        diff_mode: DiffMode = DiffMode.AUTOMATIC,
        init_guess=None,
        integration_coords_ic=None,
        boundary_coords: Array = None,
        boundary_value: float = 0.0,
        initial_coords=None,
    ) -> Array:
        """
        Reduced-order (subsampled) inference model for CROM - Only supports AD solver.
        """
        assert not (
            self.autodecoder
            and (
                inversion_mode == InversionMode.ENCODER_INVERSION
                or inversion_mode == InversionMode.ENCODER
            )
        ), "Cannot use encoder inversion with autodecoder"
        if boundary_coords is not None:
            num_boundary_coords = boundary_coords.shape[0]
            integration_coords = jnp.concatenate(
                [integration_coords, boundary_coords], axis=0
            )
            zeros = jnp.zeros((num_boundary_coords, self.field_dim)).T
            field_initial = jnp.concatenate([field_initial, zeros], axis=1)
        if not self.autodecoder:
            encoder = eqx.filter_jit(self.encoder)
        decoder = eqx.filter_jit(
            eqx.filter_vmap(self.decoder.call_coords_latent, in_axes=(0, None))
        )
        evolve = eqx.filter_jit(self.evolve_fn)
        if (
            inversion_mode == InversionMode.ENCODER
            or inversion_mode == InversionMode.ENCODER_INVERSION
        ):
            latent = encoder(field_initial.reshape((self.field_dim, -1)))
            latent = self.inversion_to_latent(
                field_initial, latent, integration_coords_ic, decoder, max_steps=None
            )
        else:
            if init_guess is None:
                latent = jnp.zeros((self.latent_dim,))
            else:
                latent = init_guess
            if initial_coords is None:
                latent = self.inversion_to_latent(
                    field_initial, latent, integration_coords, decoder, max_steps=None
                )
            else:
                latent = self.inversion_to_latent(
                    field_initial, latent, initial_coords, decoder, max_steps=None
                )
        sols = []
        if return_latents:
            latents = [latent]
        for _ in range(num_steps):
            field_2, field_1 = evolve(
                self.decoder, integration_coords, latent, delta_t, *solver_args
            )
            # field_1, field_2 = field_1.T, field_2.T
            sols.append(field_1)
            if boundary_coords is not None:
                field_2 = field_2.at[:, -num_boundary_coords:, ...].set(boundary_value)
            latent = self.inversion_to_latent(
                field_2, latent, integration_coords, decoder
            )
            if return_latents:
                latents.append(latent)
        if return_latents:
            return jnp.stack(sols, axis=0), latents
        return jnp.stack(sols, axis=0)


class NodeROM(CROMOffline):
    """
    Neural ODE based models (Phi-ROM and DINo).
    """

    node: NeuralODE

    def __init__(
        self,
        latent_dim: int,
        num_sensors: int,
        field_dim: int,
        spatial_dim: int,
        decoder: eqx.Module = None,
        decoder_arch: DecoderArchEnum = DecoderArchEnum.MLP,
        node: NeuralODE = None,
        *,
        key: PRNGKeyArray,
        **kwargs
    ):
        key, subkey = jax.random.split(key)
        super().__init__(
            latent_dim,
            num_sensors,
            field_dim,
            spatial_dim,
            True,
            None,
            decoder,
            decoder_arch,
            **kwargs,
            key=subkey
        )
        if node is None:
            self.node = NeuralODE(latent_dim, **kwargs.get("node_kwargs", {}), key=key)
        else:
            self.node = node
