"""
This file contains the trainer class for Phi-ROM.
"""

from enum import Enum
from functools import partial
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax as optx
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, PRNGKeyArray

from PHIROM.pde.data_utils import iterator_sharded_prefetch

from ..modules.models import NodeROM
from ..pde.data_utils import DataLoader
from .callbacks import *


class NodeTrainingModeEnum(str, Enum):
    """
    Enum for training modes for NODE.

    Attributes:
        LABELS: Use labels to train the NODE. Not used in Phi-ROM.
        JACOBIAN_PSI: Jacobian x Psi training mode.
        JACOBIAN_INVERSE: Jacobian inverse training mode. Phi-ROM uses this mode.
        ZERO: Only minimize the reconstruction loss. Used for testing.

    """

    LABELS = "labels"
    JACOBIAN_PSI = "jacobian_psi"
    JACOBIAN_INVERSE = "jacobian_inverse"
    ZERO = "zero"

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


def apply_pinv_qr(A, b):
    Q, R = jnp.linalg.qr(A)
    return solve_triangular(R, Q.T @ b)


class PhiROMTrainer:
    """
    Trainer class for the Physics-Informed ROM (PhiROM) model.

    args:
        model: The model to be trained. Should be an instance of NodeROM.
        model_state: The initial state of the model.
        optimizer: The optax optimizer for the decoder.
        optimizer_node: The optimizer for training the dynamics network. Must be None for Phi-ROM.
        optimizer_latent: The optimizer for the latent memory. Must be None for Phi-ROM.
        node_training_mode: The training mode for the NODE.
        opt_state: The initial state of the optimizer.
        opt_state_node: The initial state of the NODE optimizer.
        opt_state_latent: The initial state of the latent memory optimizer.
        loss: The loss function to be used. Default is "nmse".
        evolve_fn: Solver function for evolving the PDE.
        evolve_start: Number of warmup epochs before minimizing the dynamics loss.
        num_trajectories: Number of trajectories. Default is None.
        num_time_steps: Number of time steps. Default is None.
        latent_dim: Dimension of the latent space. Default is None.
        latent_memory: Latent memory array. Default is None.
        callbacks: List of callbacks to be used during training. Default is empty list.
        gamma: Hyperreduction factor for the dynamics loss. Default is 0.1.
        devices: List of devices to be used for training. Default is all available devices.
        use_ad: Whether to use AD for calculating the PDE residuals. If True, the trained model will be PINN-ROM. Default is False.
        xlb_macro: XLB macro function for calculating the velocity from the mesoscopic populations. Only used for LBM data. Default is None.
        xlb_second_moment: XLB second moment function for calculating the density from the mesoscopic populations. Only used for LBM data. Default is None.
        loss_lambda: Weight for the reconstruction and dynamics losses. Default is 0.8. (lambda * reconstruction + (1-lambda) * dynamics)

    """

    def __init__(
        self,
        model: NodeROM,
        model_state: PyTree,
        optimizer: optx.GradientTransformation,
        optimizer_node: optx.GradientTransformation,
        optimizer_latent: optx.GradientTransformation,
        node_training_mode: NodeTrainingModeEnum,
        opt_state: optx.OptState = None,
        opt_state_node: optx.OptState = None,
        opt_state_latent: optx.OptState = None,
        loss: str = "nmse",
        evolve_fn: Callable = None,
        evolve_start: int = 0,
        num_trajectories: int = None,
        num_time_steps: int = None,
        latent_dim: int = None,
        latent_memory: Array = None,
        callbacks: Sequence[Callback] = [],
        gamma: float = 0.1,
        devices: list[jax.Device] = jax.devices(),
        use_ad: bool = False,
        xlb_macro: Callable = None,
        xlb_second_moment: Callable = None,
        loss_lambda: float = 0.8,
        *,
        key: PRNGKeyArray,
    ):

        assert len(devices) > 0, "No accelerator devices found"
        self.model = model
        self.optimizer = optimizer
        self.optimizer_latent = optimizer_latent
        self.optimizer_node = optimizer_node
        self.curr_epoch = 0
        self.callbacks = callbacks
        self.loss = loss
        self.gamma = gamma
        self.evolve_fn = evolve_fn
        self.evolve_start = evolve_start
        self.devices = devices
        self.key = key
        self.use_ad = use_ad
        self.history = {"loss_reconstruction": [], "loss_time_stepping": []}
        if latent_memory is None:
            latent_memory = jnp.zeros((num_trajectories, num_time_steps, latent_dim))
        self.latent_memory = latent_memory
        if opt_state is None:
            if self.optimizer_node is None:
                if self.optimizer_latent is None:
                    opt_state = self.optimizer.init(
                        (
                            eqx.filter(model.decoder, eqx.is_array_like),
                            eqx.filter(model.node, eqx.is_array_like),
                            self.latent_memory,
                        )
                    )
                    opt_state_latent = None
                    opt_state_node = None
                else:
                    opt_state = self.optimizer.init(
                        (
                            eqx.filter(model.decoder, eqx.is_array_like),
                            eqx.filter(model.node, eqx.is_array_like),
                        )
                    )
                    opt_state_latent = self.optimizer_latent.init(latent_memory)
                    opt_state_node = None
            else:
                if optimizer_latent is None:
                    opt_state = self.optimizer.init(
                        (
                            eqx.filter(model.decoder, eqx.is_array_like),
                            self.latent_memory,
                        )
                    )
                    opt_state_latent = None
                    opt_state_node = self.optimizer_node.init(
                        (
                            eqx.filter(model.decoder, eqx.is_array_like),
                            eqx.filter(model.node, eqx.is_array_like),
                            self.latent_memory,
                        )
                    )
                else:
                    opt_state = self.optimizer.init(
                        eqx.filter(model.decoder, eqx.is_array_like)
                    )
                    opt_state_latent = self.optimizer_latent.init(latent_memory)
                    opt_state_node = self.optimizer_node.init(
                        eqx.filter(model.node, eqx.is_array_like)
                    )

        self.opt_state = opt_state
        self.opt_state_latent = opt_state_latent
        self.opt_state_node = opt_state_node
        self.node_training_mode = node_training_mode
        self.gamma = gamma
        self.model_state = model_state
        self.macro = xlb_macro
        self.second_moment = xlb_second_moment
        self.loss_lambda = loss_lambda

    @staticmethod
    def _loss_recons(
        decoder: eqx.Module,
        model_state: PyTree,
        latent_memory: ArrayLike,
        traj_indices: ArrayLike,
        time_indices: ArrayLike,
        fields: ArrayLike,
        spatial_coords: ArrayLike,
        loss: str,
    ):
        """
        Reconstruction loss.

        Args:
            decoder: The decoder model.
            model_state: The state of the model.
            latent_memory: The latent memory array.
            traj_indices: Indices of the trajectories corresponding to the snapshots in the batch.
            time_indices: Indices of the time steps corresponding to the snapshots in the batch.
            fields: The true fields.
            spatial_coords: The spatial coordinates to reconstruct the fields at.
            loss: The loss function to be used. One of ['nmse', 'mse'].
        """

        decoder = eqx.filter_vmap(
            eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))
        )
        latents = latent_memory[traj_indices, time_indices]
        fields_reconstructed = decoder(spatial_coords, latents)
        fields_reconstructed = jnp.transpose(fields_reconstructed, [0, 2, 1]).reshape(
            fields.shape
        )
        norm_axis = (*range(2, fields.ndim),)

        if loss == "mse":
            loss_reconstruction = jnp.mean(jnp.square(fields_reconstructed - fields))
        elif loss == "nmse":
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(fields_reconstructed - fields, axis=norm_axis)
                / jnp.linalg.norm(fields, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        return loss_reconstruction, (
            model_state,
            {"loss_reconstruction": loss_reconstruction},
        )

    @staticmethod
    def _loss_recons_xlb(
        decoder,
        model_state,
        latent_memory,
        traj_indices,
        time_indices,
        fields,
        spatial_coords,
        loss: str,
        macro,
        second_moment,
    ):
        """
        Reconstruction loss for LBM data using XLB macro and second moment functions.

        Args:
            decoder: The decoder model.
            model_state: The state of the model.
            latent_memory: The latent memory array.
            traj_indices: Indices of the trajectories corresponding to the snapshots in the batch.
            time_indices: Indices of the time steps corresponding to the snapshots in the batch.
            fields: The true fields.
            spatial_coords: The spatial coordinates to reconstruct the fields at.
            loss: The loss function to be used. One of ['nmse', 'mse'].
            macro: Function to compute macroscopic variables from mesoscopic populations.
            second_moment: Function to compute second moment variables from mesoscopic populations.
        """

        print("Training for XLB")
        decoder = eqx.filter_vmap(
            eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))
        )
        macro = eqx.filter_vmap(macro)
        second_moment = eqx.filter_vmap(second_moment)

        latents = latent_memory[traj_indices, time_indices]
        fields_reconstructed = decoder(spatial_coords, latents)
        fields_reconstructed = jnp.transpose(fields_reconstructed, [0, 2, 1])
        fields_reconstructed = jnp.reshape(fields_reconstructed, fields.shape)

        norm_axis = (*range(2, fields.ndim),)

        fields_macro = jnp.concat(macro(fields), axis=1)
        recons_macro = jnp.concat(macro(fields_reconstructed), axis=1)

        fields_moments = second_moment(fields)
        recons_moments = second_moment(fields_reconstructed)

        if loss == "mse":
            loss_reconstruction = jnp.mean(jnp.square(fields_reconstructed - fields))
            macro_loss = jnp.mean(jnp.square(recons_macro - fields_macro))
            macro_loss += jnp.mean(jnp.square(recons_moments - fields_moments))
            macro_loss = 0.5 * macro_loss

        elif loss == "nmse":
            macro_loss = jnp.mean(
                jnp.linalg.norm(recons_macro - fields_macro, axis=norm_axis)
                / jnp.linalg.norm(fields_macro, axis=norm_axis)
            )
            macro_loss += jnp.mean(
                jnp.linalg.norm(recons_moments - fields_moments, axis=norm_axis)
                / jnp.linalg.norm(fields_moments, axis=norm_axis)
            )
            macro_loss = 0.5 * macro_loss
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(fields_reconstructed - fields, axis=norm_axis)
                / jnp.linalg.norm(fields, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        loss_reconstruction = loss_reconstruction + macro_loss
        return loss_reconstruction, (
            model_state,
            {"loss_reconstruction": loss_reconstruction},
        )

    @staticmethod
    def _loss_jac_inverse(
        node_latent,
        decoder,
        model_state,
        traj_indices: Array,
        time_indices: Array,
        fields: Array,
        spatial_coords: Array,
        time_coords: Array,
        dt: float,
        dx: float,
        solver_fn: Callable,
        solver_args: list,
        node_args: list,
        loss: str,
        gamma: float,
        use_ad: bool,
        key: PRNGKeyArray,
    ):
        """
        Jacobian inverse training mode. Used for Phi-ROM and PINN-ROM.

        Args:
            node_latent: Tuple of the NODE model and the latent memory array.
            decoder: The decoder model.
            model_state: The state of the model.
            traj_indices: Indices of the trajectories corresponding to the snapshots in the batch.
            time_indices: Indices of the time steps corresponding to the snapshots in the batch.
            fields: The true fields.
            spatial_coords: The spatial coordinates to reconstruct the fields at.
            time_coords: The temporal coordinates corresponding to the snapshots in the batch.
            dt: Time step size.
            dx: Spatial step size.
            solver_fn: Function to compute the PDE residuals. If use_ad is False, should take fields as input. If use_ad is True, should take decoder, spatial_coords, and latents as input.
            solver_args: Additional arguments for the solver function.
            node_args: Additional arguments for the NODE function.
            loss: The loss function to be used. One of ['nmse', 'mse'].
            gamma: Hyperreduction factor for the dynamics loss.
            use_ad: Whether to use auto-diff for calculating the PDE residuals. If True, the trained model will be PINN-ROM.
            key: PRNG key for random number generation.

        """

        print("Training NODE with inverse jacobian")

        node, latent_memory = node_latent

        psi = eqx.filter_vmap(
            node.mlp, in_axes=(None, 0, 0, None), out_axes=(0, None), axis_name="batch"
        )
        if not use_ad:
            solver = eqx.filter_vmap(
                solver_fn, in_axes=(0, 0, 0) + (0,) * len(solver_args)
            )
        else:
            solver = eqx.filter_vmap(
                solver_fn, in_axes=(None, 0, 0, 0) + (0,) * len(solver_args)
            )

        latents = latent_memory[traj_indices, time_indices]

        def dummy_field_reconstructor(latent, coords, decoder):
            field_fn = eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))
            field = field_fn(coords, latent).T
            return field

        if not use_ad:  # PhiROM
            recons_fileds = eqx.filter_vmap(
                dummy_field_reconstructor, in_axes=(0, 0, None)
            )(
                latents, spatial_coords, decoder
            )  # (B, N, D)         # Reconstructed fields at all spatial coords
            recons_fileds = jnp.reshape(recons_fileds, fields.shape)
            residuals = solver(
                recons_fileds, dt, dx, *solver_args
            )  # Residuals at all spatial coords
        else:  # PINN-ROM
            print("Training with PINN")
            residuals = solver(
                decoder, spatial_coords, latents, dt, *solver_args
            )  # Residuals at all spatial coords

        latent_dot, model_state = psi(
            None, latents, node_args, model_state
        )  # d alpha / dt computed by the dynamics network

        # Hyper-reduction
        key, subkey = jax.random.split(key)
        coords_indices = jnp.arange(spatial_coords.shape[1])
        coords_indices = jnp.tile(coords_indices, (spatial_coords.shape[0], 1))
        coords_indices = jax.random.permutation(
            subkey, coords_indices, axis=1, independent=True
        )
        coords_indices = coords_indices[:, : int(gamma * spatial_coords.shape[1])]
        take = partial(jnp.take, axis=0, unique_indices=True)
        sub_coords = jax.vmap(take)(
            spatial_coords, coords_indices
        )  # Hyper-reduced spatial coords for computing the Jacobian
        print("Hyper-reduced inversion coords shape:", sub_coords.shape)

        batch_size, latent_size = latents.shape
        jacs = eqx.filter_vmap(
            eqx.filter_jacfwd(dummy_field_reconstructor), in_axes=(0, 0, None)
        )(
            latents, sub_coords, decoder
        )  # Jacobian of the reconstructed fields at the hyper-reduced spatial coords w.r.t. the latent variables

        residuals = jnp.reshape(residuals, shape=(batch_size, fields.shape[1], -1))
        take = partial(jnp.take, axis=-1, unique_indices=True)
        residuals = jax.vmap(take)(residuals, coords_indices)
        residuals = jnp.reshape(
            residuals, shape=(batch_size, -1)
        )  # Residuals (from the solver) at the hyper-reduced spatial coords
        jacs = jnp.reshape(jacs, shape=(batch_size, -1, latent_size))
        rhs = eqx.filter_vmap(apply_pinv_qr, in_axes=(0))(
            jacs, residuals
        )  # d alpha / dt computed from the pseudo-inverse of the Jacobian times the residuals

        if loss == "mse":
            loss_node = jnp.mean(jnp.square(latent_dot - rhs))
        elif loss == "nmse":
            loss_node = jnp.mean(
                jnp.linalg.norm(latent_dot - rhs, axis=-1)
                / jnp.linalg.norm(jax.lax.stop_gradient(rhs), axis=-1)
            )
        return loss_node, (model_state, {"loss_time_stepping": loss_node})

    @staticmethod
    def _loss_jac_x_psi(
        node_latent,
        decoder,
        model_state,
        traj_indices: Array,
        time_indices: Array,
        fields: Array,
        spatial_coords: Array,
        time_coords: Array,
        dt: float,
        dx: float,
        solver_fn: Callable,
        solver_args: list,
        node_args: list,
        loss: str,
        gamma: float,
        use_ad: bool,
        key: PRNGKeyArray,
    ):
        """
        Jacobian x Psi training mode. DEPRECATED: Not used in Phi-ROM.
        """

        print("DEPRECATED: This training mode is not used in Phi-ROM!")
        print("Training NODE with Jacobian x Psi")
        node, latent_memory = node_latent

        psi = eqx.filter_vmap(
            node.mlp, in_axes=(None, 0, 0, None), out_axes=(0, None), axis_name="batch"
        )
        if not use_ad:
            solver = eqx.filter_vmap(
                solver_fn, in_axes=(0, 0, 0) + (0,) * len(solver_args)
            )
        else:
            solver = eqx.filter_vmap(
                solver_fn, in_axes=(None, 0, 0, 0) + (0,) * len(solver_args)
            )
        norm_axis = (*range(2, fields.ndim),)
        latents = latent_memory[traj_indices, time_indices]

        def dummy_field_reconstructor(latent, coords, decoder):
            field_fn = eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))
            field = field_fn(coords, latent).T
            field = jnp.reshape(field, fields.shape[1:])
            return field

        def jvp_residual(latent, coords, latent_dot, decoder):
            return eqx.filter_jvp(
                dummy_field_reconstructor,
                (latent, coords),
                (latent_dot, None),
                decoder=decoder,
            )

        latent_dot, model_state = psi(None, latents, node_args, model_state)
        fields_reconstructed, jac_residual_product = eqx.filter_vmap(
            jvp_residual, in_axes=(0, 0, 0, None)
        )(latents, spatial_coords, latent_dot, decoder)

        if not use_ad:
            residuals = solver(fields_reconstructed, dt, dx, *solver_args)
        else:
            residuals = solver(decoder, spatial_coords, latents, dt, *solver_args)

        if loss == "mse":
            loss_node = jnp.mean(jnp.square(jac_residual_product - residuals))
        elif loss == "nmse":
            loss_node = jnp.mean(
                jnp.linalg.norm(jac_residual_product - residuals, axis=norm_axis)
                / jnp.linalg.norm(jax.lax.stop_gradient(residuals), axis=norm_axis)
            )

        return loss_node, (model_state, {"loss_time_stepping": loss_node})

    @staticmethod
    def loss_jac_zero(
        node_latent,
        decoder,
        model_state,
        traj_indices: Array,
        time_indices: Array,
        fields: Array,
        spatial_coords: Array,
        time_coords: Array,
        dt: float,
        dx: float,
        solver_fn: Callable,
        solver_args: list,
        node_args: list,
        loss: str,
        gamma: float,
        use_ad: bool,
        key: PRNGKeyArray,
    ):
        """
        Zero NODE training mode. Used for training with only reconstruction loss.
        """

        print("No Jacobian Training")
        return 0.0, (model_state, {"loss_time_stepping": 0.0})

    def _inner_step(
        self,
        model,
        model_state,
        opt_state_decoder,
        opt_state_node,
        opt_state_latent,
        latent_memory,
        trajectories,
        spatial_coords,
        temporal_coords,
        dt,
        dx,
        traj_indices,
        time_indices,
        solver_fn,
        solver_args,
        node_args,
        gamma,
        sharding_model,
        sharding_data,
        stop_memory_gradient,
        *,
        key,
    ):
        """
        Single inner training step.

        Args:
            model: The model to be trained.
            model_state: The state of the model.
            opt_state_decoder: The state of the decoder optimizer.
            opt_state_node: The state of the NODE optimizer.
            opt_state_latent: The state of the latent memory optimizer.
            latent_memory: The latent memory array.
            trajectories: The true fields.
            spatial_coords: The spatial coordinates to reconstruct the fields at.
            temporal_coords: The temporal coordinates corresponding to the snapshots in the batch.
            dt: Time step size.
            dx: Spatial step size.
            traj_indices: Indices of the trajectories corresponding to the snapshots in the batch.
            time_indices: Indices of the time steps corresponding to the snapshots in the batch.
            solver_fn: Function to compute the PDE residuals. If use_ad is False, should take fields as input. If use_ad is True, should take decoder, spatial_coords, and latents as input.
            solver_args: Additional arguments for the solver function.
            node_args: Additional arguments for the NODE function.
            gamma: Hyperreduction factor for the dynamics loss.
            sharding_model: Sharding specification for the model parameters.
            sharding_data: Sharding specification for the data arrays.
            stop_memory_gradient: Multiplier for stopping gradient through latent memory in dynamics loss. 0.0 stops gradient, 1.0 allows gradient. Used for stabilizing training.
            key: PRNG key for random number generation.

        Returns:
            Updated model, model state, optimizer states, and latent memory after one training step along with the loss dictionary.

        """

        (
            model,
            model_state,
            latent_memory,
            opt_state_decoder,
            opt_state_node,
            opt_state_latent,
        ) = eqx.filter_shard(
            (
                model,
                model_state,
                latent_memory,
                opt_state_decoder,
                opt_state_node,
                opt_state_latent,
            ),
            sharding_model,
        )
        (
            trajectories,
            spatial_coords,
            temporal_coords,
            dt,
            dx,
            traj_indices,
            time_indices,
            solver_args,
        ) = eqx.filter_shard(
            (
                trajectories,
                spatial_coords,
                temporal_coords,
                dt,
                dx,
                traj_indices,
                time_indices,
                solver_args,
            ),
            sharding_data,
        )

        node = model.node
        decoder = model.decoder

        if self.optimizer_node is None and self.optimizer_latent is None:
            print("training everything together")

            if self.macro is None:

                def dummy_recons_loss(
                    dec_node_latent,
                    model_state,
                    traj_indices,
                    time_indices,
                    trajectories,
                    spatial_coords,
                    loss,
                ):
                    decoder, node, latent_memory = dec_node_latent
                    return self._loss_recons(
                        decoder,
                        model_state,
                        latent_memory,
                        traj_indices,
                        time_indices,
                        trajectories,
                        spatial_coords,
                        loss,
                    )

            else:

                def dummy_recons_loss(
                    dec_node_latent,
                    model_state,
                    traj_indices,
                    time_indices,
                    trajectories,
                    spatial_coords,
                    loss,
                ):
                    decoder, node, latent_memory = dec_node_latent
                    return self._loss_recons_xlb(
                        decoder,
                        model_state,
                        latent_memory,
                        traj_indices,
                        time_indices,
                        trajectories,
                        spatial_coords,
                        loss,
                        self.macro,
                        self.second_moment,
                    )

            def dumm_jac_loss(
                dec_node_latent,
                model_state,
                traj_indices,
                time_indices,
                fields,
                spatial_coords,
                time_coords,
                dt,
                dx,
                solver_fn,
                solver_args,
                node_args,
                loss,
                gamma,
                key,
            ):
                decoder, node, latent_memory = dec_node_latent
                if self.node_training_mode == NodeTrainingModeEnum.JACOBIAN_PSI:
                    dyn_loss_fn = self._loss_jac_x_psi
                elif self.node_training_mode == NodeTrainingModeEnum.JACOBIAN_INVERSE:
                    dyn_loss_fn = self._loss_jac_inverse
                elif self.node_training_mode == NodeTrainingModeEnum.ZERO:
                    dyn_loss_fn = self.loss_jac_zero
                else:
                    raise NotImplementedError(
                        "Unsupported NODE Training Mode {}".format(
                            self.node_training_mode
                        )
                    )

                return dyn_loss_fn(
                    (node, latent_memory),
                    decoder,
                    model_state,
                    traj_indices,
                    time_indices,
                    fields,
                    spatial_coords,
                    time_coords,
                    dt,
                    dx,
                    solver_fn,
                    solver_args,
                    node_args,
                    loss,
                    gamma,
                    self.use_ad,
                    key,
                )

            f_prime_loss_recons = eqx.filter_value_and_grad(
                dummy_recons_loss, has_aux=True
            )
            f_prim_loss_jacobian = eqx.filter_value_and_grad(
                dumm_jac_loss, has_aux=True
            )

            (
                loss_value_recons,
                (model_state, loss_dict),
            ), loss_grad_recons = f_prime_loss_recons(  # Reconstruction loss and its gradients
                (decoder, node, latent_memory),
                model_state,
                traj_indices,
                time_indices,
                trajectories,
                spatial_coords,
                self.loss,
            )
            (
                loss_value_jacobian,
                (model_state, loss_dict_node),
            ), loss_grad_jacobian = f_prim_loss_jacobian(  # Dynamics loss and its gradients
                (decoder, node, latent_memory),
                model_state,
                traj_indices,
                time_indices,
                trajectories,
                spatial_coords,
                temporal_coords,
                dt,
                dx,
                solver_fn,
                solver_args,
                node_args,
                self.loss,
                gamma,
                key,
            )
            loss_grad_jacobian = jax.tree.map(
                lambda p: jnp.where(jnp.isnan(p), jnp.zeros_like(p), p),
                loss_grad_jacobian,
            )
            grads_decoder_rec, grads_node_rec, grads_latent_rec = loss_grad_recons
            grads_decoder_jac, grads_node_jac, grads_latent_jac = loss_grad_jacobian
            (grads_decoder_jac, grads_latent_jac) = (
                jax.tree.map(  # Stop gradient through latent memory in dynamics loss during warmup
                    lambda x: stop_memory_gradient
                    * x,  # stops gradient if stop_memory_gradient = 0.0
                    (grads_decoder_jac, grads_latent_jac),
                )
            )
            grads_decoder, grads_latent, grads_node = (
                jax.tree.map(  # Weighted sum of gradients from reconstruction and dynamics losses
                    lambda x, y: self.loss_lambda * x + (1 - self.loss_lambda) * y,
                    (grads_decoder_rec, grads_latent_rec, grads_node_rec),
                    (grads_decoder_jac, grads_latent_jac, grads_node_jac),
                )
            )
            (
                updates_decoder,
                updates_node,
                updates_latent,
            ), opt_state_decoder = self.optimizer.update(  # Parameter updates for decoder, NODE, and latent memory
                (grads_decoder, grads_node, grads_latent),
                opt_state_decoder,
                params=(decoder, node, latent_memory),
            )
            decoder, node, latent_memory = eqx.apply_updates(
                (decoder, node, latent_memory),
                (updates_decoder, updates_node, updates_latent),
            )
            model = eqx.tree_at(
                lambda model: (model.decoder, model.node), model, (decoder, node)
            )
            return (
                model,
                model_state,
                latent_memory,
                opt_state_decoder,
                opt_state_node,
                opt_state_latent,
                {**loss_dict, **loss_dict_node},
            )
        else:
            raise NotImplementedError("Unsupported optimizer configuration")

    def fit(
        self,
        dataloader_train: DataLoader,
        epochs: int,
        warm_start: bool = True,
        **kwargs,
    ):
        """
        Fit the model to the training data.

        Args:
            dataloader_train: DataLoader for the training data.
            epochs: Number of epochs to train the model.
            warm_start: Whether to continue training from the current epoch. If False, resets the epoch count and history.
            **kwargs: Additional arguments for the callbacks.

        Returns:
            The trained model, model state, optimizer state, and training history.

        """

        if not warm_start:
            self.history = {
                "loss_reconstruction": [],
                "loss_time_stepping": [],
            }
            self.curr_epoch = 0
        print("Training Physics-Informed NODE - Weight: {}".format(self.loss_lambda))
        jitted_inner_step = eqx.filter_jit(self._inner_step, donate="all")
        model = self.model
        model_state = self.model_state
        opt_state_decoder = self.opt_state
        opt_state_latent = self.opt_state_latent
        opt_state_node = self.opt_state_node
        latent_memory = self.latent_memory
        curr_epoch = self.curr_epoch

        # Set up data parallelism for mutli-gpu training
        num_batches = len(dataloader_train)
        num_devices = len(self.devices)
        assert (
            dataloader_train.batch_size % num_devices == 0
        ), "Batch size should be divisible by number of devices"
        mesh = jax.make_mesh((num_devices,), ("shard",), devices=self.devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(("shard",))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)

        (
            model,
            model_state,
            latent_memory,
            opt_state_decoder,
            opt_state_node,
            opt_state_latent,
        ) = eqx.filter_shard(
            (
                model,
                model_state,
                latent_memory,
                opt_state_decoder,
                opt_state_node,
                opt_state_latent,
            ),
            sharding_model,
        )

        for epoch in range(self.curr_epoch, epochs + curr_epoch):
            model = eqx.nn.inference_mode(model, False)
            if epoch >= self.evolve_start:
                memory_grad_multiplier = 1.0  # IF 1, DECODER and LATENT MEMORY GRADIENTS FROM DYNAMICS LOSS ARE NOT STOPPED
            else:
                memory_grad_multiplier = 0.0  # IF 0, DECODER and LATENT MEMORY GRADIENTS FROM DYNAMICS LOSS ARE STOPPED
            mean_reconstruction = 0.0
            mean_time_stepping = 0.0
            prefetcher = iterator_sharded_prefetch(  # Prefetch and shard one batch ahead of time to improve training speed
                iter(dataloader_train), 2, sharding_data
            )
            for batch in prefetcher:
                trajectories = batch.get("data", None)
                temporal_coords = batch.get("t", None)
                time_indices = batch.get("time_idx", None)
                spatial_coords = batch.get("coords", None)
                dt = batch.get("dt", None)
                dx = batch.get("dx", None)
                traj_indices = batch.get("idx", None)
                solver_args = batch.get("solver_args", None)
                node_args = batch.get("node_args", None)
                self.key, subkey = jax.random.split(self.key)
                (
                    model,
                    model_state,
                    latent_memory,
                    opt_state_decoder,
                    opt_state_node,
                    opt_state_latent,
                    loss_dict,
                ) = jitted_inner_step(
                    model,
                    model_state,
                    opt_state_decoder,
                    opt_state_node,
                    opt_state_latent,
                    latent_memory,
                    trajectories,
                    spatial_coords,
                    temporal_coords,
                    dt,
                    dx,
                    traj_indices,
                    time_indices,
                    self.evolve_fn,
                    solver_args,
                    node_args,
                    self.gamma,
                    sharding_model,
                    sharding_data,
                    jnp.array(memory_grad_multiplier),
                    key=subkey,
                )
                mean_reconstruction += loss_dict["loss_reconstruction"]
                mean_time_stepping += loss_dict["loss_time_stepping"]

            mean_reconstruction /= num_batches
            mean_time_stepping /= num_batches
            self.history["loss_reconstruction"].append(mean_reconstruction)
            self.history["loss_time_stepping"].append(mean_time_stepping)
            print(
                f"Epoch {epoch + 1}/{epochs + curr_epoch}, Reconstruction Loss: {mean_reconstruction},\
                  Time Stepping Loss: {mean_time_stepping}"
            )

            model = eqx.nn.inference_mode(model, True)
            self.model = model
            self.model_state = model_state
            self.curr_epoch += 1
            for callback in self.callbacks:
                (
                    self.model,
                    self.model_state,
                    latent_memory,
                    (opt_state_decoder, opt_state_latent, opt_state_node),
                    self.history,
                    kwargs,
                ) = callback(
                    self.model,
                    self.model_state,
                    latent_memory,
                    (opt_state_decoder, opt_state_latent, opt_state_node),
                    self.history,
                    kwargs,
                    epoch,
                )
            self.latent_memory = latent_memory
            self.opt_state = opt_state_decoder
            self.opt_state_latent = opt_state_latent
            self.opt_state_node = opt_state_node

        return self.model, self.model_state, self.opt_state, self.history


class IrregularPhiROMTrainer(PhiROMTrainer):
    """
    Trainer class for the Physics-Informed ROM (PhiROM) model with irregular coordinates.
    Inherits from the base PhiROMTrainer class and overrides the loss functions to handle irregular spatial coordinates.

    """

    @staticmethod
    def _loss_recons(
        decoder,
        model_state,
        latent_memory,
        traj_indices,
        time_indices,
        fields,
        spatial_coords,
        loss: str,
    ):
        """
        Reconstruction loss for irregular spatial coordinates.
        Args:
            decoder: The decoder model.
            model_state: The state of the model.
            latent_memory: The latent memory array.
            traj_indices: Indices of the trajectories corresponding to the snapshots in the batch.
            time_indices: Indices of the time steps corresponding to the snapshots in the batch.
            fields: The true fields.
            spatial_coords: The spatial coordinates to reconstruct the fields at.
            loss: The loss function to be used. One of ['nmse', 'mse'].
        """

        decoder = eqx.filter_vmap(
            eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))
        )
        latents = latent_memory[traj_indices, time_indices]
        fields_reconstructed = decoder(spatial_coords, latents)
        fields_reconstructed = jnp.transpose(
            fields_reconstructed, [0, 2, 1]
        )  # RESHAPE NOT NEEDED FOR IRREGULAR COORDINATES
        norm_axis = -1  # Norm on spatial axis
        print("Irregular Training Shape:", fields.shape)

        if loss == "mse":
            loss_reconstruction = jnp.mean(jnp.square(fields_reconstructed - fields))
        elif loss == "nmse":
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(fields_reconstructed - fields, axis=norm_axis)
                / jnp.linalg.norm(fields, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        return loss_reconstruction, (
            model_state,
            {"loss_reconstruction": loss_reconstruction},
        )

    @staticmethod
    def _loss_recons_xlb(
        decoder,
        model_state,
        latent_memory,
        traj_indices,
        time_indices,
        fields,
        spatial_coords,
        loss: str,
        macro,
        second_moment,
    ):
        raise NotImplementedError("XLB Loss not yet implemented for irregular training")

    @staticmethod
    def _loss_jac_inverse(
        node_latent,
        decoder,
        model_state,
        traj_indices: Array,
        time_indices: Array,
        fields: Array,
        spatial_coords: Array,
        time_coords: Array,
        dt: float,
        dx: float,
        solver_fn: Callable,
        solver_args: list,
        node_args: list,
        loss: str,
        gamma: float,
        use_ad: bool,
        key: PRNGKeyArray,
    ):
        """
        Jacobian inverse training mode for irregular spatial coordinates. Used for Phi-ROM and PINN-ROM
        Args:
            node_latent: Tuple of the NODE model and the latent memory array.
            decoder: The decoder model.
            model_state: The state of the model.
            traj_indices: Indices of the trajectories corresponding to the snapshots in the batch.
            time_indices: Indices of the time steps corresponding to the snapshots in the batch.
            fields: The true fields.
            spatial_coords: The spatial coordinates to reconstruct the fields at.
            time_coords: The temporal coordinates corresponding to the snapshots in the batch.
            dt: Time step size.
            dx: Spatial step size.
            solver_fn: Function to compute the PDE residuals. If use_ad is False, should take fields as input. If use_ad is True, should take decoder, spatial_coords, and latents as input.
            solver_args: Additional arguments for the solver function.
            node_args: Additional arguments for the NODE function.
            loss: The loss function to be used. One of ['nmse', 'mse'].
            gamma: Hyperreduction factor for the dynamics loss.
            use_ad: Whether to use auto-diff for calculating the PDE residuals. If True, the trained model will be PINN-ROM.
            key: PRNG key for random number generation
        """

        print("Training NODE with inverse jacobian")
        node, latent_memory = node_latent

        psi = eqx.filter_vmap(
            node.mlp, in_axes=(None, 0, 0, None), out_axes=(0, None), axis_name="batch"
        )
        if not use_ad:
            solver = eqx.filter_vmap(
                solver_fn, in_axes=(0, 0, 0) + (0,) * len(solver_args)
            )
        else:
            solver = eqx.filter_vmap(
                solver_fn, in_axes=(None, 0, 0, 0) + (0,) * len(solver_args)
            )

        latents = latent_memory[traj_indices, time_indices]

        def dummy_field_reconstructor(latent, coords, decoder):
            field_fn = eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))
            field = field_fn(coords, latent).T
            return field

        if not use_ad:
            recons_fileds = eqx.filter_vmap(
                dummy_field_reconstructor, in_axes=(0, 0, None)
            )(latents, spatial_coords, decoder)
            recons_fileds = jnp.reshape(recons_fileds, fields.shape)
            residuals = solver(recons_fileds, dt, dx, *solver_args)
        else:
            residuals = solver(decoder, spatial_coords, latents, dt, *solver_args)

        latent_dot, model_state = psi(None, latents, node_args, model_state)

        key, subkey = jax.random.split(key)
        coords_indices = jnp.arange(spatial_coords.shape[1])
        coords_indices = jnp.tile(coords_indices, (spatial_coords.shape[0], 1))
        coords_indices = jax.random.permutation(
            subkey, coords_indices, axis=1, independent=True
        )
        coords_indices = coords_indices[:, : int(gamma * spatial_coords.shape[1])]
        take = partial(jnp.take, axis=0, unique_indices=True)
        sub_coords = jax.vmap(take)(spatial_coords, coords_indices)
        print("Subsampled inversion coords shape:", sub_coords.shape)

        batch_size, latent_size = latents.shape
        jacs = eqx.filter_vmap(
            eqx.filter_jacfwd(dummy_field_reconstructor), in_axes=(0, 0, None)
        )(latents, sub_coords, decoder)
        print(jacs.shape)

        residuals = jnp.reshape(residuals, shape=(batch_size, fields.shape[1], -1))
        take = partial(jnp.take, axis=-1, unique_indices=True)
        residuals = jax.vmap(take)(residuals, coords_indices)
        residuals = jnp.reshape(residuals, shape=(batch_size, -1))
        jacs = jnp.reshape(jacs, shape=(batch_size, -1, latent_size))
        rhs = eqx.filter_vmap(apply_pinv_qr, in_axes=(0))(jacs, residuals)

        if loss == "mse":
            loss_node = jnp.mean(jnp.square(latent_dot - rhs))
        elif loss == "nmse":
            loss_node = jnp.mean(
                jnp.linalg.norm(latent_dot - rhs, axis=-1)
                / jnp.linalg.norm(jax.lax.stop_gradient(rhs), axis=-1)
            )
        return loss_node, (model_state, {"loss_time_stepping": loss_node})

    @staticmethod
    def _loss_jac_x_psi(
        node_latent,
        decoder,
        model_state,
        traj_indices: Array,
        time_indices: Array,
        fields: Array,
        spatial_coords: Array,
        time_coords: Array,
        dt: float,
        dx: float,
        solver_fn: Callable,
        solver_args: list,
        node_args: list,
        loss: str,
        gamma: float,
        use_ad: bool,
        key: PRNGKeyArray,
    ):
        raise NotImplementedError(
            "Jac x Psi not implemented for IrregularPhiROMTrainer"
        )

    @staticmethod
    def loss_jac_zero(
        node_latent,
        decoder,
        model_state,
        traj_indices: Array,
        time_indices: Array,
        fields: Array,
        spatial_coords: Array,
        time_coords: Array,
        dt: float,
        dx: float,
        solver_fn: Callable,
        solver_args: list,
        node_args: list,
        loss: str,
        gamma: float,
        use_ad: bool,
        key: PRNGKeyArray,
    ):
        print("No Jacobian Training")
        return 0.0, (model_state, {"loss_time_stepping": 0.0})

    def _inner_step(
        self,
        model,
        model_state,
        opt_state_decoder,
        opt_state_node,
        opt_state_latent,
        latent_memory,
        trajectories,
        trajs_irreg,
        spatial_coords,
        coords_irreg,
        temporal_coords,
        dt,
        dx,
        traj_indices,
        time_indices,
        solver_fn,
        solver_args,
        node_args,
        gamma,
        sharding_model,
        sharding_data,
        stop_memory_gradient,
        *,
        key,
    ):

        (
            model,
            model_state,
            latent_memory,
            opt_state_decoder,
            opt_state_node,
            opt_state_latent,
        ) = eqx.filter_shard(
            (
                model,
                model_state,
                latent_memory,
                opt_state_decoder,
                opt_state_node,
                opt_state_latent,
            ),
            sharding_model,
        )
        (
            trajectories,
            trajs_irreg,
            spatial_coords,
            coords_irreg,
            temporal_coords,
            dt,
            dx,
            traj_indices,
            time_indices,
            solver_args,
        ) = eqx.filter_shard(
            (
                trajectories,
                trajs_irreg,
                spatial_coords,
                coords_irreg,
                temporal_coords,
                dt,
                dx,
                traj_indices,
                time_indices,
                solver_args,
            ),
            sharding_data,
        )

        node = model.node
        decoder = model.decoder

        if self.optimizer_node is None and self.optimizer_latent is None:
            print("training everything together")

            if self.macro is None:

                def dummy_recons_loss(
                    dec_node_latent,
                    model_state,
                    traj_indices,
                    time_indices,
                    trajectories,
                    spatial_coords,
                    loss,
                ):
                    decoder, node, latent_memory = dec_node_latent
                    return self._loss_recons(
                        decoder,
                        model_state,
                        latent_memory,
                        traj_indices,
                        time_indices,
                        trajectories,
                        spatial_coords,
                        loss,
                    )

            else:
                raise NotImplementedError(
                    "XLB Loss not implemented for irregular training"
                )

            def dumm_jac_loss(
                dec_node_latent,
                model_state,
                traj_indices,
                time_indices,
                fields,
                spatial_coords,
                time_coords,
                dt,
                dx,
                solver_fn,
                solver_args,
                node_args,
                loss,
                gamma,
                key,
            ):
                decoder, node, latent_memory = dec_node_latent
                if self.node_training_mode == NodeTrainingModeEnum.JACOBIAN_PSI:
                    dyn_loss_fn = self._loss_jac_x_psi
                elif self.node_training_mode == NodeTrainingModeEnum.JACOBIAN_INVERSE:
                    dyn_loss_fn = self._loss_jac_inverse
                elif self.node_training_mode == NodeTrainingModeEnum.ZERO:
                    dyn_loss_fn = self.loss_jac_zero
                else:
                    raise NotImplementedError(
                        "Unsupported NODE Training Mode {}".format(
                            self.node_training_mode
                        )
                    )

                return dyn_loss_fn(
                    (node, latent_memory),
                    decoder,
                    model_state,
                    traj_indices,
                    time_indices,
                    fields,
                    spatial_coords,
                    time_coords,
                    dt,
                    dx,
                    solver_fn,
                    solver_args,
                    node_args,
                    loss,
                    gamma,
                    self.use_ad,
                    key,
                )

            f_prime_loss_recons = eqx.filter_value_and_grad(
                dummy_recons_loss, has_aux=True
            )
            f_prim_loss_jacobian = eqx.filter_value_and_grad(
                dumm_jac_loss, has_aux=True
            )

            (loss_value_recons, (model_state, loss_dict)), loss_grad_recons = (
                f_prime_loss_recons(
                    (decoder, node, latent_memory),
                    model_state,
                    traj_indices,
                    time_indices,
                    trajs_irreg,
                    coords_irreg,
                    self.loss,
                )
            )
            (loss_value_jacobian, (model_state, loss_dict_node)), loss_grad_jacobian = (
                f_prim_loss_jacobian(
                    (decoder, node, latent_memory),
                    model_state,
                    traj_indices,
                    time_indices,
                    trajectories,
                    spatial_coords,
                    temporal_coords,
                    dt,
                    dx,
                    solver_fn,
                    solver_args,
                    node_args,
                    self.loss,
                    gamma,
                    key,
                )
            )
            loss_grad_jacobian = jax.tree.map(
                lambda p: jnp.where(jnp.isnan(p), jnp.zeros_like(p), p),
                loss_grad_jacobian,
            )
            grads_decoder_rec, grads_node_rec, grads_latent_rec = loss_grad_recons
            grads_decoder_jac, grads_node_jac, grads_latent_jac = loss_grad_jacobian
            (grads_decoder_jac, grads_latent_jac) = jax.tree.map(
                lambda x: stop_memory_gradient * x,
                (grads_decoder_jac, grads_latent_jac),
            )
            grads_decoder, grads_latent, grads_node = jax.tree.map(
                lambda x, y: self.loss_lambda * x + (1 - self.loss_lambda) * y,
                (grads_decoder_rec, grads_latent_rec, grads_node_rec),
                (grads_decoder_jac, grads_latent_jac, grads_node_jac),
            )
            (updates_decoder, updates_node, updates_latent), opt_state_decoder = (
                self.optimizer.update(
                    (grads_decoder, grads_node, grads_latent),
                    opt_state_decoder,
                    params=(decoder, node, latent_memory),
                )
            )
            decoder, node, latent_memory = eqx.apply_updates(
                (decoder, node, latent_memory),
                (updates_decoder, updates_node, updates_latent),
            )
            model = eqx.tree_at(
                lambda model: (model.decoder, model.node), model, (decoder, node)
            )
            print(self.loss_lambda)
            return (
                model,
                model_state,
                latent_memory,
                opt_state_decoder,
                opt_state_node,
                opt_state_latent,
                {**loss_dict, **loss_dict_node},
            )
        else:
            raise NotImplementedError("Unsupported optimizer configuration")

    def fit(
        self,
        dataloader_train: DataLoader,
        epochs: int,
        warm_start: bool = True,
        **kwargs,
    ):
        if not warm_start:
            self.history = {
                "loss_reconstruction": [],
                "loss_time_stepping": [],
            }
            self.curr_epoch = 0
        print("Training Physics-Informed NODE - Weight: {}".format(self.loss_lambda))
        jitted_inner_step = eqx.filter_jit(self._inner_step, donate="all")
        model = self.model
        model_state = self.model_state
        opt_state_decoder = self.opt_state
        opt_state_latent = self.opt_state_latent
        opt_state_node = self.opt_state_node
        latent_memory = self.latent_memory
        curr_epoch = self.curr_epoch

        num_batches = len(dataloader_train)
        num_devices = len(self.devices)
        assert (
            dataloader_train.batch_size % num_devices == 0
        ), "Batch size should be divisible by number of devices"
        mesh = jax.make_mesh((num_devices,), ("shard",), devices=self.devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(("shard",))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)

        (
            model,
            model_state,
            latent_memory,
            opt_state_decoder,
            opt_state_node,
            opt_state_latent,
        ) = eqx.filter_shard(
            (
                model,
                model_state,
                latent_memory,
                opt_state_decoder,
                opt_state_node,
                opt_state_latent,
            ),
            sharding_model,
        )

        for epoch in range(self.curr_epoch, epochs + curr_epoch):
            model = eqx.nn.inference_mode(model, False)
            if epoch >= self.evolve_start:
                memory_grad_multiplier = (
                    1.0  # IF 1, DECODER and LATENT MEMORY GRADIENTS ARE NOT STOPPED
                )
            else:
                memory_grad_multiplier = (
                    0.0  # IF 0, DECODER and LATENT MEMORY GRADIENTS ARE STOPPED
                )
            mean_reconstruction = 0.0
            mean_time_stepping = 0.0
            prefetcher = iterator_sharded_prefetch(
                iter(dataloader_train), 2, sharding_data
            )
            for batch in prefetcher:
                trajectories = batch.get("data", None)
                trajectories_irregular = batch.get("data_irregular", None)
                temporal_coords = batch.get("t", None)
                time_indices = batch.get("time_idx", None)
                spatial_coords = batch.get("coords", None)
                coords_irregular = batch.get("coords_irregular", None)
                dt = batch.get("dt", None)
                dx = batch.get("dx", None)
                traj_indices = batch.get("idx", None)
                solver_args = batch.get("solver_args", None)
                node_args = batch.get("node_args", None)
                self.key, subkey = jax.random.split(self.key)
                (
                    model,
                    model_state,
                    latent_memory,
                    opt_state_decoder,
                    opt_state_node,
                    opt_state_latent,
                    loss_dict,
                ) = jitted_inner_step(
                    model,
                    model_state,
                    opt_state_decoder,
                    opt_state_node,
                    opt_state_latent,
                    latent_memory,
                    trajectories,
                    trajectories_irregular,
                    spatial_coords,
                    coords_irregular,
                    temporal_coords,
                    dt,
                    dx,
                    traj_indices,
                    time_indices,
                    self.evolve_fn,
                    solver_args,
                    node_args,
                    self.gamma,
                    sharding_model,
                    sharding_data,
                    jnp.array(memory_grad_multiplier),
                    key=subkey,
                )
                mean_reconstruction += loss_dict["loss_reconstruction"]
                mean_time_stepping += loss_dict["loss_time_stepping"]

            mean_reconstruction /= num_batches
            mean_time_stepping /= num_batches
            self.history["loss_reconstruction"].append(mean_reconstruction)
            self.history["loss_time_stepping"].append(mean_time_stepping)
            print(
                f"Epoch {epoch + 1}/{epochs + curr_epoch}, Reconstruction Loss: {mean_reconstruction},\
                  Time Stepping Loss: {mean_time_stepping}"
            )

            model = eqx.nn.inference_mode(model, True)
            self.model = model
            self.model_state = model_state
            self.curr_epoch += 1
            for callback in self.callbacks:
                (
                    self.model,
                    self.model_state,
                    latent_memory,
                    (opt_state_decoder, opt_state_latent, opt_state_node),
                    self.history,
                    kwargs,
                ) = callback(
                    self.model,
                    self.model_state,
                    latent_memory,
                    (opt_state_decoder, opt_state_latent, opt_state_node),
                    self.history,
                    kwargs,
                    epoch,
                )
            self.latent_memory = latent_memory
            self.opt_state = opt_state_decoder
            self.opt_state_latent = opt_state_latent
            self.opt_state_node = opt_state_node

        return self.model, self.model_state, self.opt_state, self.history
