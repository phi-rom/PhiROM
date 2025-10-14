"""
This file contains trainer classes for the baseline models (CROM and DINo).
"""

from enum import Enum
from functools import partial
from typing import Callable, Sequence, Tuple, Union

import diffrax as diff
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import optax as optx
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, PRNGKeyArray

from PHIROM.pde.data_utils import (
    iterator_sharded_prefetch,
    prefetch_to_device,
    torch_iterator_prefetch,
)
from PHIROM.training.train import NodeTrainingModeEnum

from ..modules.models import CROMOffline, CROMOnline, NodeROM
from ..pde.data_utils import (
    DataLoader,
    NumpyLoader,
    TimeWindowDataset,
    TrajectoryDataset,
)
from .callbacks import *


class CROMOfflineTrainer:
    """
    CROM Offline Trainer class.

    Args:
        model: CROMOffline, model to be trained
        optimizer: optax.GradientTransformation, optimizer
        opt_state: optax.OptState, optimizer state
        loss: str, loss function. Should be one of ['nmse', 'mse']
        evolve_fn: Callable, time stepping function for PDE
        evolve_time: bool, whether to evolve in time
        evolve_mode: str, mode of evolution. Should be one of ['label_label', 'pred_pred', 'pred_label']
        evolve_start: int, starting epoch for time evolution loss (ignored if evolve_time is False)
        gamma: float, weight for time evolution loss
        max_evolve_split: int, maximum number of time steps to evolve in time. For example, if `max_evolve_split=1`, one time step is divided into two.
        split_start: int, starting epoch for splitting time steps. Must be greater than or equal to `evolve_start` (ignored if `max_evolve_split` is 0)
        callbacks: Sequence[Callback], list of callbacks
        key: PRNGKeyArray, random key
    """

    def __init__(
        self,
        model: CROMOffline,
        optimizer: optx.GradientTransformation,
        opt_state: optx.OptState = None,
        loss: str = "nmse",
        evolve_time: bool = True,
        evolve_fn: Callable = None,
        evolve_mode: str = "label_label",
        evolve_start: int = 0,
        gamma: float = 0.5,
        max_evolve_split: int = 0,
        split_start: int = 0,
        random_split: bool = True,
        callbacks: Sequence[Callback] = [],
        devices: list[jax.Device] = jax.devices(),
        *,
        key: PRNGKeyArray,
    ):
        assert evolve_start >= 0, "evolve_start must be greater than or equal to 0"
        assert (
            max_evolve_split >= 0
        ), "max_evolve_split must be greater than or equal to 0"
        assert len(devices) > 0, "No accelerator devices found"
        self.model = model
        self.optimizer = optimizer
        if opt_state is None:
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array_like))
        self.opt_state = opt_state
        self.key = key
        self.history = {"loss_reconstruction": [], "loss_time_stepping": []}
        self.curr_epoch = 0
        self.evolve_time = evolve_time
        self.callbacks = callbacks
        self.evolve_mode = evolve_mode
        self.loss = loss
        self.gamma = gamma
        self.evolve_fn = evolve_fn
        self.evolve_start = evolve_start
        self.max_split = max_evolve_split
        self.split_start = split_start
        self.random_split = random_split
        self.devices = devices

    @staticmethod
    def _loss_fn(
        params,
        static,
        field_1,
        field_2,
        spatial_coords,
        dt,
        dx,
        evolve_fn: Callable,
        evolve_time: bool,
        evolve_mode: str,
        loss: str,
        gamma: float,
        splits: int,
        solver_args,
    ):
        model = eqx.combine(params, static)
        shape = field_1.shape
        latent = eqx.filter_vmap(model.encoder)(
            jnp.reshape(field_1, (shape[0], shape[1], -1))
        )
        field_1_reconstructed = eqx.filter_vmap(
            eqx.filter_vmap(model.decoder.call_coords_latent, in_axes=(0, None))
        )(spatial_coords, latent)
        field_1_reconstructed = jnp.transpose(field_1_reconstructed, [0, 2, 1]).reshape(
            shape
        )
        norm_axis = (*range(2, field_1.ndim),)
        if loss == "mse":
            loss_reconstruction = jnp.mean(jnp.square(field_1_reconstructed - field_1))
        elif loss == "nmse":
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(field_1_reconstructed - field_1, axis=norm_axis)
                / jnp.linalg.norm(field_1, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        if not evolve_time:
            return loss_reconstruction, {
                "loss_reconstruction": loss_reconstruction,
                "loss_time_stepping": 0.0,
            }

        raise NotImplementedError("Not Implemented")

    def _inner_step(
        self,
        params,
        static,
        opt_state,
        field_1,
        field_2,
        spatial_coords,
        dt,
        dx,
        evolve_fn: Callable,
        evolve_time: bool,
        evolve_mode: str,
        loss: str,
        gamma: float,
        splits: int,
        solver_args,
    ):
        (loss_value, loss_dict), grads = jax.value_and_grad(
            self._loss_fn, has_aux=True
        )(
            params,
            static,
            field_1,
            field_2,
            spatial_coords,
            dt,
            dx,
            evolve_fn,
            evolve_time,
            evolve_mode,
            loss,
            gamma,
            splits,
            solver_args,
        )
        updates, opt_state = self.optimizer.update(grads, opt_state, params=params)
        params = optx.apply_updates(params, updates)
        return params, opt_state, loss_dict

    def fit(
        self,
        dataloader_train: NumpyLoader,
        epochs: int,
        warm_start: bool = True,
        **kwargs,
    ):
        if not warm_start:
            self.opt_state = self.optimizer.init(
                eqx.filter(self.model, eqx.is_array_like)
            )
            self.history = {"loss_reconstruction": [], "loss_time_stepping": []}
            self.curr_epoch = 0

        jitted_inner_step = jax.jit(
            self._inner_step,
            static_argnames=(
                "static",
                "evolve_time",
                "evolve_mode",
                "loss",
                "evolve_fn",
                "splits",
            ),
            donate_argnames=("params", "opt_state"),
        )
        params, static = eqx.partition(self.model, eqx.is_array_like)
        opt_state = self.opt_state
        curr_epoch = self.curr_epoch
        num_batches = len(dataloader_train)
        batch_splits = jnp.zeros((num_batches,), dtype=jnp.int32)
        for epoch in range(self.curr_epoch, epochs + curr_epoch):
            mean_reconstruction_loss = 0.0
            mean_time_stepping_loss = 0.0
            if epoch < self.evolve_start:
                evolve_time = False
            else:
                evolve_time = self.evolve_time
                if self.split_start <= epoch and self.max_split > 0:
                    if self.random_split:
                        self.key, subkey = jax.random.split(self.key)
                        batch_splits = jax.random.randint(
                            subkey, (num_batches,), 0, self.max_split + 1
                        )
                    else:
                        batch_splits = (
                            jnp.ones((num_batches,), dtype=jnp.int32) * self.max_split
                        )
            prefetcher = torch_iterator_prefetch(iter(dataloader_train), 2)
            for batch, splits in zip(prefetcher, batch_splits):
                (
                    field_1,
                    field_2,
                    temporal_coords,
                    spatial_coords,
                    dt,
                    dx,
                    traj,
                    *solver_args,
                ) = batch
                for i in range(len(solver_args)):
                    solver_args[i] = solver_args[i][0]
                params, opt_state, loss_dict = jitted_inner_step(
                    params,
                    static,
                    opt_state,
                    field_1[0],
                    field_2[0],
                    spatial_coords[0],
                    dt[0],
                    dx[0],
                    self.evolve_fn,
                    evolve_time,
                    self.evolve_mode,
                    self.loss,
                    self.gamma,
                    splits.item(),
                    solver_args,
                )
                mean_reconstruction_loss += loss_dict["loss_reconstruction"].item()
                mean_time_stepping_loss += loss_dict["loss_time_stepping"].item()

            mean_reconstruction_loss /= num_batches
            mean_time_stepping_loss /= num_batches
            self.history["loss_reconstruction"].append(mean_reconstruction_loss)
            self.history["loss_time_stepping"].append(mean_time_stepping_loss)
            print(
                f"Epoch {epoch + 1}/{epochs + curr_epoch}, Reconstruction Loss: {mean_reconstruction_loss},\
                  Time Stepping Loss: {mean_time_stepping_loss}"
            )
            self.model = eqx.combine(params, static)
            self.opt_state = opt_state
            self.curr_epoch += 1
            for callback in self.callbacks:
                self.model, self.opt_state, self.history, kwargs = callback(
                    self.model, self.opt_state, self.history, kwargs, epoch
                )

        return self.model, self.opt_state, self.history


class CROMAutoDecoderTrainer(CROMOfflineTrainer):

    def __init__(
        self,
        model: CROMOffline,
        optimizer: optx.GradientTransformation,
        optimizer_latent: optx.GradientTransformation,
        opt_state: optx.OptState = None,
        opt_state_latent: optx.OptState = None,
        loss: str = "nmse",
        evolve_time: bool = True,
        evolve_fn: Callable = None,
        evolve_mode: str = "label_label",
        evolve_start: int = 0,
        gamma: float = 0.5,
        max_evolve_split: int = 0,
        split_start: int = 0,
        random_split: bool = True,
        num_trajectories: int = None,
        num_time_steps: int = None,
        latent_dim: int = None,
        latent_memory: Array = None,
        callbacks: list[Callback] = [],
        devices: list[jax.Device] = jax.devices(),
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            model,
            optimizer,
            opt_state,
            loss,
            evolve_time,
            evolve_fn,
            evolve_mode,
            evolve_start,
            gamma,
            max_evolve_split,
            split_start,
            random_split,
            callbacks,
            devices,
            key=key,
        )

        if latent_memory is None:
            latent_memory = jnp.zeros((num_trajectories, num_time_steps, latent_dim))
        self.latent_memory = latent_memory
        self.optimizer_latent = optimizer_latent
        if opt_state_latent is None:
            opt_state_latent = optimizer_latent.init(latent_memory)
        self.opt_state_latent = opt_state_latent

    @staticmethod
    def _loss_fn(
        params,
        static,
        latent_memory,
        traj_indices,
        time_indices,
        field_1,
        field_2,
        spatial_coords,
        dt,
        dx,
        evolve_fn: Callable,
        evolve_time: bool,
        evolve_mode: str,
        loss: str,
        gamma: float,
        splits: int,
        solver_args,
    ):
        latent = latent_memory[traj_indices, time_indices]
        model = eqx.combine(params, static)
        field_1_reconstructed = eqx.filter_vmap(
            eqx.filter_vmap(model.decoder.call_coords_latent, in_axes=(0, None))
        )(spatial_coords, latent)
        field_1_reconstructed = jnp.transpose(field_1_reconstructed, [0, 2, 1]).reshape(
            field_1.shape
        )
        norm_axis = (*range(2, field_1.ndim),)
        if loss == "mse":
            loss_reconstruction = jnp.mean(jnp.square(field_1_reconstructed - field_1))
        elif loss == "nmse":
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(field_1_reconstructed - field_1, axis=norm_axis)
                / jnp.linalg.norm(field_1, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        if not evolve_time:
            return loss_reconstruction, {
                "loss_reconstruction": loss_reconstruction,
                "loss_time_stepping": 0.0,
            }

        raise NotImplementedError("Not implemented")

    def _inner_step(
        self,
        params,
        static,
        opt_state_model,
        opt_state_latent,
        latent_memory,
        field_1,
        field_2,
        spatial_coords,
        dt,
        dx,
        traj_indices: ArrayLike,
        time_indices: ArrayLike,
        evolve_fn: Callable,
        evolve_time: bool,
        evolve_mode: str,
        loss: str,
        gamma: float,
        splits: int,
        solver_args,
    ):

        f_loss_latent_grad = jax.value_and_grad(self._loss_fn, has_aux=True, argnums=2)
        f_loss_weights_grad = jax.value_and_grad(self._loss_fn, has_aux=True, argnums=0)

        (loss_value_latent, loss_dict_latent), latent_memory_grad = f_loss_latent_grad(
            params,
            static,
            latent_memory,
            traj_indices,
            time_indices,
            field_1,
            field_2,
            spatial_coords,
            dt,
            dx,
            evolve_fn,
            evolve_time,
            evolve_mode,
            loss,
            gamma,
            splits,
            solver_args,
        )

        updates_latent, opt_state_latent = self.optimizer_latent.update(
            latent_memory_grad, opt_state_latent, params=latent_memory
        )
        latent_memory = optx.apply_updates(latent_memory, updates_latent)

        (loss_value_model, loss_dict_model), model_grads = f_loss_weights_grad(
            params,
            static,
            latent_memory,
            traj_indices,
            time_indices,
            field_1,
            field_2,
            spatial_coords,
            dt,
            dx,
            evolve_fn,
            evolve_time,
            evolve_mode,
            loss,
            gamma,
            splits,
            solver_args,
        )
        updates_model, opt_state_model = self.optimizer.update(
            model_grads, opt_state_model, params=params
        )
        params = optx.apply_updates(params, updates_model)

        return params, latent_memory, opt_state_model, opt_state_latent, loss_dict_model

    def fit(
        self,
        dataloader_train: NumpyLoader,
        epochs: int,
        warm_start: bool = True,
        **kwargs,
    ):
        if not warm_start:
            self.history = {"loss_reconstruction": [], "loss_time_stepping": []}
            self.curr_epoch = 0

        jitted_inner_step = jax.jit(
            self._inner_step,
            static_argnames=(
                "static",
                "evolve_time",
                "evolve_mode",
                "loss",
                "evolve_fn",
                "splits",
            ),
            donate_argnames=("params", "opt_state_model", "opt_state_latent"),
        )

        mesh = jax.make_mesh((len(jax.devices()),), ("shard",), devices=self.devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(("shard",))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
        params, static = eqx.partition(self.model, eqx.is_array_like)
        opt_state_model = self.opt_state
        opt_state_latent = self.opt_state_latent
        latent_memory = self.latent_memory
        curr_epoch = self.curr_epoch
        num_batches = len(dataloader_train)
        batch_splits = jnp.zeros((num_batches,), dtype=jnp.int32)
        for epoch in range(self.curr_epoch, epochs + curr_epoch):
            mean_reconstruction = 0.0
            mean_time_stepping = 0.0
            if epoch < self.evolve_start:
                evolve_time = False
            else:
                evolve_time = self.evolve_time
                if self.split_start <= epoch and self.max_split > 0:
                    if self.random_split:
                        self.key, subkey = jax.random.split(self.key)
                        batch_splits = jax.random.randint(
                            subkey, (num_batches,), 0, self.max_split + 1
                        )
                    else:
                        batch_splits = (
                            jnp.ones((num_batches,), dtype=jnp.int32) * self.max_split
                        )
            prefetcher = iterator_sharded_prefetch(
                iter(dataloader_train), 2, sharding_data
            )
            for batch, splits in zip(prefetcher, batch_splits):

                field_1 = batch.get("data", None)
                time_indices = batch.get("time_idx", None)
                spatial_coords = batch.get("coords", None)
                dt = batch.get("dt", None)
                dx = batch.get("dx", None)
                traj_indices = batch.get("idx", None)
                solver_args = batch.get("solver_args", None)
                params, latent_memory, opt_state_model, opt_state_latent, loss_dict = (
                    jitted_inner_step(
                        params,
                        static,
                        opt_state_model,
                        opt_state_latent,
                        latent_memory,
                        field_1,
                        None,
                        spatial_coords,
                        dt,
                        dx,
                        traj_indices,
                        time_indices,
                        self.evolve_fn,
                        evolve_time,
                        self.evolve_mode,
                        self.loss,
                        self.gamma,
                        splits.item(),
                        solver_args,
                    )
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
            self.model = eqx.combine(params, static)
            self.opt_state = opt_state_model
            self.opt_state_latent = opt_state_latent
            self.curr_epoch += 1
            self.latent_memory = latent_memory
            for callback in self.callbacks:
                (
                    self.model,
                    _,
                    self.latent_memorylatent_memory,
                    self.opt_state,
                    self.history,
                    kwargs,
                ) = callback(
                    self.model,
                    None,
                    latent_memory,
                    self.opt_state,
                    self.history,
                    kwargs,
                    epoch,
                )

        return self.model, self.opt_state, self.history


class DINOTrainer(CROMAutoDecoderTrainer):
    """
    DINo trainer with data on regular grid.
    """

    def __init__(
        self,
        model: NodeROM,
        model_state: PyTree,
        optimizer: optx.GradientTransformation,
        optimizer_node: optx.GradientTransformation,
        optimizer_latent: optx.GradientTransformation,
        opt_state: optx.OptState = None,
        opt_state_node: optx.OptState = None,
        opt_state_latent: optx.OptState = None,
        loss: str = "nmse",
        evolve_fn: Callable = None,
        evolve_start: int = 0,
        max_evolve_split: int = 0,
        split_start: int = 0,
        random_split: bool = True,
        num_trajectories: int = None,
        num_time_steps: int = None,
        latent_dim: int = None,
        latent_memory: Array = None,
        callbacks: Sequence[Callback] = [],
        gamma: float = 0.99,
        gamma_decay_rate: float = 0.99,
        gamma_decay_epochs: int = 20,
        final_gamma: float = 0.0,
        devices: list[jax.Device] = jax.devices(),
        *,
        key: PRNGKeyArray,
    ):
        if opt_state is None:
            opt_state = optimizer.init(eqx.filter(model.decoder, eqx.is_array_like))
        super().__init__(
            model,
            optimizer,
            optimizer_latent,
            opt_state,
            opt_state_latent,
            loss,
            False,
            evolve_fn,
            "pred_pred",
            evolve_start,
            0.5,
            max_evolve_split,
            split_start,
            random_split,
            num_trajectories,
            num_time_steps,
            latent_dim,
            latent_memory,
            callbacks,
            devices,
            key=key,
        )
        self.optimizer_node = optimizer_node
        if opt_state_node is None:
            opt_state_node = optimizer_node.init(
                eqx.filter(model.node, eqx.is_array_like)
            )
        self.opt_state_node = opt_state_node
        self.gamma = gamma
        self.gamma_decay_rate = gamma_decay_rate
        self.gamma_decay_epochs = gamma_decay_epochs
        self.final_gamma = final_gamma
        self.model_state = model_state

    @staticmethod
    def _loss_fn(
        decoder,
        model_state,
        latent_memory,
        traj_indices,
        trajectories,
        spatial_coords,
        loss,
    ):
        decoder = eqx.filter_vmap(
            eqx.filter_vmap(
                eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None)),
                in_axes=(None, 0),
            )
        )
        latents = latent_memory[traj_indices, :]

        trajs_reconstructed = decoder(spatial_coords, latents)
        trajs_reconstructed = jnp.transpose(trajs_reconstructed, [0, 1, 3, 2]).reshape(
            trajectories.shape
        )
        norm_axis = (*range(3, trajectories.ndim),)

        if loss == "mse":
            loss_reconstruction = jnp.mean(
                jnp.square(trajs_reconstructed - trajectories)
            )
        elif loss == "nmse":
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(trajs_reconstructed - trajectories, axis=norm_axis)
                / jnp.linalg.norm(trajectories, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        return loss_reconstruction, model_state

    @staticmethod
    def _loss_node(
        node,
        model_state,
        latent_memory,
        traj_indices: Array,
        temporal_coords: Array,
        dt: float,
        dx: float,
        node_args: list,
        loss: str,
        gamma: float,
        key: PRNGKeyArray,
    ):

        def mlp_fn(t, x, args):
            arg, state = args
            out, _ = node.mlp(t, x, arg, state)
            return out

        term = diff.ODETerm(mlp_fn)
        dummy_ode_state = eqx.filter_vmap(node.solver.init, in_axes=(None, 0, 0, 0, 0))(
            term,
            temporal_coords[:, 0],
            temporal_coords[:, 1],
            latent_memory[traj_indices, 0],
            (node_args, model_state),
        )

        segment_mask = jax.random.choice(
            key,
            a=jnp.array([True, False]),
            shape=(temporal_coords.shape[1],),
            p=jnp.array([gamma, 1.0 - gamma]),
        )
        segment_mask = segment_mask.at[0].set(True)
        segment_mask = segment_mask[:-1]
        latents = jnp.zeros_like(latent_memory[traj_indices, :])
        latents = latents.at[:, 0].set(latent_memory[traj_indices, 0])

        def _update_latents(i, args):
            (
                latents,
                latent_memory,
                node,
                model_state,
                node_args,
                temporal_coords,
                traj_indices,
                segment_mask,
                ode_state,
            ) = args
            latent_memory_selected = latent_memory[traj_indices, :]
            t0 = jax.lax.dynamic_index_in_dim(
                temporal_coords, i, axis=1, keepdims=False
            )
            t1 = jax.lax.dynamic_index_in_dim(
                temporal_coords, i + 1, axis=1, keepdims=False
            )
            do_segment = jax.lax.dynamic_index_in_dim(
                segment_mask, i, axis=0, keepdims=False
            )
            latents, ode_state, model_state = jax.lax.cond(
                do_segment,
                _true_segment_fn,
                _false_continue_fn,
                i,
                latents,
                latent_memory_selected,
                node,
                model_state,
                node_args,
                t0,
                t1,
                ode_state,
            )
            return (
                latents,
                latent_memory,
                node,
                model_state,
                node_args,
                temporal_coords,
                traj_indices,
                segment_mask,
                ode_state,
            )

        def _true_segment_fn(
            i, latents, latent_memory, node, model_state, node_args, t0, t1, ode_state
        ):
            if node_args is None:
                node = eqx.filter_vmap(
                    node.call_step,
                    in_axes=(0, 0, 0, None, None, None, None, None, None),
                )
            else:
                node = eqx.filter_vmap(
                    node.call_step, in_axes=(0, 0, 0, None, 0, None, None, None, None)
                )
            latent_ic = jax.lax.dynamic_index_in_dim(
                latent_memory, i, axis=1, keepdims=False
            )
            latent_next, ode_state, model_state = node(
                latent_ic, t0, t1, None, node_args, None, None, None, model_state
            )
            latent_next = latent_next.squeeze()
            latents = jax.lax.dynamic_update_index_in_dim(
                latents, latent_next, i + 1, axis=1
            )
            return latents, ode_state, model_state

        def _false_continue_fn(
            i, latents, latent_memory, node, model_state, node_args, t0, t1, ode_state
        ):
            if node_args is None:
                node = eqx.filter_vmap(
                    node.call_step,
                    in_axes=(0, 0, 0, None, None, None, None, None, None),
                )
            else:
                node = eqx.filter_vmap(
                    node.call_step, in_axes=(0, 0, 0, None, 0, None, None, None, None)
                )
            latent_ic = jax.lax.dynamic_index_in_dim(latents, i, axis=1, keepdims=False)
            latent_next, ode_state, model_state = node(
                latent_ic, t0, t1, None, node_args, None, None, None, model_state
            )
            latent_next = (
                latent_next.squeeze()
            )  # FIXME: Squeeze won't work with a batch size of 1
            latents = jax.lax.dynamic_update_index_in_dim(
                latents, latent_next, i + 1, axis=1
            )
            return latents, ode_state, model_state

        (
            latents,
            latent_memory,
            node,
            model_state,
            node_args,
            temporal_coords,
            traj_indices,
            segment_mask,
            dummy_ode_state,
        ) = jax.lax.fori_loop(
            0,
            temporal_coords.shape[1] - 1,
            _update_latents,
            (
                latents,
                latent_memory,
                node,
                model_state,
                node_args,
                temporal_coords,
                traj_indices,
                segment_mask,
                dummy_ode_state,
            ),
        )

        loss_node = jnp.mean(jnp.square(latents - latent_memory[traj_indices]))
        return loss_node, model_state

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
        node_args,
        gamma,
        sharding_model,
        sharding_data,
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
        trajectories, temporal_coords, spatial_coords, dt, dx, traj_indices = (
            eqx.filter_shard(
                (trajectories, temporal_coords, spatial_coords, dt, dx, traj_indices),
                sharding_data,
            )
        )

        decoder = model.decoder
        node = model.node

        def dummy_recons_loss_latent(
            latent_memory,
            decoder,
            model_state,
            traj_indices,
            trajectories,
            spatial_coords,
            loss,
        ):
            return self._loss_fn(
                decoder,
                model_state,
                latent_memory,
                traj_indices,
                trajectories,
                spatial_coords,
                loss,
            )

        f_prime_loss_latent = eqx.filter_value_and_grad(
            dummy_recons_loss_latent, has_aux=True
        )
        f_prime_loss_decoder = eqx.filter_value_and_grad(self._loss_fn, has_aux=True)
        f_prime_loss_node = eqx.filter_value_and_grad(self._loss_node, has_aux=True)

        (loss_value_latent, model_state), loss_grad_latent = f_prime_loss_latent(
            latent_memory,
            decoder,
            model_state,
            traj_indices,
            trajectories,
            spatial_coords,
            self.loss,
        )
        updates_latent, opt_state_latent = self.optimizer_latent.update(
            loss_grad_latent, opt_state_latent, params=latent_memory
        )
        latent_memory = eqx.apply_updates(latent_memory, updates_latent)

        (loss_value_decoder, model_state), loss_grad_decoder = f_prime_loss_decoder(
            decoder,
            model_state,
            latent_memory,
            traj_indices,
            trajectories,
            spatial_coords,
            self.loss,
        )
        updates_decoder, opt_state_decoder = self.optimizer.update(
            loss_grad_decoder, opt_state_decoder, params=decoder
        )
        decoder = eqx.apply_updates(decoder, updates_decoder)

        (loss_value_node, model_state), loss_grad_node = f_prime_loss_node(
            node,
            model_state,
            latent_memory,
            traj_indices,
            temporal_coords,
            dt,
            dx,
            node_args,
            self.loss,
            gamma,
            key,
        )
        updates_node, opt_state_node = self.optimizer_node.update(
            loss_grad_node, opt_state_node, params=node
        )
        node = eqx.apply_updates(node, updates_node)

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
            {
                "loss_reconstruction": loss_value_decoder,
                "loss_time_stepping": loss_value_node,
            },
        )

    def fit(
        self,
        dataloader_train: DataLoader,
        epochs: int,
        warm_start: bool = False,
        **kwargs,
    ):
        if not warm_start:
            self.history = {
                "loss_reconstruction": [],
                "loss_time_stepping": [],
            }
            self.curr_epoch = 0
        print("Training Data-Driven DINO")
        jitted_inner_step = eqx.filter_jit(self._inner_step, donate="all")
        model = self.model
        opt_state_decoder = self.opt_state
        opt_state_latent = self.opt_state_latent
        opt_state_node = self.opt_state_node
        latent_memory = self.latent_memory
        curr_epoch = self.curr_epoch
        model_state = self.model_state

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
            mean_reconstruction = 0.0
            mean_time_stepping = 0.0
            prefetcher = iterator_sharded_prefetch(
                iter(dataloader_train), 2, sharding_data
            )
            for batch in prefetcher:
                trajectories = batch.get("data", None)
                temporal_coords = batch.get("t", None)
                spatial_coords = batch.get("coords", None)
                dt = batch.get("dt", None)
                dx = batch.get("dx", None)
                indices = batch.get("idx", None)
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
                    indices,
                    node_args,
                    jnp.array(self.gamma),
                    sharding_model,
                    sharding_data,
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
            if (epoch + 1) % self.gamma_decay_epochs == 0:
                self.gamma *= self.gamma_decay_rate
                print(f"Decaying gamma to {self.gamma}")
                if self.gamma < self.final_gamma + 1e-3:
                    self.gamma = self.final_gamma
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


class IrregularDINoTrainer(DINOTrainer):
    """
    DINo trainer with irregular data.
    """

    @staticmethod
    def _loss_fn(
        decoder,
        model_state,
        latent_memory,
        traj_indices,
        trajectories,
        spatial_coords,
        loss,
    ):
        decoder = eqx.filter_vmap(
            eqx.filter_vmap(
                eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None)),
                in_axes=(None, 0),
            )
        )
        latents = latent_memory[traj_indices, :]

        trajs_reconstructed = decoder(spatial_coords, latents)
        trajs_reconstructed = jnp.transpose(trajs_reconstructed, [0, 1, 3, 2])
        norm_axis = -1

        if loss == "mse":
            loss_reconstruction = jnp.mean(
                jnp.square(trajs_reconstructed - trajectories)
            )
        elif loss == "nmse":
            loss_reconstruction = jnp.mean(
                jnp.linalg.norm(trajs_reconstructed - trajectories, axis=norm_axis)
                / jnp.linalg.norm(trajectories, axis=norm_axis)
            )
        else:
            raise ValueError("Invalid loss function. Should be one of ['nmse', 'mse']")

        return loss_reconstruction, model_state

    def fit(
        self,
        dataloader_train: DataLoader,
        epochs: int,
        warm_start: bool = False,
        **kwargs,
    ):
        if not warm_start:
            self.history = {
                "loss_reconstruction": [],
                "loss_time_stepping": [],
            }
            self.curr_epoch = 0
        print("Training Irregular Data-Driven DINO")
        jitted_inner_step = eqx.filter_jit(self._inner_step, donate="all")
        model = self.model
        opt_state_decoder = self.opt_state
        opt_state_latent = self.opt_state_latent
        opt_state_node = self.opt_state_node
        latent_memory = self.latent_memory
        curr_epoch = self.curr_epoch
        model_state = self.model_state

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
            mean_reconstruction = 0.0
            mean_time_stepping = 0.0
            prefetcher = iterator_sharded_prefetch(
                iter(dataloader_train), 2, sharding_data
            )
            for batch in prefetcher:
                trajectories = batch["data_irregular"]
                temporal_coords = batch.get("t", None)
                spatial_coords = batch["coords_irregular"]
                dt = batch.get("dt", None)
                dx = batch.get("dx", None)
                indices = batch.get("idx", None)
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
                    indices,
                    node_args,
                    jnp.array(self.gamma),
                    sharding_model,
                    sharding_data,
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
            if (epoch + 1) % self.gamma_decay_epochs == 0:
                self.gamma *= self.gamma_decay_rate
                print(f"Decaying gamma to {self.gamma}")
                if self.gamma < self.final_gamma + 1e-3:
                    self.gamma = self.final_gamma
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
