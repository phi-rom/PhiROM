"""
Callback functions for training Equinox models. Each callback function should have the following signature:
    Args:
    - model: eqx.Module, the model to be trained
    - model_state: PyTree, the state of the model
    - latent_memory: Array, the latent memory of the model
    - opt_state: optax.OptState, the optimizer state
    - history: Dict[str, ArrayLike], the history of the training process
    - training_config: Dict[str, ArrayLike], the configuration of the training process
    - epoch: int, the current epoch number
    Returns:
    - Tuple containing the updated model, optimizer state, history, and training configuration
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader, Dataset

from ..modules.inference import find_latent_descent
from ..pde.data_utils import iterator_sharded_prefetch, torch_iterator_prefetch
from ..utils.serial import save_model, save_opt_state
from .evaluation import unroll_descent, unroll_descent_XLB


class Callback:
    """
    Base class for all callbacks. Child classes should implement the `call` method.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

    def call(
        self,
        model: eqx.Module,
        model_state: PyTree,
        latent_memory: Array,
        opt_state: optax.OptState,
        history: Dict[str, ArrayLike],
        training_config: Dict[str, ArrayLike],
        epoch: int,
    ) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        The main method to be implemented in the child class.

        Args:
        - model: eqx.Module, the model to be trained
        - opt_state: optax.OptState, the optimizer state
        - history: Dict[str, ArrayLike], the history of the training process
        - training_config: Dict[str, ArrayLike], the configuration of the training process
        - epoch: int, the current epoch number

        Returns:
        - Tuple containing the updated model, optimizer state, history, and training configuration
        """
        raise NotImplementedError(
            "The `call` method should be implemented in the child class."
        )


class CheckpointCallback(Callback):
    """
    A callback class for saving the model and optimizer state at specified intervals.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        hyper_params: Dict,
        add_time: bool = False,
        save_every: int = 50,
    ):
        """
        Initializes the CheckpointCallback instance.

        Args:
        - checkpoint_dir: str, the directory to save the checkpoint.
        - checkpoint_name: str, the name of the checkpoint file.
        - hyper_params: Dict, hyperparameters to be saved along with the model.
        - save_every: int, the frequency of saving the checkpoint. Default is 10 (save every 10 epochs).
        """
        self.checkpoint_name = checkpoint_name
        self.hyper_params = hyper_params
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        if add_time:
            now = datetime.now()
            self.checkpoint_dir = os.path.join(
                checkpoint_dir, now.strftime("%Y-%m-%d_%H-%M-%S")
            )
        # self.checkpoint_dir = os.path.join(checkpoint_dir, self.checkpoint_name)

        # if add_time:
        #     now = datetime.now()
        #     self.checkpoint_name = f"{checkpoint_name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        # self.checkpoint_dir = os.path.join(checkpoint_dir, self.checkpoint_name)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def call(
        self,
        model: eqx.Module,
        model_state: PyTree,
        latent_memory,
        opt_state: optax.OptState,
        history: Dict[str, ArrayLike],
        training_config: Dict[str, ArrayLike],
        epoch: int,
    ) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        Saves the model and optimizer state at the specified interval and returns the inputs unchanged.

        Args:
        - model: eqx.Module, the model being trained.
        - opt_state: optax.OptState, the current optimizer state.
        - history: Dict[str, ArrayLike], a dictionary to store training history.
        - training_config: Dict[str, ArrayLike], a dictionary containing training configurations.
        - epoch: int, the current epoch count.

        Returns:
        - Tuple containing the model, optimizer state, history, and training configuration unchanged.
        """
        if epoch % self.save_every == 0:
            path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.eqx")
            path_opt = os.path.join(self.checkpoint_dir, f"opt_state_epoch_{epoch}.opx")
            path_latent = os.path.join(
                self.checkpoint_dir, f"latent_memory_epoch_{epoch}.npy"
            )
            save_model(path, self.hyper_params, model, model_state)
            save_opt_state(path_opt, opt_state)
            jnp.save(path_latent, latent_memory)
            print(f"Checkpoint saved at {path}")
        return model, model_state, latent_memory, opt_state, history, training_config


class NODEUnrollingEvaluationCallback(Callback):
    """
    A callback class for evaluating the model unrolling performance at specified intervals.

    """

    def __init__(
        self,
        dataset,
        T_train: int,
        T_extrapolate: int = None,
        eval_every: int = 50,
        metric: str = "nmse",
        print_results: bool = True,
        dict_key_prefix: str = "unrolling_error",
        plot_results: bool = False,
        plot_dir: str = None,
        devices=jax.devices(),
        batch_size: int = len(jax.devices()),
    ):
        super().__init__()
        if metric not in ["mse", "nmse"]:
            raise ValueError("The metric should be either 'mse' or 'mae'.")
        if plot_results and plot_dir is None:
            raise ValueError("The plot directory should be specified.")
        self.metric = metric
        self.eval_every = eval_every
        self.print_results = print_results
        self.dataset = dataset
        self.key = dict_key_prefix
        self.T_train = T_train
        if T_extrapolate is None:
            T_extrapolate = T_train
        self.T_extrapolate = T_extrapolate
        self.prev_inter = 0
        self.prev_extra = 0
        self.plot_results = plot_results
        self.plot_dir = plot_dir
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        num_devices = len(devices)
        mesh = jax.make_mesh((num_devices,), ("shard",), devices=devices)
        pspec_data = jax.sharding.PartitionSpec(("shard",))
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
        self.sharding_data = sharding_data

    def call(
        self,
        model: eqx.Module,
        model_state: PyTree,
        latent_memory: Array,
        opt_state: optax.OptState,
        history: Dict[str, ArrayLike],
        training_config: Dict[str, ArrayLike],
        epoch: int,
    ) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        if epoch % self.eval_every != 0:
            if self.key + "_interpolate" not in history:
                history[self.key + "_interpolate"] = []
                history[self.key + "_extrapolate"] = []
            history[self.key + "_interpolate"].append(self.prev_inter)
            history[self.key + "_extrapolate"].append(self.prev_extra)
            return (
                model,
                model_state,
                latent_memory,
                opt_state,
                history,
                training_config,
            )
        # _, errors = unroll_descent(model, self.dataset, t1=self.T_extrapolate, t0=0, n_steps=500)
        iterator = iterator_sharded_prefetch(
            iter(self.dataloader), 1, self.sharding_data
        )
        errors_interpolate = 0
        errors_extrapolate = 0
        for batch in iterator:
            _, errors_batch = unroll_descent(
                model,
                model_state,
                batch,
                t1=self.T_extrapolate,
                t0=0,
                n_steps=1000,
                loss=self.metric,
                optimizer=optax.adam(1e-1),
            )
            errors_interpolate_batch = errors_batch[:, : self.T_train].mean()
            if self.T_train == self.T_extrapolate:
                errors_extrapolate_batch = 0
            else:
                errors_extrapolate_batch = errors_batch[:, self.T_train :].mean()
                errors_extrapolate += errors_extrapolate_batch
            errors_interpolate += errors_interpolate_batch

        error_interpolate = errors_interpolate / len(self.dataloader)
        error_extrapolate = errors_extrapolate / len(self.dataloader)

        if self.print_results:
            print(
                f"Epoch {epoch} - {self.key}: Interpolation error: {error_interpolate}, Extrapolation error: {error_extrapolate}"
            )
        if self.key + "_interpolate" in history:
            history[self.key + "_interpolate"].append(error_interpolate)
            history[self.key + "_extrapolate"].append(error_extrapolate)
        else:
            history[self.key + "_interpolate"] = [error_interpolate]
            history[self.key + "_extrapolate"] = [error_extrapolate]
        if self.plot_results:
            plt.figure(figsize=(7, 5), dpi=150)
            for key in history:
                if "loss" in key:
                    plt.plot(history[key], label=key)
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.savefig(os.path.join(self.plot_dir, f"loss_plot_{self.key}.png"))
            plt.close()
            plt.figure(figsize=(7, 5), dpi=150)
            for key in history:
                if "loss" not in key:
                    plt.plot(history[key], label=key)
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            # set ylim
            # plt.ylim(0, 1)
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.savefig(os.path.join(self.plot_dir, f"error_plot_{self.key}.png"))
            plt.close()
            # save history
            jnp.savez(os.path.join(self.plot_dir, f"history_{self.key}.npz"), **history)
        self.prev_inter = error_interpolate
        self.prev_extra = error_extrapolate
        return model, model_state, latent_memory, opt_state, history, training_config


class NODEUnrollingEvaluationCallbackXLB(Callback):
    """
    A callback class for evaluating the model unrolling performance at specified intervals (modified for LBM).

    """

    def __init__(
        self,
        dataset,
        xlb_macro,
        T_train: int,
        T_extrapolate: int = None,
        eval_every: int = 50,
        metric: str = "nmse",
        print_results: bool = True,
        dict_key_prefix: str = "unrolling_error",
        plot_results: bool = False,
        plot_dir: str = None,
        devices=jax.devices(),
        batch_size: int = len(jax.devices()),
    ):
        super().__init__()
        if metric not in ["mse", "nmse"]:
            raise ValueError("The metric should be either 'mse' or 'mae'.")
        if plot_results and plot_dir is None:
            raise ValueError("The plot directory should be specified.")
        self.metric = metric
        self.eval_every = eval_every
        self.print_results = print_results
        self.dataset = dataset
        self.key = dict_key_prefix
        self.T_train = T_train
        if T_extrapolate is None:
            T_extrapolate = T_train
        self.T_extrapolate = T_extrapolate
        self.prev_inter = 0
        self.prev_extra = 0
        self.plot_results = plot_results
        self.plot_dir = plot_dir
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        num_devices = len(devices)
        mesh = jax.make_mesh((num_devices,), ("shard",), devices=devices)
        pspec_data = jax.sharding.PartitionSpec(("shard",))
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
        self.sharding_data = sharding_data
        self.xlb_macro = xlb_macro

    def call(
        self,
        model: eqx.Module,
        model_state: PyTree,
        latent_memory: Array,
        opt_state: optax.OptState,
        history: Dict[str, ArrayLike],
        training_config: Dict[str, ArrayLike],
        epoch: int,
    ) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        if epoch % self.eval_every != 0:
            if self.key + "_interpolate" not in history:
                history[self.key + "_interpolate"] = []
                history[self.key + "_extrapolate"] = []
            history[self.key + "_interpolate"].append(self.prev_inter)
            history[self.key + "_extrapolate"].append(self.prev_extra)
            return (
                model,
                model_state,
                latent_memory,
                opt_state,
                history,
                training_config,
            )
        # _, errors = unroll_descent(model, self.dataset, t1=self.T_extrapolate, t0=0, n_steps=500)
        iterator = iterator_sharded_prefetch(
            iter(self.dataloader), 1, self.sharding_data
        )
        errors_interpolate = 0
        errors_extrapolate = 0
        for batch in iterator:
            _, errors_batch = unroll_descent_XLB(
                model,
                model_state,
                self.xlb_macro,
                batch,
                t1=self.T_extrapolate,
                t0=0,
                n_steps=500,
                loss=self.metric,
            )
            errors_interpolate_batch = errors_batch[:, : self.T_train].mean()
            if self.T_train == self.T_extrapolate:
                errors_extrapolate_batch = 0
            else:
                errors_extrapolate_batch = errors_batch[:, self.T_train :].mean()
                errors_extrapolate += errors_extrapolate_batch
            errors_interpolate += errors_interpolate_batch

        error_interpolate = errors_interpolate / len(self.dataloader)
        error_extrapolate = errors_extrapolate / len(self.dataloader)

        if self.print_results:
            print(
                f"Epoch {epoch} - {self.key}: Interpolation error: {error_interpolate}, Extrapolation error: {error_extrapolate}"
            )
        if self.key + "_interpolate" in history:
            history[self.key + "_interpolate"].append(error_interpolate)
            history[self.key + "_extrapolate"].append(error_extrapolate)
        else:
            history[self.key + "_interpolate"] = [error_interpolate]
            history[self.key + "_extrapolate"] = [error_extrapolate]
        if self.plot_results:
            plt.figure(figsize=(7, 5), dpi=150)
            for key in history:
                if "loss" in key:
                    plt.plot(history[key], label=key)
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.savefig(os.path.join(self.plot_dir, f"loss_plot_{self.key}.png"))
            plt.close()
            plt.figure(figsize=(7, 5), dpi=150)
            for key in history:
                if "loss" not in key:
                    plt.plot(history[key], label=key)
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.savefig(os.path.join(self.plot_dir, f"error_plot_{self.key}.png"))
            plt.close()
            # save history
            jnp.savez(os.path.join(self.plot_dir, f"history_{self.key}.npz"), **history)
        self.prev_inter = error_interpolate
        self.prev_extra = error_extrapolate
        return model, model_state, latent_memory, opt_state, history, training_config
