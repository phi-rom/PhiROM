"""
This file contains the base modules and building blocks for Phi-ROM and the baselines.
"""

import abc
from typing import Any, Callable, List, Sequence, Tuple, Union

import diffrax as diff
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree


def get_activation(activation: str) -> Callable:
    activation = activation.lower()
    if activation == "elu":
        return jnn.elu
    elif activation == "sin":
        return jnp.sin
    elif activation == "softplus":
        return jnn.softplus
    elif activation == "tanh":
        return jnn.tanh
    elif activation == "swish":
        return ParameterizedSwish()
    elif activation == "mish":
        return jnn.mish
    elif activation == "relu":
        return jnn.relu
    elif activation == "elu":
        return jnn.elu
    elif activation == "sigmoid":
        return jnn.sigmoid
    elif activation == "sigmoid_half":
        return sigmoid_half
    else:
        raise ValueError("Activation function not supported")


def get_solver(solver: str) -> diff.AbstractSolver:
    solver = solver.lower()
    if solver == "bosh3":
        return diff.Bosh3()
    elif solver == "dopri5":
        return diff.Dopri5()
    elif solver == "dopri8":
        return diff.Dopri8()
    elif solver == "tsit5":
        return diff.Tsit5()
    elif solver == "euler":
        return diff.Euler()
    else:
        raise ValueError("Solver not supported")


def sigmoid_half(x: Array) -> Array:
    return 0.5 * jnn.sigmoid(x)


def get_initializer(initializer: str):
    initializer = initializer.lower()
    if initializer == "he_normal":
        return jax.nn.initializers.he_normal()
    elif initializer == "he_uniform":
        return jax.nn.initializers.he_uniform()
    elif initializer == "glorot_normal":
        return jax.nn.initializers.glorot_normal()
    elif initializer == "glorot_uniform":
        return jax.nn.initializers.glorot_uniform()
    elif initializer == "lecun_normal":
        return jax.nn.initializers.lecun_normal()
    elif initializer == "lecun_uniform":
        return jax.nn.initializers.lecun_uniform()
    else:
        raise ValueError("Initializer not supported")


class ParameterizedSwish(eqx.Module):
    """
    Parameterized Swish activation function as in https://arxiv.org/abs/2209.14855
    """

    beta: Array

    def __init__(self):
        self.beta = jnp.array(0.5)

    def __call__(self, x: Array) -> Array:
        return (x * jnn.sigmoid(x * jnn.softplus(self.beta))) / 1.1


class Normalization(eqx.Module):
    """
    Normalization layer for input spatial coordinates of INRs.
    """

    mean: ArrayLike
    std: ArrayLike

    def __init__(self, mean: Array, std: Array):
        if np.ndim(mean) == 0:
            mean = np.array([mean])
        if np.ndim(std) == 0:
            std = np.array([std])
        self.mean = mean
        self.std = std
        print(f"Normalization with mean shape {mean.shape}")

    def __call__(self, x: Array, *, key=None) -> Array:
        return (x - jax.lax.stop_gradient(self.mean)) / (
            jax.lax.stop_gradient(self.std) + 1e-8
        )


class Denormalization(eqx.Module):
    """
    Denormalization layer for the output vector fields of INRs. The outputs are multiplied by the standard deviation and added to the mean.
    """

    mean: ArrayLike
    std: ArrayLike

    def __init__(self, mean: Array, std: Array):
        if np.ndim(mean) == 0:
            mean = np.array([mean])
        if np.ndim(std) == 0:
            std = np.array([std])
        self.mean = mean
        self.std = std
        print(f"Denormalization with mean shape {mean.shape}")

    def __call__(self, x: Array) -> Array:
        return x * jax.lax.stop_gradient(self.std) + jax.lax.stop_gradient(self.mean)


class Standardization(eqx.Module):
    """
    Standardization layer for the input spatial coordinates of INRs.
    """

    min: ArrayLike
    max: ArrayLike

    def __init__(self, min: Array, max: Array):
        if np.ndim(min) == 0:
            min = np.array([min])
        if np.ndim(max) == 0:
            max = np.array([max])
        self.min = min
        self.max = max
        print(min, max)

    def __call__(self, x: Array) -> Array:
        return (
            2
            * (x - jax.lax.stop_gradient(self.min))
            / (jax.lax.stop_gradient(self.max) - jax.lax.stop_gradient(self.min))
            - 1
        )


class FourierLayer(eqx.Module):
    """
    Fourier layer for Hyper INR adopted from DINo (https://arxiv.org/abs/2209.14855).
    """

    trainable: bool = eqx.field(static=True)
    width: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    weight_scale: float
    weights: Array

    def __init__(
        self,
        in_dim: int,
        width: int,
        weight_scale: float,
        trainable: bool,
        *,
        key: PRNGKeyArray,
    ):
        self.width = width
        self.trainable = trainable
        self.in_dim = in_dim
        self.weight_scale = weight_scale
        initializer = lambda *args, **kwargs: jax.nn.initializers.he_uniform()(
            *args, **kwargs
        )
        self.weights = initializer(key, (self.in_dim, self.width))

    def __call__(self, x: Array) -> Array:
        if self.trainable:
            x = jnp.dot(x, self.weights * jax.lax.stop_gradient(self.weight_scale))
        else:
            x = jnp.dot(
                x,
                jax.lax.stop_gradient(self.weights)
                * jax.lax.stop_gradient(self.weight_scale),
            )

        return jnp.concat([jnp.sin(x), jnp.cos(x)], axis=-1)


class EncoderMLP(eqx.Module):
    """
    Encoder module (only for auto-encoders) adopted from CROM (https://arxiv.org/pdf/2206.02607).

    Args:
        field_dim: int, input field dimension
        num_sensors: int, number of sensors
        latent_dim: int, latent dimension
        activation: Callable, activation function
        key: PRNGKeyArray, random key
    """

    field_dim: int = eqx.field(static=True)
    num_sensors: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    conv_layers: eqx.nn.Sequential
    linear_layers: eqx.nn.Sequential
    normalization_layer: Normalization

    def __init__(
        self,
        field_dim: int,
        num_sensors: int,
        latent_dim: int,
        activation: Union[Callable, str],
        mean_field: Array = None,
        std_field: Array = None,
        *,
        key: PRNGKeyArray,
        **_,
    ):
        if mean_field is not None and std_field is not None:
            self.normalization_layer = Normalization(mean_field, std_field)
        else:
            self.normalization_layer = None
        if not isinstance(activation, Callable):
            activation = get_activation(activation)
        self.field_dim = field_dim
        self.num_sensors = num_sensors
        self.latent_dim = latent_dim
        conv_layers = []
        while (
            int((num_sensors - 6) / 4) + 1 >= 32 / field_dim
        ):  # output of conv layers should be at closest to 32 but not less (based on original CROM implementation)
            key, subkey = jax.random.split(key)
            conv_layers.append(
                eqx.nn.Conv1d(
                    in_channels=field_dim,
                    out_channels=field_dim,
                    kernel_size=6,
                    stride=4,
                    key=subkey,
                )
            )
            num_sensors = int((num_sensors - 6) / 4) + 1
        self.conv_layers = eqx.nn.Sequential(conv_layers)
        key, subkey1, subkey2 = jax.random.split(key, 3)
        layer_1 = eqx.nn.Linear(num_sensors * field_dim, 32, key=subkey1)
        layer_2 = eqx.nn.Linear(32, latent_dim, key=subkey2)
        self.linear_layers = eqx.nn.Sequential(
            [layer_1, eqx.nn.Lambda(activation), layer_2, eqx.nn.Lambda(activation)]
        )

    def __call__(self, x: Array) -> Array:
        """
        Forward pass.

        Args:
            x: Array, input tensor in channel-first format (field_dim, num_sensors)
        """
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        x = self.conv_layers(x)
        x = jax.numpy.ravel(x)
        x = self.linear_layers(x)
        return x


class BaseDecoder(eqx.Module):
    """
    Base decoder class for INRs. This is an abstract class.
    grads_x and second_grads_x are the gradients of the output with respect to the input coordinates, needed for PINN-ROM.
    """

    spatial_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(self, spatial_dim: int, latent_dim: int, out_dim: int, **_):
        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

    @abc.abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def call_coords_latent(self, coords: Array, latent: Array) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def grads_x(self, coord: Array, latent: Array) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def second_grads_x(self, coord: Array, latent: Array) -> Array:
        raise NotImplementedError


class DecoderMLP(BaseDecoder):
    """
    Simple MLP decoder. Last layer has no activation function.

    Args:
        in_dim: int, input dimension
        out_dim: int, output field dimension
        n_layers: int, number of layers
        width: int, width of the network
        activation: Callable, activation function
        key: PRNGKeyArray, random key
    """

    n_layers: int = eqx.field(static=True)
    width_scale: int = eqx.field(static=True)
    mlp: eqx.Module
    normal_or_standard: Union[Normalization, Standardization]
    denorm: Denormalization
    latent_scale: ArrayLike

    def __init__(
        self,
        spatial_dim: int,
        latent_dim: int,
        out_dim: int,
        activation: Union[Callable, str],
        width_scale: int = 32,
        n_layers: int = 4,
        mean_x: Array = None,
        std_x: Array = None,
        min_x: Array = None,
        max_x: Array = None,
        final_activation=None,
        mean_field: ArrayLike = None,
        std_field: ArrayLike = None,
        latent_scale: ArrayLike = None,
        *,
        key: PRNGKeyArray,
        **_,
    ):
        if not isinstance(activation, Callable):
            activation = get_activation(activation)
        super().__init__(
            spatial_dim=spatial_dim, latent_dim=latent_dim, out_dim=out_dim
        )
        self.n_layers = n_layers
        self.width_scale = width_scale
        if final_activation is None:
            self.mlp = eqx.nn.MLP(
                in_size=latent_dim + spatial_dim,
                out_size=out_dim,
                width_size=width_scale,
                depth=n_layers,
                activation=activation,
                key=key,
            )
        else:
            print("Using final activation - ", final_activation)
            final_activation = get_activation(final_activation)
            self.mlp = eqx.nn.MLP(
                in_size=latent_dim + spatial_dim,
                out_size=out_dim,
                width_size=width_scale,
                depth=n_layers,
                activation=activation,
                final_activation=final_activation,
                key=key,
            )
        if mean_x is not None and std_x is not None:
            self.normal_or_standard = Normalization(mean_x, std_x)
        elif min_x is not None and max_x is not None:
            self.normal_or_standard = Standardization(min_x, max_x)
        else:
            self.normal_or_standard = None

        if mean_field is not None and std_field is not None:
            self.denorm = Denormalization(mean_field, std_field)
            print("Denormalizing output")
        else:
            self.denorm = None
        if latent_scale is not None:
            self.latent_scale = latent_scale
            print(f"Scaling latent space by {latent_scale}")
        else:
            self.latent_scale = None

        # pretty print the paramters
        print(f"Decoder with {n_layers} layers and width {width_scale}")
        print(f"Activation: {activation}")

    def __call__(self, x: Array) -> Array:
        if self.normal_or_standard is not None:
            coords = x[: self.spatial_dim]
            coords = self.normal_or_standard(coords)
            x = x.at[: self.spatial_dim].set(coords)
        if self.latent_scale is not None:
            x = x.at[self.spatial_dim :].set(
                x[self.spatial_dim :] * jax.lax.stop_gradient(self.latent_scale)
            )
        x = self.mlp(x)
        if self.denorm is not None:
            print("Denormalizing output")
            x = self.denorm(x)
        return x

    def call_coords_latent(self, coords: Array, latent: Array) -> Array:
        x = jax.numpy.concatenate([coords, latent], axis=-1)
        return self(x)

    def grads_x(self, coord, latent):
        return jax.jacrev(self.call_coords_latent, argnums=0)(coord, latent)

    def second_grads_x(self, coord, latent):
        hessian = jax.hessian(self.call_coords_latent, argnums=0)(coord, latent)
        return jnp.stack([hessian[:, i, i] for i in range(self.spatial_dim)], axis=-1)


class HyperDecoder(BaseDecoder):
    """
    Hyper decoder adopted from DINo (https://arxiv.org/abs/2209.14855).
    """

    n_layers: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    filters: eqx.nn.Sequential
    linear_layers_inr: eqx.nn.Sequential
    linear_layers_hyper: eqx.nn.Sequential
    out_layer: eqx.nn.Linear
    denorm: Denormalization
    std_coords: Standardization

    def __init__(
        self,
        spatial_dim: int,
        latent_dim: int,
        out_dim: int,
        n_layers: int,
        width: int,
        input_scale: float = 1.0,
        min_x: ArrayLike = None,
        max_x: ArrayLike = None,
        mean_field: ArrayLike = None,
        std_field: ArrayLike = None,
        std_coords=False,
        *,
        key: PRNGKeyArray,
        **_,
    ):
        super().__init__(
            spatial_dim=spatial_dim, latent_dim=latent_dim, out_dim=out_dim
        )
        self.n_layers = n_layers
        self.width = width
        assert n_layers > 0, "Number of layers must be positive"
        assert width % 2 == 0, "Width must be even"

        if std_coords:
            self.std_coords = Standardization(min_x, max_x)
            print("Standardizing coordinates")
        else:
            self.std_coords = None

        key, *subkeys = jax.random.split(key, n_layers + 2)
        filters = [
            FourierLayer(
                in_dim=spatial_dim,
                width=self.width // 2,
                weight_scale=1.0,
                trainable=True,
                key=subkeys[i],
            )
            for i in range(n_layers + 1)
        ]
        for i in range(n_layers + 1):
            filters[i] = eqx.tree_at(
                lambda layer: layer.weights,
                filters[i],
                jax.nn.initializers.he_uniform()(subkeys[i], filters[i].weights.shape),
            )
        self.filters = eqx.nn.Sequential(filters)

        key, *subkeys = jax.random.split(key, n_layers + 2)
        inr_layers = [
            eqx.nn.Linear(self.spatial_dim, self.width, use_bias=True, key=subkeys[0])
        ] + [
            eqx.nn.Linear(self.width, self.width, use_bias=True, key=subkeys[i])
            for i in range(1, n_layers + 1)
        ]
        bound1 = 1 / jnp.sqrt(self.spatial_dim)
        bound = 1 / jnp.sqrt(self.width)
        bounds = [bound1] + [bound] * n_layers
        for i, bound in enumerate(bounds):
            key, subkey, subkey2 = jax.random.split(key, 3)
            inr_layers[i] = eqx.tree_at(
                lambda layer: layer.bias,
                inr_layers[i],
                jax.random.uniform(
                    subkey, inr_layers[i].bias.shape, minval=-bound, maxval=bound
                ),
            )
            inr_layers[i] = eqx.tree_at(
                lambda layer: layer.weight,
                inr_layers[i],
                jax.nn.initializers.he_uniform()(subkey2, inr_layers[i].weight.shape),
            )
        self.linear_layers_inr = eqx.nn.Sequential(inr_layers)

        key, *subkeys = jax.random.split(key, n_layers + 2)
        self.linear_layers_hyper = eqx.nn.Sequential(
            [
                eqx.nn.Linear(
                    self.latent_dim, self.width, use_bias=False, key=subkeys[i]
                )
                for i in range(n_layers + 1)
            ]
        )

        key, subkey = jax.random.split(key)
        self.out_layer = eqx.nn.Linear(self.width, self.out_dim, key=subkey)

        # pretty print the paramters
        print(f"Hyper Decoder with {n_layers} layers and width {width}")
        print(f"Input scale: {input_scale}")

        if mean_field is not None and std_field is not None:
            self.denorm = Denormalization(mean_field, std_field)
        else:
            self.denorm = None

    def __call__(self, x: Array) -> Array:
        return self.call_coords_latent(x[: self.spatial_dim], x[self.spatial_dim :])

    def call_coords_latent(
        self, coords: PRNGKeyArray, latent: PRNGKeyArray
    ) -> PRNGKeyArray:
        if self.std_coords is not None:
            coords = self.std_coords(coords)
        hyper_bias = self.linear_layers_hyper[0](latent)
        bias = self.linear_layers_inr[0](coords * 0)
        x = hyper_bias + bias
        x = x * self.filters[0](coords)
        for i in range(1, self.n_layers):
            hyper_bias = self.linear_layers_hyper[i](latent)
            x = self.linear_layers_inr[i](x) + hyper_bias
            x = x * self.filters[i](coords)
        x = self.out_layer(x)
        if self.denorm is not None:
            x = self.denorm(x)
        return x

    def grads_x(self, coord: PRNGKeyArray, latent: PRNGKeyArray) -> PRNGKeyArray:
        return jax.jacrev(self.call_coords_latent, argnums=0)(coord, latent)

    def second_grads_x(self, coord: PRNGKeyArray, latent: PRNGKeyArray) -> PRNGKeyArray:
        hessian = jax.hessian(self.call_coords_latent, argnums=0)(coord, latent)
        return jnp.stack([hessian[:, i, i] for i in range(self.spatial_dim)], axis=-1)


class HyperConcatMLP(eqx.Module):
    """
    Parameterized dynamics network.
    """

    latent_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    param_size: int = eqx.field(static=True)
    normalization: Normalization
    psi_layers: List[eqx.Module]
    psi_activations: List[eqx.Module]
    param_transform: eqx.Module
    latent_scale: ArrayLike

    def __init__(
        self,
        param_size: int,
        latent_dim: int,
        depth: int,
        width: int,
        activation: str,
        mean: ArrayLike = None,
        std: ArrayLike = None,
        latent_scale: ArrayLike = None,
        initializer: str = None,
        *,
        key: PRNGKeyArray,
    ):
        self.depth = depth
        self.width = width
        self.latent_dim = latent_dim
        self.param_size = param_size

        if mean is not None and std is not None:
            self.normalization = Normalization(mean, std)
        else:
            self.normalization = None

        key, subkey = jax.random.split(key)
        layers = [
            eqx.nn.Linear(self.latent_dim * 2, self.width, key=subkey),
        ]
        activations = [get_activation(activation)]
        for _ in range(1, self.depth):
            key, subkey = jax.random.split(key)
            layers += [
                eqx.nn.Linear(self.width, self.width, key=subkey),
            ]
            activations += [get_activation(activation)]
        key, subkey = jax.random.split(key)
        layers += [eqx.nn.Linear(self.width, self.latent_dim, key=subkey)]

        if initializer is not None:
            print("Initializing weights with", initializer)
            initializer = get_initializer(initializer)
            for i, layer in enumerate(layers):
                key, subkey = jax.random.split(key)
                shape = layer.weight.shape
                new_weight = initializer(subkey, shape, layer.weight.dtype)
                layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
                layers[i] = layer

        self.psi_layers = layers
        self.psi_activations = activations

        key, subkey = jax.random.split(key)
        if self.normalization is not None:
            param_transform = [
                self.normalization,
                eqx.nn.Linear(param_size, self.latent_dim, key=subkey),
            ]
            if initializer is not None:
                layer = param_transform[1]
                key, subkey = jax.random.split(key)
                shape = layer.weight.shape
                new_weight = initializer(subkey, shape, layer.weight.dtype)
                layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
                param_transform[1] = layer
        else:
            param_transform = [eqx.nn.Linear(param_size, self.latent_dim, key=subkey)]
            if initializer is not None:
                layer = param_transform[0]
                key, subkey = jax.random.split(key)
                shape = layer.weight.shape
                new_weight = initializer(subkey, shape, layer.weight.dtype)
                layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
                param_transform[0] = layer
        self.param_transform = eqx.nn.Sequential(param_transform)

        if latent_scale is not None:
            self.latent_scale = latent_scale
            print("Psi with latent scale")
        else:
            self.latent_scale = None

        print(
            f"PSI: Hyper Concat MLP with {depth} layers and width {width} using {activation} activation"
        )

    def __call__(self, t, x, args, state):

        if self.latent_scale is not None:
            x = x * jax.lax.stop_gradient(self.latent_scale)

        x = jnp.concatenate([x, self.param_transform(args)], axis=-1)
        for i, layer in enumerate(self.psi_layers[:-1]):
            x = layer(x)
            x = self.psi_activations[i](x)
        x = self.psi_layers[-1](x)
        return x, state


class OdeMLP(eqx.Module):
    """
    Simple dynamics network with no linear transformation for the papameters (if parameterized).
    """

    mlp: eqx.Module
    normalization: eqx.Module

    def __init__(self, mlp, mean=None, std=None, initializer=None):
        self.mlp = mlp
        if mean is not None and std is not None:
            self.normalization = Normalization(mean, std)
            print("ODE MLP with parameter normalization")
            print(f"Mean: {mean}")
            print(f"Std: {std}")
        else:
            self.normalization = None
            print("ODE MLP without parameter normalization")

    def __call__(self, t, x, args, state):
        if args is not None:
            if args.ndim == 0:
                args = args[jnp.newaxis]
            if self.normalization is not None:
                args = self.normalization(args)
            x = jnp.concatenate([x, args], axis=-1)
        return self.mlp(x), state


class NeuralODE(eqx.Module):
    """
    Wrapper class for the dynamics network for integration as latent ODE.
    """

    in_size: int = eqx.field(static=True)
    param_size: int = eqx.field(static=True)
    adaptive: bool = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    mlp: OdeMLP
    solver: diff.AbstractSolver

    def __init__(
        self,
        in_size: int,
        depth: int,
        width: int,
        activation: str,
        param_size: int = 0,
        node_arch="mlp",
        solver: str = "bosh3",
        adaptive: bool = False,
        max_steps: int = None,
        mean_params: np.array = None,
        std_params: np.array = None,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key, subkey = jax.random.split(key)
        self.in_size = in_size
        if node_arch == "mlp":
            mlp = [
                eqx.nn.Linear(in_size + param_size, width, key=subkey),
                eqx.nn.Lambda(get_activation(activation)),
            ]
            for i in range(depth - 1):
                key, subkey = jax.random.split(key)
                mlp += [
                    eqx.nn.Linear(width, width, key=subkey),
                    eqx.nn.Lambda(get_activation(activation)),
                ]
            key, subkey = jax.random.split(key)
            mlp += [eqx.nn.Linear(width, in_size, key=subkey)]
            mlp = eqx.nn.Sequential(mlp)
            self.mlp = OdeMLP(mlp, mean_params, std_params, **kwargs)
        elif node_arch == "hyper_concat":
            mlp = HyperConcatMLP(
                param_size,
                in_size,
                depth,
                width,
                activation,
                mean_params,
                std_params,
                key=key,
                **kwargs,
            )
            self.mlp = mlp
        else:
            raise ValueError(f"{node_arch} not supported")
        self.solver = get_solver(solver)
        self.param_size = param_size
        self.adaptive = adaptive
        self.max_steps = max_steps

        print(
            f"Neural ODE with {depth} layers and width {width} using {activation} activation"
        )
        print(f"Solver: {solver}")
        print(f"Adaptive: {adaptive}")
        if adaptive:
            print(f"Max steps: {max_steps}")

    def __call__(
        self,
        x0: Array,
        ts: Array,
        solver_state: PyTree = None,
        args: Array = None,
        solver: diff.AbstractSolver = None,
        adaptive: bool = None,
        max_steps: int = None,
        state=None,
    ) -> Array:
        """
        Integrate the dynamics network using the numerical solver and return the solution at all the given time steps in ts.
        """
        solver = self.solver if solver is None else solver
        adaptive = self.adaptive if adaptive is None else adaptive
        max_steps = self.max_steps if max_steps is None else max_steps

        def mlp_fn(t, x, args):
            arg, state = args
            out, _ = self.mlp(t, x, arg, state)
            return out

        term = diff.ODETerm(mlp_fn)

        if adaptive:
            sol = diff.diffeqsolve(
                term,
                solver,
                y0=x0,
                t0=ts[0],
                t1=ts[-1],
                dt0=None,
                stepsize_controller=diff.PIDController(rtol=1e-3, atol=1e-6),
                max_steps=max_steps * (len(ts) - 1) if max_steps is not None else None,
                saveat=diff.SaveAt(ts=ts, solver_state=True),
                args=(args, state),
                solver_state=solver_state,
                throw=False,
            )
        else:
            sol = diff.diffeqsolve(
                term,
                solver,
                y0=x0,
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                saveat=diff.SaveAt(ts=ts, solver_state=True),
                args=(args, state),
                solver_state=solver_state,
                throw=False,
            )
        return sol.ys, sol.solver_state, state

    def call_step(
        self,
        x0: Array,
        t0: float,
        t1: float,
        solver_state: PyTree = None,
        args: Array = None,
        solver: diff.AbstractSolver = None,
        adaptive: bool = None,
        max_steps: int = None,
        state=None,
    ) -> Array:
        """
        Integrate the dynamics network using the numerical solver and return the solution at the given time t1.
        """
        solver = self.solver if solver is None else solver
        adaptive = self.adaptive if adaptive is None else adaptive
        max_steps = self.max_steps if max_steps is None else max_steps

        def mlp_fn(t, x, args):
            arg, state = args
            out, _ = self.mlp(t, x, arg, state)
            return out

        term = diff.ODETerm(mlp_fn)

        if adaptive:
            sol = diff.diffeqsolve(
                term,
                solver,
                y0=x0,
                t0=t0,
                t1=t1,
                dt0=None,
                stepsize_controller=diff.PIDController(rtol=1e-3, atol=1e-6),
                max_steps=max_steps,
                saveat=diff.SaveAt(t1=True, solver_state=True),
                args=(args, state),
                solver_state=solver_state,
                throw=False,
            )
        else:
            sol = diff.diffeqsolve(
                term,
                solver,
                y0=x0,
                t0=t0,
                t1=t1,
                dt0=t1 - t0,
                saveat=diff.SaveAt(t1=True, solver_state=True),
                args=(args, state),
                solver_state=solver_state,
                throw=False,
            )
        return sol.ys, sol.solver_state, state
