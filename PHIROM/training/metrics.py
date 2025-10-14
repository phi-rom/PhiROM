import jax.numpy as jnp
from jaxtyping import Array


def mean_squared_error(predicted_trajectory: Array, exact_trajectory: Array):
    """
    Compute the mean squared error.

    Args:
        predicted_trajectory: Array, predicted trajectory (time_steps, field_dim, nx, ny, ...)
        exact_trajectory: Array, exact trajectory (time_steps, field_dim, nx, ny, ...)

    Returns:
        Array, mean squared error over the spatial dimensions (time_steps, field_dim)
    """
    return jnp.mean(
        (predicted_trajectory - exact_trajectory) ** 2,
        axis=(*range(2, predicted_trajectory.ndim),),
    )


def mean_absolute_error(predicted_trajectory: Array, exact_trajectory: Array):
    """
    Compute the mean absolute error.

    Args:
        predicted_trajectory: Array, predicted trajectory (time_steps, field_dim, nx, ny, ...)
        exact_trajectory: Array, exact trajectory (time_steps, field_dim, nx, ny, ...)

    Returns:
        Array, mean absolute error over the spatial dimensions (time_steps, field_dim)
    """
    return jnp.mean(
        jnp.abs(predicted_trajectory - exact_trajectory),
        axis=(*range(2, predicted_trajectory.ndim),),
    )


def normalized_mse(predicted_trajectory, exact_trajectory):
    """
    Compute the normalized mean squared error.

    Args:
        predicted_trajectory: Array, predicted trajectory (time_steps, field_dim, nx, ny, ...)
        exact_trajectory: Array, exact trajectory (time_steps, field_dim, nx, ny, ...)

    Returns:
        Array, normalized squared error over the spatial dimensions (time_steps, field_dim)
    """
    return jnp.linalg.norm(
        predicted_trajectory - exact_trajectory,
        axis=(*range(2, predicted_trajectory.ndim),),
    ) / jnp.linalg.norm(exact_trajectory, axis=(*range(2, predicted_trajectory.ndim),))
