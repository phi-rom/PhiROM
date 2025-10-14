"""
This script generates a dataset of 2D diffusion fields using the explicit finite difference method.
The initial conditions are Gaussian distributions with random means and standard deviations and scaled by a random amplitude.
"""

import jax.numpy as jnp
import numpy as np
import xarray
from netCDF4 import Dataset

nx, ny = 40, 40  # Grid size
dx, dy = 1, 1  # Spatial step
D = 2.0  # Diffusion coefficient
dt = 0.1  # Time step
T = 20  # Total time for the simulation

# Stability condition for the explicit method
stability_limit = dx**2 * dy**2 / (2 * D * (dx**2 + dy**2))
if dt > stability_limit:
    print(f"Warning: Time step {dt} exceeds the stability limit {stability_limit}.")

x = np.linspace(-19.5, 19.5, nx)
y = np.linspace(-19.5, 19.5, ny)
X, Y = np.meshgrid(x, y)
SIZE_DATASET = 200

X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
dataset = np.zeros((SIZE_DATASET, int(T / dt) + 1, ny, nx))

np.random.seed(0)
for i in range(SIZE_DATASET):
    sigma = np.random.uniform(3.0, 10.0)
    x_mean, y_mean = np.random.randint(-12, 12), np.random.randint(-12, 12)
    amplitude = np.random.uniform(0.5, 2)

    u0 = amplitude * np.exp(-((X - x_mean) ** 2 + (Y - y_mean) ** 2) / (2 * sigma**2))
    dataset[i, 0] = u0

    u = u0.copy()
    nsteps = int(T / dt)
    for step in range(nsteps):
        field_1 = jnp.array(u)[jnp.newaxis, :, :]
        field_1_padded = jnp.pad(field_1, ((0, 0), (1, 1), (1, 1)), mode="constant")
        field_1_padded = field_1_padded.at[:, 1:-1, 1:-1].set(
            field_1_padded[:, 1:-1, 1:-1]
            + D
            * dt
            * (
                (
                    field_1_padded[:, 2:, 1:-1]
                    - 2 * field_1_padded[:, 1:-1, 1:-1]
                    + field_1_padded[:, :-2, 1:-1]
                )
                / dx**2
                + (
                    field_1_padded[:, 1:-1, 2:]
                    - 2 * field_1_padded[:, 1:-1, 1:-1]
                    + field_1_padded[:, 1:-1, :-2]
                )
                / dx**2
            )
        )
        field_2 = field_1_padded[:, 1:-1, 1:-1]
        u_new = np.array(field_2[0])
        u = u_new.copy()
        dataset[i, step + 1] = u

x = np.pad(x, (1, 1), mode="constant")
y = np.pad(y, (1, 1), mode="constant")
x[0] = x[1] - dx
x[-1] = x[-2] + dx
y[0] = y[1] - dy
y[-1] = y[-2] + dy
dataset_bc = np.pad(dataset, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")

size_x = nx // 2
size_y = ny // 2
dts = np.array([0.1 for _ in range(SIZE_DATASET)])
diffusivity = np.array([D for _ in range(SIZE_DATASET)])

ds = xarray.Dataset(
    {
        "u": (("batch", "time", "x", "y"), dataset_bc),
        "dt": (("batch",), dts),
        "diffusivity": (("batch",), diffusivity),
    },
    coords={
        "batch": np.arange(SIZE_DATASET),
        "x": x,
        "y": y,
    },
    attrs={"dx": 1.0, "outer_steps": nsteps + 1, "lengths": [size_x, size_y]},
)
ds.to_netcdf(f"diffusion_{nx + 2}x{ny + 2}.h5")
