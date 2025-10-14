"""
Generate Navier-Stokes turbulence data in 2D with JAX-CFD.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial

import h5py as h5
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import seaborn as sns
import xarray
from netCDF4 import Dataset


def loguniform(key, low=1e-3, high=1e-2, size=None):
    return jnp.exp(
        jax.random.uniform(key, size, minval=jnp.log(low), maxval=jnp.log(high))
    )


size = 64
density = 1.0
viscosity = 1e-2
seed = 10
inner_steps = 5
outer_steps = 200

max_velocity = 1.0
cfl_safety_factor = 0.5

N_SAMPLES = 1024


# Define the physical dimensions of the simulation.
grid = cfd.grids.Grid((size, size), domain=((0, jnp.pi), (0, jnp.pi)))

# Construct a random initial velocity. The `filtered_velocity_field` function
# ensures that the initial velocity is divergence free and it filters out
# high frequency fluctuations.
v0 = cfd.initial_conditions.filtered_velocity_field(
    jax.random.PRNGKey(seed), grid, max_velocity
)

# Choose a time step.
dt = cfd.equations.stable_time_step(max_velocity, cfl_safety_factor, viscosity, grid)

# Define a step function and use it to compute a trajectory.
step_fn = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=density, viscosity=viscosity, dt=dt, grid=grid
    ),
    steps=inner_steps,
)
rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
trajs = []
viscosities = []
densities = []
dts = []
key = jax.random.PRNGKey(seed)
for i in range(0, N_SAMPLES):
    key, subkey = jax.random.split(key)
    v0 = cfd.initial_conditions.filtered_velocity_field(subkey, grid, max_velocity)
    _, trajectory = jax.device_get(rollout_fn(v0))
    trajs.append(trajectory)
    viscosities.append(viscosity)
    densities.append(density)
    dts.append(dt)


us = np.stack([traj[0].data for traj in trajs])
vs = np.stack([traj[1].data for traj in trajs])
trajs = np.stack([us, vs], axis=2)

X, Y = grid.mesh()

ds = xarray.Dataset(
    {
        "u": (("batch", "time", "dim", "x", "y"), trajs),
        "dt": (("batch",), dts),
        "regular_grid": (("xy", "x", "y"), np.stack([X, Y])),
        "density": (("batch",), np.array([density for density in densities])),
        "viscosity": (("batch",), np.array([viscosity for viscosity in viscosities])),
    },
    coords={
        "batch": np.arange(len(trajs)),
        "x": grid.axes()[0],
        "y": grid.axes()[1],
    },
    attrs={
        "dx": grid.step,
        "inner_steps": inner_steps,
        "outer_steps": outer_steps,
        "velocity_max": max_velocity,
        "lengths": [grid.domain[0][1], grid.domain[1][1]],
    },
)
ds.to_netcdf(f"ns_turbulence_{size}x{size}_ins={inner_steps}.h5")
