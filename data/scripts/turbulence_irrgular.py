"""
Generating sparse Navier-Stokes data for a 2D turbulence simulation using JAX-CFD.
"""

import os
from functools import partial

import equinox as eqx
import h5py as h5
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray
from jax.scipy.interpolate import RegularGridInterpolator
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
SUB_RATIO = 5 / 100  # set sampling ratio


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
trajss = np.stack([us, vs], axis=2)

X, Y = grid.mesh()


num_devices = len(jax.devices())
mesh = jax.make_mesh((num_devices,), ("shard",), devices=jax.devices())
pspec_data = jax.sharding.PartitionSpec(("shard",))
sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
trajss = eqx.filter_shard(trajss, sharding_data)


interpolator = RegularGridInterpolator(
    (grid.axes()[0], grid.axes()[1]), np.transpose(trajss, (3, 4, 0, 1, 2))
)


X_highres = jnp.linspace(grid.axes()[0][0], grid.axes()[0][-1], 128)
Y_highres = jnp.linspace(grid.axes()[0][0], grid.axes()[0][-1], 128)
X_highres, Y_highres = jnp.meshgrid(X_highres, Y_highres, indexing="ij")
X_flat = X_highres.flatten()
Y_flat = Y_highres.flatten()
Xs = jnp.stack([X_flat, Y_flat], axis=1)
Z_highres = interpolator(Xs).transpose(1, 2, 3, 0).reshape(N_SAMPLES, 200, 2, 128, 128)

key, subkey = jax.random.split(key)

n_coords = int(SUB_RATIO * size**2)
coords_idx = jax.random.randint(subkey, (n_coords,), 0, Xs.shape[0])
coords_idx = jnp.tile(coords_idx, (N_SAMPLES, 1))
irregular_trajs = jax.vmap(
    jax.vmap(jax.vmap(jnp.take, in_axes=(0, None)), in_axes=(0, None))
)(Z_highres.reshape(N_SAMPLES, 2, 2, -1), coords_idx)
irregular_coords = jax.vmap(jnp.take, in_axes=(None, 0, None))(Xs, coords_idx, 0)

ds = xarray.Dataset(
    {
        "irregular_u": (("batch", "time", "dim", "coords"), irregular_trajs),
        "regular_u": (("batch", "time", "dim", "x", "y"), trajss),
        "irregular_coords": (("batch", "coords", "sdim"), irregular_coords),
        "regular_grid": (("xy", "x", "y"), np.stack([X, Y])),
        "dt": (("batch",), dts),
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
ds.to_netcdf(f"ns_turbulence_irregular{SUB_RATIO}_{size}x{size}_ins={inner_steps}.h5")
