"""
This script generates a dataset of 1D Burgers' equation fields using the explicit finite difference method.
The initial condition is given by a prespecfied profile.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial

import h5py as h5
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from netCDF4 import Dataset

dt = 0.07
dx = 0.39215686274509803


def evolve(field_1, dt, dx, x, mu, *args):
    q = field_1.reshape((1, -1))
    q_pad = jnp.pad(q, ((0, 0), (1, 1)), "edge")
    q_pad = q_pad.at[:, 1:-1].set(q)
    q_ = 1.0 * (
        -(0.5 * (q_pad[:, 1:-1]) ** 2 - 0.5 * (q_pad[:, 0:-2]) ** 2) / dx
        + 0.02 * jnp.exp(mu * x.reshape(1, -1))
    )
    q_ = q_.at[:, 0].set(0.0)
    return field_1 + dt * q_.reshape(field_1.shape)


with h5.File("./h5_f_0000000000.h5") as f:
    x = jnp.array(f["x"][:])
    time = jnp.array(f["time"][:])
    q = jnp.array(f["q"][:])


def gen(mu, q, x):
    sol = [q]
    for i in range(199):
        q = evolve(q, dt, dx, x, mu)
        sol.append(q)
    return jnp.stack(sol, axis=0).squeeze()


sols = []
mus_training = jnp.linspace(0.015, 0.03, 8)
mus_interp = [(mus_training[i] + mus_training[i + 1]) / 2 for i in range(7)]
mus_extrap = [mus_training[0] - 0.015 / 7, mus_training[-1] + 0.015 / 7]

sols = []
for mu in mus_training:
    sols.append(gen(mu, q, x))
for mu in mus_interp:
    sols.append(gen(mu, q, x))
for mu in mus_extrap:
    sols.append(gen(mu, q, x))

sols = jnp.stack(sols, axis=0)
sols = sols[:, :, jnp.newaxis]

ds = xarray.Dataset(
    {
        "u": (("batch", "time", "field", "x"), sols),
        "dt": (("batch",), np.array([dt for _ in range(len(sols))])),
        "dx": (("batch",), np.array([dx for _ in range(len(sols))])),
        "mu": (
            ("batch",),
            np.array(
                [mu for mu in list(mus_training) + list(mus_interp) + list(mus_extrap)]
            ),
        ),
    },
    coords={"batch": np.arange(len(sols)), "x": x.squeeze()},
    attrs={
        "time_steps": 200,
    },
)
ds.to_netcdf(f"burgers.h5")
