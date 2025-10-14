"""
This script generates a 2D Korteweg-de Vries (KdV) equation dataset using he spectral method from the Exponax library.
"""

import exponax as ex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray
from netCDF4 import Dataset

DOMAIN_EXTENT = 15.0
NUM_SAMPLES = 1024
NUM_SPATIAL_POINTS = 64
DT = 0.04
HYPER_NU = 0.03
NU = 0.0

key = jax.random.PRNGKey(42)
ic_gen = ex.ic.RandomTruncatedFourierSeries(2, cutoff=2, max_one=True)
ics = ex.build_ic_set(
    ic_gen, num_points=NUM_SPATIAL_POINTS, num_samples=NUM_SAMPLES, key=key
)

grid = ex.make_grid(2, DOMAIN_EXTENT, NUM_SPATIAL_POINTS)

kdv_stepper = ex.stepper.KortewegDeVries(
    2,
    DOMAIN_EXTENT,
    NUM_SPATIAL_POINTS,
    DT,
    single_channel=True,
    diffusivity=NU,
    hyper_diffusivity=HYPER_NU,
    order=2,
)

trajs = jax.vmap(ex.rollout(kdv_stepper, 100, include_init=False))(ics)


ds = xarray.Dataset(
    {
        "u": (("batch", "time", "field", "x", "y"), trajs),
        "dt": (("batch",), [DT for _ in trajs]),
        "grid": (("dim", "x", "y"), grid),
    },
    coords={},
    attrs={
        "dx": kdv_stepper.dx,
        "inner_steps": 1,
        "outer_steps": 100,
        "lengths": [DOMAIN_EXTENT],
        "hyper_diffusivity": HYPER_NU,
        "domain_extent": DOMAIN_EXTENT,
    },
)
ds.to_netcdf(f"kdv_2d_{NUM_SPATIAL_POINTS}_ins={1}.h5")
