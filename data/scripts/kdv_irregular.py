import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
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

# check if there are nans in trajs
print(jnp.any(jnp.isnan(trajs)))


import numpy as np

X, Y = grid
num_devices = len(jax.devices())
mesh = jax.make_mesh((num_devices,), ("shard",), devices=jax.devices())
pspec_data = jax.sharding.PartitionSpec(("shard",))
sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
trajs = eqx.filter_shard(trajs, sharding_data)

snapshot = trajs[0, 0]

# create an interpolation function for the snapshot that can be evaluated at any point in the domain
from jax.scipy.interpolate import RegularGridInterpolator

interpolator = RegularGridInterpolator(
    (grid[0][:, 0], grid[1][0, :]), np.transpose(trajs, (3, 4, 0, 1, 2))
)


SUB_RATIO = 2 / 100
key, subkey = jax.random.split(key)
X_highres = jnp.linspace(grid[0][:, 0][0], grid[0][:, 0][-1], 128)
Y_highres = jnp.linspace(grid[0][:, 0][0], grid[0][:, 0][-1], 128)
X_highres, Y_highres = jnp.meshgrid(X_highres, Y_highres, indexing="ij")
X_flat = X_highres.flatten()
Y_flat = Y_highres.flatten()
Xs = jnp.stack([X_flat, Y_flat], axis=1)
Z_highres = (
    interpolator(Xs).transpose(1, 2, 3, 0).reshape(NUM_SAMPLES, 100, 1, 128, 128)
)


# randomly sample 1024 coordinates from Xs for each trajectory in trajss
n_coords = int(SUB_RATIO * 64**2)
coords_idx = jax.random.randint(subkey, (n_coords,), 0, Xs.shape[0])
coords_idx = jnp.tile(coords_idx, (NUM_SAMPLES, 1))
irregular_trajs = jax.vmap(
    jax.vmap(jax.vmap(jnp.take, in_axes=(0, None)), in_axes=(0, None))
)(Z_highres.reshape(NUM_SAMPLES, 100, 1, -1), coords_idx)
irregular_coords = jax.vmap(jnp.take, in_axes=(None, 0, None))(Xs, coords_idx, 0)
print(SUB_RATIO)


ds = xarray.Dataset(
    {
        "irregular_u": (("batch", "time", "dim", "coords"), irregular_trajs),
        "regular_u": (("batch", "time", "dim", "x", "y"), trajs),
        "irregular_coords": (("batch", "coords", "sdim"), irregular_coords),
        "regular_grid": (("xy", "x", "y"), grid),
        "dt": (("batch",), [DT for _ in trajs]),
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
ds.to_netcdf(f"kdv_2d_irreg_{SUB_RATIO}_CUTOFF=2_{NUM_SPATIAL_POINTS}_ins={1}.h5")
