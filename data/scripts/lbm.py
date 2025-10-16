"""
Generating LBM data for a cylinder in a channel flow using XLB.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import xlb
from netCDF4 import Dataset
from xlb.compute_backend import ComputeBackend
from xlb.distribute import distribute
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import (
    ExtrapolationOutflowBC,
    HalfwayBounceBackBC,
    RegularizedBC,
)
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import save_image

# computational backend
backend = ComputeBackend.JAX

# precision policy
precision_policy = PrecisionPolicy.FP32FP32

# choose the velocity set
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=backend)

# configure the geometry and resolution of the computational grid
cylinder_diameter = 20
nx = int(9 * cylinder_diameter)
ny = int(4.1 * cylinder_diameter)
grid_shape = (nx, ny)
grid = grid_factory(grid_shape, compute_backend=backend)

# configure the inlet velocity
inlet_velocity_mean = 0.1
u_max = 1.5 * inlet_velocity_mean
# Now we initialize XLB which configures default settings for velocity_set (D2Q9, D3Q19 or D3Q27), computational backend (JAX or WARP) and the compute/storage precision policy.
xlb.init(
    velocity_set=velocity_set,
    default_backend=backend,
    default_precision_policy=precision_policy,
)


def bc_profile(u_max):
    u_max = u_max  # u_max = 0.04

    def bc_profile_jax():
        # Get the grid dimensions for the y direction
        H_y = float(grid_shape[1] - 1)  # Height in y direction

        y = jnp.arange(grid_shape[1])

        # Calculate normalized distance from center
        y_center = y - (H_y / 2.0)
        r_squared = (2.0 * y_center / H_y) ** 2.0

        # Parabolic profile for x velocity, zero for y and z
        u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
        u_y = jnp.zeros_like(u_x)

        return jnp.stack([u_x, u_y])

    return bc_profile_jax


def setup_boundaries(u_max):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    cyliner_radius = cylinder_diameter // 2
    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    indices = np.where(
        (X - 2.0 * cylinder_diameter) ** 2 + (Y - 2.0 * cylinder_diameter) ** 2
        < cyliner_radius**2
    )
    cyliner = [tuple(indices[i]) for i in range(velocity_set.d)]

    bc_inlet = RegularizedBC("velocity", profile=bc_profile(u_max), indices=inlet)
    bc_walls = HalfwayBounceBackBC(indices=walls)
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    # bc_outlet = RegularizedBC("pressure", prescribed_value=(1., ), indices=outlet)
    bc_cyliner = HalfwayBounceBackBC(indices=cyliner)
    return [bc_walls, bc_inlet, bc_outlet, bc_cyliner]


macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, backend=ComputeBackend.JAX
    ),
)
bc_list = setup_boundaries(u_max)
stepper = IncompressibleNavierStokesStepper(
    grid=grid, boundary_conditions=bc_list, collision_type="BGK", streaming_scheme="push"
)
stepper = distribute(stepper, grid, velocity_set)

key = jax.random.PRNGKey(0)
NUM_SAMPLES = 40
# reynolds = jax.random.uniform(key, (NUM_SAMPLES,), jnp.float32, minval=100.0, maxval=200.0) # Test data
# reynolds = jax.random.uniform(key, (NUM_SAMPLES,), jnp.float32, minval=200.0, maxval=300.0) # Test data - extrapolation
reynolds = jax.numpy.linspace(100.0, 200.0, NUM_SAMPLES)  # Training Data
trajs = []
omegas = []

f0, f1, bc_mask, missing_mask = stepper.prepare_fields()
for Re in reynolds:
    visc = inlet_velocity_mean * cylinder_diameter / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    omegas.append(omega)
omegas = jnp.array(omegas)
f0 = jnp.tile(f0, (NUM_SAMPLES, 1, 1, 1))

trajs = [f0]
for t in range(1, 5000):
    _, f0 = jax.vmap(stepper, in_axes=(0, None, None, None, 0, None))(
        f0, f1, bc_mask, missing_mask, omegas, 0
    )
    if t % 5 == 0:
        trajs.append(f0)

trajs = jnp.stack(trajs, axis=0)
trajs = trajs.transpose(1, 0, 2, 3, 4)
# trajs = jax.vmap(jax.vmap(macro))(trajs)[1]           # Run to convert to velocity for DINo

x = np.arange(grid_shape[0])
y = np.arange(grid_shape[1])

dataset = xr.Dataset(
    {
        "u": (
            ("batch", "time", "field", "x", "y"),
            trajs[:600:],
        ),  # discard the first 600 steps (undeveloped flow)
        "omega": (("batch",), omegas),
        "reynolds": (("batch",), reynolds),
    },
    coords={"x": x, "y": y},
    attrs={
        "inner_step": 5,
        "outer_steps": 500,
        "cylinder_diameter": cylinder_diameter,
        "inlet_velocity_mean": inlet_velocity_mean,
        "u_max": u_max,
    },
)

dataset.to_netcdf(f"cylinder_population_ins=5_N40.h5")

# dataset.to_netcdf(f"cylinder_ins=5_N40.h5") # if velocity for DINo
