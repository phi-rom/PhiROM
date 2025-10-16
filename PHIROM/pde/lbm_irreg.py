import os
import sys

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
import torch
import xlb
from torch.utils.data import Dataset
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

print(xlb.__file__)


class IrregularCylindeDatasetTorch(Dataset):

    def __init__(
        self,
        path,
        start_time,
        num_time_steps,
        histpry_steps=1,
        future_steps=0,
        indices=None,
        paramed=False,
    ):
        self.num_time_steps = num_time_steps
        self.history_steps = histpry_steps
        self.future_steps = future_steps
        self.paramed = paramed
        self.max_start = self.num_time_steps - self.history_steps - self.future_steps
        with h5.File(path) as f:
            u = f["u"][:]
            irregular_u = f["irregular_u"][:]
            irregular_coords = f["irregular_coords"][:]
            x = f["x"][:]
            y = f["y"][:]
            omega = f["omega"][:]
            # reynold = f['reynold'][:]
            self.u_max = f.attrs["u_max"][:]
            self.cylinder_diameter = f.attrs["cylinder_diameter"][:]
            self.inner_steps = f.attrs["inner_step"][:]

        if indices is not None:
            if len(indices) == 2:
                print(indices)
                u = u[indices[0] : indices[1]]
                omega = omega[indices[0] : indices[1]]
            else:
                u = u[indices]
                omega = omega[indices]
            # reynold = reynold[indices[0]:indices[1]]

        u = u[:, start_time : start_time + num_time_steps]
        irregular_u = irregular_u[:, start_time : start_time + num_time_steps]
        # u = np.transpose(u, [0, 1, 2, 4, 3])
        # print(u.shape)
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")
        X = X.flatten()
        Y = Y.flatten()
        self.dt = 1.0
        t = (np.arange(0, num_time_steps) + start_time) * self.dt * self.inner_steps
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.u = torch.tensor(u)
        self.irregular_u = torch.tensor(irregular_u)
        self.irregular_coords = torch.tensor(irregular_coords)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.omega = torch.tensor(omega)
        # self.reynold = torch.tensor(reynold)
        self.t = torch.tensor(t)
        self.dt = torch.tensor(self.dt)
        self.dx = torch.tensor(x[1] - x[0])
        self.length = (x[-1], y[-1])
        if paramed:
            self.node_args = self.omega

    def compute_mean_std_fields(self):
        return (
            torch.mean(self.irregular_u, dim=(0, 1, 3)).numpy(),
            torch.std(self.irregular_u, dim=(0, 1, 3)).numpy(),
        )

    def compute_mean_std_coords(self):
        return (
            torch.tensor([torch.mean(self.x), torch.mean(self.y)]).numpy(),
            torch.tensor([torch.std(self.x), torch.std(self.y)]).numpy(),
        )

    def compute_min_max_coords(self):
        return (
            torch.tensor([torch.min(self.x), torch.min(self.y)]).numpy(),
            torch.tensor([torch.max(self.x), torch.max(self.y)]).numpy(),
        )

    def get_coordinates(self):
        return self.x, self.y, self.t

    def get_trajectory(self, idx):
        return self.u[idx]

    def __len__(self):
        return (self.max_start + 1) * len(self.u)

    def __getitem__(self, idx):
        traj_idx = idx // (self.max_start + 1)
        start_idx = idx % (self.max_start + 1)
        history = self.u[traj_idx, start_idx : start_idx + self.history_steps]
        history_irregular = self.irregular_u[
            traj_idx, start_idx : start_idx + self.history_steps
        ]
        coords_irregular = self.irregular_coords[traj_idx]
        if self.history_steps == 1:
            history = history.squeeze(dim=0)
            history_irregular = history_irregular.squeeze(dim=0)
        return (
            {
                "data_irregular": history_irregular,
                "coords_irregular": coords_irregular,
                "data": history,
                "t": self.t[start_idx : start_idx + self.history_steps].squeeze(dim=0),
                "time_idx": start_idx,
                "coords": self.X,
                "dt": self.dt,
                "dx": self.dx,
                "idx": traj_idx,
                "solver_args": [self.omega[traj_idx]],
                "node_args": self.omega[traj_idx],
            }
            if self.paramed
            else {
                "data_irregular": history_irregular,
                "coords_irregular": coords_irregular,
                "data": history,
                "t": self.t[start_idx : start_idx + self.history_steps].squeeze(dim=0),
                "time_idx": start_idx,
                "coords": self.X,
                "dt": self.dt,
                "dx": self.dx,
                "idx": traj_idx,
                "solver_args": [self.omega[traj_idx]],
            }
        )


class IrregularCylinderTrajDatasetTorch(Dataset):

    def __init__(self, path, start_time, num_time_steps, indices=None, paramed=False):
        self.num_time_steps = num_time_steps
        self.paramed = paramed
        with h5.File(path) as f:
            u = f["u"][:]
            irregular_u = f["irregular_u"][:]
            irregular_coords = f["irregular_coords"][:]
            x = f["x"][:]
            y = f["y"][:]
            omega = f["omega"][:]
            # reynold = f['reynold'][:]
            self.u_max = f.attrs["u_max"][:]
            self.cylinder_diameter = f.attrs["cylinder_diameter"][:]
            self.inner_steps = f.attrs["inner_step"][:]

        if indices is not None:
            if len(indices) == 2:
                u = u[indices[0] : indices[1]]
                omega = omega[indices[0] : indices[1]]
            else:
                u = u[indices]
                omega = omega[indices]
            # reynold = reynold[indices[0]:indices[1]]

        u = u[:, start_time : start_time + num_time_steps]
        irregular_u = irregular_u[:, start_time : start_time + num_time_steps]
        # u = np.transpose(u, [0, 1, 2, 4, 3])
        # convert x and y to float
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")
        X = X.flatten()
        Y = Y.flatten()
        self.dt = 1.0
        t = (np.arange(0, num_time_steps) + start_time) * self.dt * self.inner_steps
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.u = torch.tensor(u)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.omega = torch.tensor(omega)
        self.irregular_u = torch.tensor(irregular_u)
        self.irregular_coords = torch.tensor(irregular_coords)
        # self.reynold = torch.tensor(reynold)
        self.t = torch.tensor(t)
        self.dt = torch.tensor(self.dt)
        self.dx = torch.tensor(x[1] - x[0])
        self.length = (x[-1], y[-1])
        if paramed:
            self.node_args = self.omega

    def compute_mean_std_fields(self):
        return (
            torch.mean(self.irregular_u, dim=(0, 1, 3)).numpy(),
            torch.std(self.irregular_u, dim=(0, 1, 3)).numpy(),
        )

    def compute_mean_std_coords(self):
        return (
            torch.tensor([torch.mean(self.x), torch.mean(self.y)]).numpy(),
            torch.tensor([torch.std(self.x), torch.std(self.y)]).numpy(),
        )

    def compute_min_max_coords(self):
        return (
            torch.tensor([torch.min(self.x), torch.min(self.y)]).numpy(),
            torch.tensor([torch.max(self.x), torch.max(self.y)]).numpy(),
        )

    def get_coordinates(self):
        return self.x, self.y, self.t

    def get_trajectory(self, idx):
        return self.u[idx]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return (
            {
                "data_irregular": self.irregular_u[idx],
                "coords_irregular": self.irregular_coords[idx],
                "data": self.u[idx],
                "t": self.t,
                "coords": self.X,
                "dt": self.dt,
                "dx": self.dx,
                "idx": idx,
                "solver_args": [self.omega[idx]],
                "node_args": self.omega[idx],
            }
            if self.paramed
            else {
                "data_irregular": self.irregular_u[idx],
                "coords_irregular": self.irregular_coords[idx],
                "data": self.u[idx],
                "t": self.t,
                "coords": self.X,
                "dt": self.dt,
                "dx": self.dx,
                "idx": idx,
                "solver_args": [self.omega[idx]],
            }
        )


def cylinder_residual_builder(
    u_max, grid, grid_shape, cylinder_diameter, velocity_set, macro, window_length=20.0
):

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
        bc_cyliner = HalfwayBounceBackBC(indices=cyliner)
        return [bc_walls, bc_inlet, bc_outlet, bc_cyliner]

    bc_list = setup_boundaries(u_max)
    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=bc_list, collision_type="BGK", streaming_scheme="push"
    )
    stepper = distribute(stepper, grid, velocity_set)
    _, f1, bc_mask, missing_mask = stepper.prepare_fields()
    print("XLB Solver, # steps: ", window_length)

    # def residual(population, dt, dx, omega, *args):
    #     residual = 0
    #     f1 = field1
    #     f0 = population
    #     for _ in range(window_length):
    #         _, velocity_current = macro(f0)
    #         f0, f1 = stepper(f0, f1, bc_mask, missing_mask, omega, 0)
    #         f0, f1 = f1, f0
    #         _, velocity_next = macro(f0)
    #         residual += velocity_next - velocity_current

    #     return residual / window_length
    # return residual

    def residual(population, dt, dx, omega, *args):
        residual = 0.0
        population_prev = population
        for _ in range(window_length):
            _, population_next = stepper(
                population_prev, f1, bc_mask, missing_mask, omega, 0
            )

            # _, velocity_next = macro(population_next)
            # _, velocity_prev = macro(population_prev)
            residual += population_next - population_prev
            population_prev = population_next

        return residual / window_length
        # return population_next

    return residual

    # def residual(population, dt, dx, omega, *args):
    #     residual = 0
    #     population_prev = population
    #     for _ in range(window_length):
    #         _, population_next = stepper(population_prev, f1, bc_mask, missing_mask, omega, 0)

    #         # # _, velocity_next = macro(population_next)
    #         # # _, velocity_prev = macro(population_prev)
    #         # residual += population_next - population_prev
    #         population_prev = population_next

    #     return (population_next - population) / window_length
    #     # return population_next

    # return residual


def d2q9_to_velocity(field, macro):
    _, u = macro(field)
    # u = u[:, :-1, :]
    # rho = rho[:, :-1, :]
    return u
