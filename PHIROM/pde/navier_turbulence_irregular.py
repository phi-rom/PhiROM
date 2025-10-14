from functools import partial

import h5py as h5
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import torch
import xarray
from torch.utils.data import Dataset

from ..modules.base import BaseDecoder


class IrregularDecayingTurbulenceDatasetTorch(Dataset):

    def __init__(
        self,
        path,
        num_time_steps,
        history_steps=1,
        future_steps=0,
        indices=None,
        return_index=True,
        paramed=False,
        return_regular=True,
    ):
        if future_steps > 0:
            raise NotImplementedError("Future steps not implemented")
        self.return_regular = return_regular
        self.num_time_steps = num_time_steps
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.return_index = return_index
        self.max_start = self.num_time_steps - self.history_steps - self.future_steps
        with h5.File(path) as f:
            self.u = f["regular_u"][:, : self.num_time_steps]
            self.irregular_u = f["irregular_u"][:, : self.num_time_steps]
            self.grid = f["regular_grid"][:]
            self.irregular_coords = f["irregular_coords"][:]
            self.inner_dt = f["dt"][:]
            self.inner_steps = f.attrs["inner_steps"][0]
            self.density = f["density"][:]
            self.viscosity = f["viscosity"][:]
            self.dx = f.attrs["dx"][0]
            self.x = f["x"][:]
            self.y = f["y"][:]
            try:
                self.length = f.attrs["lengths"][:]
            except:
                self.length = (2 * np.pi, 2 * np.pi)

        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]
            self.irregular_coords = self.irregular_coords[indices[0] : indices[1]]
            self.density = self.density[indices[0] : indices[1]]
            self.viscosity = self.viscosity[indices[0] : indices[1]]
            self.inner_dt = self.inner_dt[indices[0] : indices[1]]

        self.cfd_grid = cfd.grids.Grid(
            (self.x.shape[0], self.y.shape[0]),
            domain=((0, self.length[0]), (0, self.length[1])),
        )
        X, Y = self.grid
        X = X.flatten()
        Y = Y.flatten()
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.irregular_coords = torch.tensor(self.irregular_coords)
        self.dt = torch.tensor(self.inner_dt * self.inner_steps)
        self.dx = torch.tensor(self.dx)
        self.u = torch.tensor(self.u)
        self.solver_shape = torch.tensor(self.u.shape[2:])  # u shape: N, T, F, Nx, Ny
        self.irregular_u = torch.tensor(self.irregular_u)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.density = torch.tensor(self.density)
        self.viscosity = torch.tensor(self.viscosity)
        self.inner_dt = torch.tensor(self.inner_dt)
        self.t = torch.tensor(
            [
                np.arange(0, num_time_steps) * self.dt[i].numpy()
                for i in range(self.u.shape[0])
            ]
        )
        self.paramed = paramed
        if paramed:
            self.node_args = self.viscosity

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

        batch = {
            "data_irregular": history_irregular,
            "t": self.t[traj_idx, start_idx : start_idx + self.history_steps].squeeze(
                dim=0
            ),
            "time_idx": start_idx,
            "coords_irregular": coords_irregular,
            "dt": self.dt[traj_idx],
            "dx": self.dx,
            "idx": traj_idx,
            "solver_args": [
                self.density[traj_idx],
                self.viscosity[traj_idx],
                self.inner_dt[traj_idx],
            ],
            "solver_shape": self.solver_shape,
        }
        if self.return_regular:
            batch["data"] = history
            batch["coords"] = self.X
        if self.paramed:
            batch["node_args"] = self.viscosity[traj_idx]
        return batch


class IrregularDecayingTurbulenceTrajDatasetTorch(Dataset):

    def __init__(
        self,
        path,
        num_time_steps,
        indices=None,
        return_index=True,
        paramed=False,
        return_regular=True,
    ):
        self.num_time_steps = num_time_steps
        self.return_regular = return_regular
        with h5.File(path) as f:
            self.u = f["regular_u"][:, : self.num_time_steps]
            self.irregular_u = f["irregular_u"][:, : self.num_time_steps]
            self.grid = f["regular_grid"][:]
            self.irregular_coords = f["irregular_coords"][:]
            self.inner_dt = f["dt"][:]
            self.inner_steps = f.attrs["inner_steps"][0]
            self.density = f["density"][:]
            self.viscosity = f["viscosity"][:]
            self.dx = f.attrs["dx"][0]
            self.x = f["x"][:]
            self.y = f["y"][:]
            try:
                self.length = f.attrs["lengths"][:]
            except:
                self.length = (2 * np.pi, 2 * np.pi)

        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]
            self.irregular_coords = self.irregular_coords[indices[0] : indices[1]]
            self.density = self.density[indices[0] : indices[1]]
            self.viscosity = self.viscosity[indices[0] : indices[1]]
            self.inner_dt = self.inner_dt[indices[0] : indices[1]]

        self.cfd_grid = cfd.grids.Grid(
            (self.x.shape[0], self.y.shape[0]),
            domain=((0, self.length[0]), (0, self.length[1])),
        )
        X, Y = self.grid
        X = X.flatten()
        Y = Y.flatten()
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.irregular_coords = torch.tensor(self.irregular_coords)
        self.dt = torch.tensor(self.inner_dt * self.inner_steps)
        self.dx = torch.tensor(self.dx)
        self.u = torch.tensor(self.u)
        self.solver_shape = torch.tensor(self.u.shape[2:])  # u shape: N, T, F, Nx, Ny
        self.irregular_u = torch.tensor(self.irregular_u)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.density = torch.tensor(self.density)
        self.viscosity = torch.tensor(self.viscosity)
        self.inner_dt = torch.tensor(self.inner_dt)
        self.t = torch.tensor(
            [
                np.arange(0, num_time_steps) * self.dt[i].numpy()
                for i in range(self.u.shape[0])
            ]
        )
        self.paramed = paramed
        if paramed:
            self.node_args = self.viscosity

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
        batch = {
            "data_irregular": self.irregular_u[idx],
            "t": self.t[idx],
            "coords_irregular": self.irregular_coords[idx],
            "dt": self.dt[idx],
            "dx": self.dx,
            "idx": idx,
            "solver_args": [self.density[idx], self.viscosity[idx], self.inner_dt[idx]],
            "solver_shape": self.solver_shape,
        }
        if self.return_regular:
            batch["data"] = self.u[idx]
            batch["coords"] = self.X
        if self.paramed:
            batch["node_args"] = self.viscosity[idx]
        return batch


def cfd_evolve_builder(grid, inner_steps):
    repeater = partial(cfd.funcutils.repeated, steps=inner_steps)
    solver = partial(cfd.equations.semi_implicit_navier_stokes, grid=grid)

    def evolve(field_1, dt, dx, density, viscosity, inner_dt, *args):
        v0, v1 = field_1
        v0, v1 = grid.stagger((v0, v1))
        v = (
            cfd.grids.GridVariable(
                array=v0, bc=cfd.boundaries.periodic_boundary_conditions(2)
            ),
            cfd.grids.GridVariable(
                array=v1, bc=cfd.boundaries.periodic_boundary_conditions(2)
            ),
        )
        v = repeater(solver(dt=inner_dt, density=density, viscosity=viscosity))(v)
        return jnp.stack([v[0].data, v[1].data], axis=0)

    return evolve


def cfd_residual_builder(grid):
    solver = partial(cfd.equations.semi_implicit_navier_stokes, grid=grid)

    def residual(field_1, dt, dx, density, viscosity, inner_dt, *args):
        v0, v1 = field_1
        v0, v1 = grid.stagger((v0, v1))
        v = (
            cfd.grids.GridVariable(
                array=v0, bc=cfd.boundaries.periodic_boundary_conditions(2)
            ),
            cfd.grids.GridVariable(
                array=v1, bc=cfd.boundaries.periodic_boundary_conditions(2)
            ),
        )
        v = solver(dt=inner_dt, density=density, viscosity=viscosity)(v)
        return (jnp.stack([v[0].data, v[1].data], axis=0) - field_1) / inner_dt

    return residual
