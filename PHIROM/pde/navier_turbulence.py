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


class DecayingTurbulenceDatasetTorch(Dataset):

    def __init__(
        self,
        path,
        num_time_steps,
        history_steps=1,
        future_steps=0,
        indices=None,
        return_index=True,
        paramed=False,
    ):
        if future_steps > 0:
            raise NotImplementedError("Future steps not implemented")
        self.num_time_steps = num_time_steps
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.return_index = return_index
        self.max_start = self.num_time_steps - self.history_steps - self.future_steps
        with h5.File(path) as f:
            self.u = f["u"][:, : self.num_time_steps]
            self.x = f["x"][:]
            self.y = f["y"][:]
            self.inner_dt = f["dt"][:]
            self.inner_steps = f.attrs["inner_steps"][0]
            self.density = f["density"][:]
            self.viscosity = f["viscosity"][:]
            self.dx = f.attrs["dx"][0]
            try:
                self.length = f.attrs["lengths"][:]
            except:
                self.length = (2 * np.pi, 2 * np.pi)

        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]
            self.density = self.density[indices[0] : indices[1]]
            self.viscosity = self.viscosity[indices[0] : indices[1]]
            self.inner_dt = self.inner_dt[indices[0] : indices[1]]

        self.grid = cfd.grids.Grid(
            (self.x.shape[0], self.y.shape[0]),
            domain=((0, self.length[0]), (0, self.length[1])),
        )
        X, Y = self.grid.mesh()
        X = X.flatten()
        Y = Y.flatten()
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.dt = torch.tensor(self.inner_dt * self.inner_steps)
        self.dx = torch.tensor(self.dx)
        self.u = torch.tensor(self.u)
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
            torch.mean(self.u, dim=(0, 1, 3, 4)).numpy(),
            torch.std(self.u, dim=(0, 1, 3, 4)).numpy(),
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
        if self.history_steps == 1:
            history = history.squeeze(dim=0)
        return (
            {
                "data": history,
                "t": self.t[
                    traj_idx, start_idx : start_idx + self.history_steps
                ].squeeze(dim=0),
                "time_idx": start_idx,
                "coords": self.X,
                "dt": self.dt[traj_idx],
                "dx": self.dx,
                "idx": traj_idx,
                "solver_args": [
                    self.density[traj_idx],
                    self.viscosity[traj_idx],
                    self.inner_dt[traj_idx],
                ],
                "node_args": self.viscosity[traj_idx],
            }
            if self.paramed
            else {
                "data": history,
                "t": self.t[
                    traj_idx, start_idx : start_idx + self.history_steps
                ].squeeze(dim=0),
                "time_idx": start_idx,
                "coords": self.X,
                "dt": self.dt[traj_idx],
                "dx": self.dx,
                "idx": traj_idx,
                "solver_args": [
                    self.density[traj_idx],
                    self.viscosity[traj_idx],
                    self.inner_dt[traj_idx],
                ],
            }
        )


class DecayingTurbulenceTrajDatasetTorch(Dataset):

    def __init__(
        self, path, num_time_steps, indices=None, return_index=True, paramed=False
    ):
        self.num_time_steps = num_time_steps
        with h5.File(path) as f:
            self.u = f["u"][:, : self.num_time_steps]
            self.x = f["x"][:]
            self.y = f["y"][:]
            self.inner_dt = f["dt"][:]
            self.inner_steps = f.attrs["inner_steps"][0]
            self.density = f["density"][:]
            self.viscosity = f["viscosity"][:]
            self.dx = f.attrs["dx"][0]
            try:
                self.length = f.attrs["lengths"][:]
            except:
                self.length = (2 * np.pi, 2 * np.pi)

        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]
            self.density = self.density[indices[0] : indices[1]]
            self.viscosity = self.viscosity[indices[0] : indices[1]]
            self.inner_dt = self.inner_dt[indices[0] : indices[1]]

        self.grid = cfd.grids.Grid(
            (self.x.shape[0], self.y.shape[0]),
            domain=((0, self.length[0]), (0, self.length[1])),
        )
        X, Y = self.grid.mesh()
        X = X.flatten()
        Y = Y.flatten()
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.dt = torch.tensor(self.inner_dt * self.inner_steps)
        self.dx = torch.tensor(self.dx)
        self.u = torch.tensor(self.u)
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
            torch.mean(self.u, dim=(0, 1, 3, 4)).numpy(),
            torch.std(self.u, dim=(0, 1, 3, 4)).numpy(),
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
                "data": self.u[idx],
                "t": self.t[idx],
                "coords": self.X,
                "dt": self.dt[idx],
                "dx": self.dx,
                "idx": idx,
                "solver_args": [
                    self.density[idx],
                    self.viscosity[idx],
                    self.inner_dt[idx],
                ],
                "node_args": self.viscosity[idx],
            }
            if self.paramed
            else {
                "data": self.u[idx],
                "t": self.t[idx],
                "coords": self.X,
                "dt": self.dt[idx],
                "dx": self.dx,
                "idx": idx,
                "solver_args": [
                    self.density[idx],
                    self.viscosity[idx],
                    self.inner_dt[idx],
                ],
            }
        )


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
    # repeater = partial(cfd.funcutils.repeated, steps=inner_steps)
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
