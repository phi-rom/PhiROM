import equinox as eqx
import exponax as ex
import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, dataloader


class IrregularKdV2dDatasetTorch(Dataset):

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
        self.num_time_steps = num_time_steps
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.return_index = return_index
        self.max_start = self.num_time_steps - self.history_steps - self.future_steps
        self.paramed = paramed
        self.return_regular = return_regular
        with h5.File(path) as f:
            u = f["regular_u"][:, : self.num_time_steps]
            self.irregular_u = f["irregular_u"][:, : self.num_time_steps]
            dt = f["dt"][:]
            dx = f.attrs["dx"][:]
            grid = f["regular_grid"][:]
            self.irregular_coords = f["irregular_coords"][:]
            self.hyper_diff = f.attrs["hyper_diffusivity"][:]
            self.domain_extent = f.attrs["domain_extent"][:]

        if indices is not None:
            u = u[indices[0] : indices[1]]
            dt = dt[indices[0] : indices[1]]
            dx = dx[indices[0] : indices[1]]
            self.irregular_u = self.irregular_u[indices[0] : indices[1]]
            self.irregular_coords = self.irregular_coords[indices[0] : indices[1]]
        self.u = torch.tensor(u)
        self.dt = torch.tensor(dt)
        self.dx = torch.tensor(dx)
        self.grid = grid
        self.irregular_coords = torch.tensor(self.irregular_coords)
        self.irregular_u = torch.tensor(self.irregular_u)
        X, Y = grid
        X, Y = np.reshape(X, -1), np.reshape(Y, -1)
        X = np.stack([X, Y], axis=-1)
        self.X = torch.tensor(X)
        self.t = torch.tensor(
            [np.arange(0, num_time_steps) * dt[i] for i in range(u.shape[0])]
        )
        self.solver_shape = self.u.shape[2:]

    def compute_mean_std_fields(self):
        return (
            torch.mean(self.irregular_u, dim=(0, 1, 3)).numpy(),
            torch.std(self.irregular_u, dim=(0, 1, 3)).numpy(),
        )

    def compute_mean_std_coords(self):
        return (
            torch.tensor(torch.mean(self.X)).numpy(),
            torch.tensor(torch.std(self.X)).numpy(),
        )

    def compute_min_max_coords(self):
        return (
            torch.tensor(torch.min(self.X)).numpy(),
            torch.tensor(torch.max(self.X)).numpy(),
        )

    def get_coordinates(self):
        return self.X, self.t

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
            # "data": history,
            "t": self.t[traj_idx, start_idx : start_idx + self.history_steps].squeeze(
                dim=0
            ),
            "time_idx": start_idx,
            # "coords": self.X,
            "coords_irregular": coords_irregular,
            "dt": self.dt[traj_idx],
            "dx": self.dx,
            "idx": traj_idx,
            "solver_args": [],
        }
        if self.return_regular:
            batch["data"] = history
            batch["coords"] = self.X
        return batch


class IrregularKdV2dTrajectoryDatasetTroch(Dataset):

    def __init__(
        self,
        path,
        num_time_steps,
        indices=None,
        return_index=False,
        paramed=False,
        return_regular=True,
    ):
        self.num_time_steps = num_time_steps
        self.return_index = return_index
        self.paramed = paramed
        self.return_regular = return_regular
        with h5.File(path) as f:
            u = f["regular_u"][:, : self.num_time_steps]
            self.irregular_u = f["irregular_u"][:, : self.num_time_steps]
            dt = f["dt"][:]
            dx = f.attrs["dx"][:]
            grid = f["regular_grid"][:]
            self.irregular_coords = f["irregular_coords"][:]
            self.hyper_diff = f.attrs["hyper_diffusivity"][:]
            self.domain_extent = f.attrs["domain_extent"][:]

        if indices is not None:
            u = u[indices[0] : indices[1]]
            dt = dt[indices[0] : indices[1]]
            dx = dx[indices[0] : indices[1]]
            self.irregular_u = self.irregular_u[indices[0] : indices[1]]
            self.irregular_coords = self.irregular_coords[indices[0] : indices[1]]
        self.u = torch.tensor(u)
        self.dt = torch.tensor(dt)
        self.dx = torch.tensor(dx)
        self.grid = grid
        self.irregular_coords = torch.tensor(self.irregular_coords)
        self.irregular_u = torch.tensor(self.irregular_u)
        X, Y = grid
        X, Y = np.reshape(X, -1), np.reshape(Y, -1)
        X = np.stack([X, Y], axis=-1)
        self.X = torch.tensor(X)
        self.t = torch.tensor(
            [np.arange(0, num_time_steps) * dt[i] for i in range(u.shape[0])]
        )
        self.solver_shape = self.u.shape[2:]

    def compute_mean_std_fields(self):
        return (
            torch.mean(self.irregular_u, dim=(0, 1, 3)).numpy(),
            torch.std(self.irregular_u, dim=(0, 1, 3)).numpy(),
        )

    def compute_mean_std_coords(self):
        return (
            torch.tensor(torch.mean(self.X)).numpy(),
            torch.tensor(torch.std(self.X)).numpy(),
        )

    def compute_min_max_coords(self):
        return (
            torch.tensor(torch.min(self.X)).numpy(),
            torch.tensor(torch.max(self.X)).numpy(),
        )

    def get_coordinates(self):
        return self.X, self.t

    def get_trajectory(self, idx):
        return self.u[idx]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        batch = {
            "data_irregular": self.irregular_u[idx],
            # "data": history,
            "t": self.t[idx],
            "coords_irregular": self.irregular_coords[idx],
            "dt": self.dt[idx],
            "dx": self.dx,
            "idx": idx,
            "solver_args": [],
        }
        if self.return_regular:
            batch["data"] = self.u[idx]
            batch["coords"] = self.X
        return batch


def residual_builder(num_points, domain_extent, inner_dt, hyper_diff):
    print(
        "NUM_POINTS",
        num_points,
        "DOMAIN_EXTENT",
        domain_extent,
        "DT",
        inner_dt,
        "HYPER_DIFF",
        hyper_diff,
    )
    stepper = ex.stepper.KortewegDeVries(
        2,
        domain_extent,
        num_points,
        inner_dt,
        single_channel=True,
        hyper_diffusivity=hyper_diff,
        order=1,
    )

    def residual(field_1, dt, dx, *args):
        return (stepper(field_1) - field_1) / inner_dt

    return residual
