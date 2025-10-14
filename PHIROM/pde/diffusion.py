import equinox as eqx
import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
import torch
from DROM.modules.base import BaseDecoder
from torch.utils.data import Dataset


class DiffusionDatasetTorch(Dataset):

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
            u = f["u"][:, : self.num_time_steps]
            u = u[:, :, np.newaxis]  # (batch, time, field_dim, nx, ny)
            self.u = u
            self.x = f["x"][:]
            self.y = f["y"][:]
            self.dt = f["dt"][:]
            self.diffusivity = f["diffusivity"][:]
            self.dx = f.attrs["dx"][0]
            self.length = f.attrs["lengths"][:]

        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]
            self.diffusivity = self.diffusivity[indices[0] : indices[1]]
            self.dt = self.dt[indices[0] : indices[1]]

        X, Y = np.meshgrid(self.x, self.y)
        X = X.flatten()
        Y = Y.flatten()
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.dt = torch.tensor(self.dt)
        self.dx = torch.tensor(self.dx)
        self.u = torch.tensor(self.u)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.diffusivity = torch.tensor(self.diffusivity)
        self.t = torch.tensor(
            [
                np.arange(0, num_time_steps) * self.dt[i].numpy()
                for i in range(self.u.shape[0])
            ]
        )
        self.paramed = paramed
        if paramed:
            self.node_args = self.diffusivity

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
                "solver_args": [self.diffusivity[traj_idx]],
                "node_args": self.diffusivity[traj_idx],
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
                "solver_args": [self.diffusivity[traj_idx]],
            }
        )


class DiffusionTrajDatasetTorch(Dataset):

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
            u = f["u"][:, : self.num_time_steps]
            u = u[:, :, np.newaxis]  # (batch, time, field_dim, nx, ny)
            self.u = u
            self.x = f["x"][:]
            self.y = f["y"][:]
            self.dt = f["dt"][:]
            self.diffusivity = f["diffusivity"][:]
            self.dx = f.attrs["dx"][0]
            self.length = f.attrs["lengths"][:]

        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]
            self.diffusivity = self.diffusivity[indices[0] : indices[1]]
            self.dt = self.dt[indices[0] : indices[1]]

        X, Y = np.meshgrid(self.x, self.y)
        X = X.flatten()
        Y = Y.flatten()
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.dt = torch.tensor(self.dt)
        self.dx = torch.tensor(self.dx)
        self.u = torch.tensor(self.u)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.diffusivity = torch.tensor(self.diffusivity)
        self.t = torch.tensor(
            [
                np.arange(0, num_time_steps) * self.dt[i].numpy()
                for i in range(self.u.shape[0])
            ]
        )
        self.paramed = paramed
        if paramed:
            self.node_args = self.diffusivity

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
                "solver_args": [self.diffusivity[idx]],
                "node_args": self.diffusivity[idx],
            }
            if self.paramed
            else {
                "data": self.u[idx],
                "t": self.t[idx],
                "coords": self.X,
                "dt": self.dt[idx],
                "dx": self.dx,
                "idx": idx,
                "solver_args": [self.diffusivity[idx]],
            }
        )


def laplacian_fd_2d(field, dx):
    laplacian = (
        field[:, 2:, 1:-1] - 2 * field[:, 1:-1, 1:-1] + field[:, :-2, 1:-1]
    ) / dx**2 + (
        field[:, 1:-1, 2:] - 2 * field[:, 1:-1, 1:-1] + field[:, 1:-1, :-2]
    ) / dx**2
    return laplacian


def diffusion_residual(field_1, dt, dx, diffusivity, *args):
    field_1 = field_1.at[:, 0, :].set(0.0)
    field_1 = field_1.at[:, -1, :].set(0.0)
    field_1 = field_1.at[:, :, 0].set(0.0)
    field_1 = field_1.at[:, :, -1].set(0.0)
    residual = field_1.at[:, 1:-1, 1:-1].set(diffusivity * laplacian_fd_2d(field_1, dx))
    return residual


def automatic_diffusion_evolve_builder(
    diffusivity: float,
    return_prev_field: bool = True,
    evolve: bool = False,
    correction=True,
):
    """
    Returns a time-stepping function for 2D diffusion equation.
    """

    def evolve_fn(decoder: BaseDecoder, coords, latent, dt, *args):
        d2u_dx2 = eqx.filter_vmap(decoder.second_grads_x, in_axes=(0, None))(
            coords, latent
        )
        laplacian = d2u_dx2[..., 0] + d2u_dx2[..., 1]
        if correction:
            laplacian = laplacian.T.reshape((1, 42, 42))

            laplacian = laplacian.at[:, 0, :].set(0.0)
            laplacian = laplacian.at[:, -1, :].set(0.0)
            laplacian = laplacian.at[:, :, 0].set(0.0)
            laplacian = laplacian.at[:, :, -1].set(0.0)

        if not evolve:
            return laplacian * diffusivity
        else:
            field_1 = eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))(
                coords, latent
            )
            if correction:
                field_1 = field_1.T.reshape((1, 42, 42))
            field_2 = field_1 + diffusivity * dt * laplacian
            if return_prev_field:
                return field_2, field_1
            else:
                return field_2

    return evolve_fn
