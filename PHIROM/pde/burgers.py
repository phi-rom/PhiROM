import equinox as eqx
import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, dataloader


class BurgersDatasetTorch(Dataset):

    def __init__(
        self,
        path,
        num_time_steps,
        history_steps=1,
        future_steps=0,
        indices=None,
        return_index=True,
        crop_boundary: bool = False,
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
            dt = f["dt"][:]
            dx = f["dx"][:]
            mu = f["mu"][:]
            X = f["x"][:].T

        if indices is not None:
            u = u[indices[0] : indices[1]]
            mu = mu[indices[0] : indices[1]]
            dt = dt[indices[0] : indices[1]]
            dx = dx[indices[0] : indices[1]]
        self.u = torch.tensor(u)
        self.dt = torch.tensor(dt)
        self.dx = torch.tensor(dx)
        self.mu = torch.tensor(mu)
        self.X = torch.tensor(X[:, jnp.newaxis])
        self.t = torch.tensor(
            [np.arange(0, num_time_steps) * dt[i] for i in range(u.shape[0])]
        )
        self.node_args = self.mu

    def compute_mean_std_fields(self):
        return (
            torch.mean(self.u, dim=(0, 1, 3)).numpy(),
            torch.std(self.u, dim=(0, 1, 3)).numpy(),
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
        if self.history_steps == 1:
            history = history.squeeze(dim=0)
        return {
            "data": history,
            "t": self.t[traj_idx, start_idx : start_idx + self.history_steps].squeeze(
                dim=0
            ),
            "time_idx": start_idx,
            "coords": self.X,
            "dt": self.dt[traj_idx],
            "dx": self.dx[traj_idx],
            "idx": traj_idx,
            "solver_args": [self.X, self.mu[traj_idx]],
            "node_args": self.mu[traj_idx],
        }


class BurgersTrajectoryDatasetTroch(Dataset):

    def __init__(self, path, num_time_steps, indices=None, return_index=False):
        self.num_time_steps = num_time_steps
        self.return_index = return_index
        with h5.File(path) as f:
            u = f["u"][:, : self.num_time_steps]
            dt = f["dt"][:]
            dx = f["dx"][:]
            mu = f["mu"][:]
            X = f["x"][:]

        if indices is not None:
            u = u[indices[0] : indices[1]]
            mu = mu[indices[0] : indices[1]]
            dt = dt[indices[0] : indices[1]]
            dx = dx[indices[0] : indices[1]]
        self.u = torch.tensor(u)
        self.dt = torch.tensor(dt)
        self.dx = torch.tensor(dx)
        self.mu = torch.tensor(mu)
        self.X = torch.tensor(X[:, np.newaxis])
        self.t = torch.tensor(
            [np.arange(0, num_time_steps) * dt[i] for i in range(u.shape[0])]
        )
        self.node_args = self.mu

    def compute_mean_std_fields(self):
        return (
            torch.mean(self.u, dim=(0, 1, 3)).numpy(),
            torch.std(self.u, dim=(0, 1, 3)).numpy(),
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
        return {
            "data": self.u[idx],
            "t": self.t[idx],
            "coords": self.X,
            "dt": self.dt[idx],
            "dx": self.dx[idx],
            "idx": idx,
            "solver_args": [self.X, self.mu[idx]],
            "node_args": self.mu[idx],
        }


def evolve_burgers(field_1, dt, dx, x, mu, *args):
    print(mu)
    q = field_1.reshape((1, -1))
    q_pad = jnp.pad(q, ((0, 0), (1, 1)), "edge")
    q_pad = q_pad.at[:, 1:-1].set(q)
    q_ = 1.0 * (
        -(0.5 * (q_pad[:, 1:-1]) ** 2 - 0.5 * (q_pad[:, 0:-2]) ** 2) / dx
        + 0.02 * jnp.exp(mu * x.reshape(1, -1))
    )
    q_ = q_.at[:, 0].set(0.0)
    return field_1 + dt * q_.reshape(field_1.shape)


def residual_burgers(field_1, dt, dx, x, mu, *args):
    q = field_1.reshape((1, -1))
    q_pad = jnp.pad(q, ((0, 0), (1, 1)), "edge")
    q_pad = q_pad.at[:, 1:-1].set(q)
    q_ = 1.0 * (
        -(0.5 * (q_pad[:, 1:-1]) ** 2 - 0.5 * (q_pad[:, 0:-2]) ** 2) / dx
        + 0.02 * jnp.exp(mu * x.reshape(1, -1))
    )
    q_ = q_.at[:, 0].set(0.0)
    return q_.reshape(field_1.shape)


def residual_burgers_ad(decoder, coords, latent, dt, x, mu, *args):
    def u_2(coord, latent):
        return decoder.call_coords_latent(coord, latent) ** 2

    first_grads = eqx.filter_vmap(jax.jacfwd(u_2, argnums=0), in_axes=(0, None))(
        coords, latent
    )[:, :, 0]
    print("Burgers PINN called")
    return (-0.5 * first_grads + 0.02 * jnp.exp(mu * coords.reshape(-1, 1))).T


def evolve_ad_builder(return_f1=False):
    def evolve_burgers_ad(decoder, coords, latent, dt, x, mu, *args):
        f1 = eqx.filter_vmap(decoder.call_coords_latent, in_axes=(0, None))(
            coords, latent
        ).T
        dfdt = residual_burgers_ad(decoder, coords, latent, dt, x, mu)
        print(f1.shape)
        print(dfdt.shape)
        f2 = f1 + dt * dfdt
        print(f2.shape)
        if return_f1:
            return f2, f1
        else:
            return f2

    return evolve_burgers_ad
