import collections
import itertools
from typing import Sequence, Tuple, Union

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import prefetch_to_device
from jax import random
from jax.tree_util import tree_map
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader, Dataset, default_collate

from ..modules.models import CROMOnline


def generate_boundary_points(boundaries: Array, num_points_per_axis=3):
    """
    Generate boundary points for a given set of boundaries.
    Args:
        boundaries: Array, boundaries for each dimension
        num_points_per_axis: int, number of points per axis
    Returns:
        Array, boundary points
    """
    dimensions = boundaries.shape[0]
    axis_values = []

    for dim in range(dimensions):
        min_val, max_val = boundaries[dim]
        axis_values.append(jnp.linspace(min_val, max_val, num_points_per_axis))

    grid = jnp.array(list(itertools.product(*axis_values)))

    boundary_points = []
    for point in grid:
        if jnp.any(
            jnp.logical_or(point == boundaries[:, 0], point == boundaries[:, 1])
        ):
            boundary_points.append(point)

    return jnp.array(boundary_points)


class TrajectoryDataset(Dataset):
    """
    Load samples of an PDE Dataset, get items according to PDE.
    """

    def __init__(
        self,
        path: str,
        mode: str,
        nt: int,
        nx: int,
        dtype=np.float64,
        load_all: bool = False,
    ):
        self.nt = nt
        self.nx = nx
        self.mode = mode
        self.dtype = dtype
        f = h5py.File(path, "r")
        self.data = f[self.mode]
        self.dataset = f"pde_{nt}-{nx}"
        if load_all:
            data = {
                self.dataset: self.data[self.dataset][:],
                "x": self.data["x"][:],
                "t": self.data["t"][:],
            }
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            np.ndarray: data trajectory used for training/validation/testing
            np.ndarray: dx
            np.ndarray: dt
        """
        u = self.data[self.dataset][idx]
        x = self.data["x"][idx]
        t = self.data["t"][idx]
        X, T = np.meshgrid(x, t)
        return u, T, X


class TimeWindowDatasetNPZ(Dataset):
    """
    Loads an NPZ PDE dataset and samples a window of history and future time points from each trajectory.

    Args:
        path: str, path to the HDF5 file.
        mode: str, the mode to load the dataset in. Can be 'train', 'val', or 'test'.
        nt: int, the number of time points in the trajectory.
        nx: int, the number of spatial points in the trajectory.
        time_history: int, the number of time points in the history.
        time_future: int, the number of time points in the future.

    """

    def __init__(
        self,
        path: str,
        nt: int,
        nx: int,
        history_steps: int = 1,
        future_steps: int = 1,
        indices: Tuple[int, int] = None,
        return_index: bool = False,
    ):
        self.nt = nt
        self.nx = nx
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.return_index = return_index
        self.max_start_time = self.nt - self.history_steps - self.future_steps
        with np.load(path) as f:
            self.x = f["x"][0, :]
            self.t = f["t"][:, 0]
            self.u = f["u"]
        if indices is not None:
            self.u = self.u[indices[0] : indices[1]]

    def compute_mean_std_fields(self):
        mean = np.mean(self.u, axis=(0, 1))
        std = np.std(self.u, axis=(0, 1))
        return mean, std

    def compute_mean_std_coords(self):
        return (np.array(np.mean(self.x))), np.array(np.std(self.x))

    def compute_min_max_coords(self):
        return np.array(np.min(self.x)), np.array(np.max(self.x))

    def get_coordinates(self):
        return self.x, self.t

    def get_trajectory(self, idx: Union[int, np.ndarray]):
        return self.u[idx]

    def __len__(self):
        return (self.max_start_time + 1) * self.u.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            torch.Tensor: data trajectory used for training/validation/testing
            torch.Tensor: dt
            torch.Tensor: dx
        """
        # Get the trajectory index and the starting time index
        traj_idx = idx // (self.max_start_time + 1)
        start_time = idx % (self.max_start_time + 1)

        history = self.u[traj_idx, start_time : start_time + self.history_steps]
        future = self.u[
            traj_idx,
            start_time
            + self.history_steps : start_time
            + self.history_steps
            + self.future_steps,
        ]

        dx = self.x[1] - self.x[0]
        dt = self.t[1] - self.t[0]
        if self.return_index:
            return (
                history,
                future,
                self.t,
                self.x.reshape(-1, 1),
                dt,
                dx,
                self.u[traj_idx],
                traj_idx,
                start_time,
            )
        return history, future, self.t, self.x.reshape(-1, 1), dt, dx, self.u[traj_idx]


class TimeWindowDataset(Dataset):
    """
    Loads an HDF5 PDE dataset and samples a window of history and future time points from each trajectory.

    Args:
        path: str, path to the HDF5 file.
        mode: str, the mode to load the dataset in. Can be 'train', 'val', or 'test'.
        nt: int, the number of time points in the trajectory.
        nx: int, the number of spatial points in the trajectory.
        time_history: int, the number of time points in the history.
        time_future: int, the number of time points in the future.

    """

    def __init__(
        self,
        path: str,
        nt: int,
        nx: int,
        mode: str,
        history_steps: int = 1,
        future_steps: int = 1,
        load_all: bool = False,
    ):
        self.nt = nt
        self.nx = nx
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.mode = mode
        self.max_start_time = self.nt - self.history_steps - self.future_steps
        f = h5py.File(path, "r")
        self.data = f[self.mode]
        self.dataset = f"pde_{nt}-{nx}"
        self.load_all = load_all
        if load_all:
            data = {
                self.dataset: self.data[self.dataset][:],
                "x": self.data["x"][:],
                "t": self.data["t"][:],
            }
            f.close()
            self.data = data

    def __len__(self):
        return self.max_start_time * self.data[self.dataset].shape[0]

    def __getitem__(self, idx: int):
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            torch.Tensor: data trajectory used for training/validation/testing
            torch.Tensor: dt
            torch.Tensor: dx
        """
        # Get the trajectory index and the starting time index
        traj_idx = idx // self.max_start_time
        start_time = idx % self.max_start_time

        history = self.data[self.dataset][
            traj_idx, start_time : start_time + self.history_steps
        ]
        future = self.data[self.dataset][
            traj_idx,
            start_time
            + self.history_steps : start_time
            + self.history_steps
            + self.future_steps,
        ]

        dx = self.data["x"][traj_idx, 1] - self.data["x"][traj_idx, 0]
        dt = self.data["t"][traj_idx, 1] - self.data["t"][traj_idx, 0]

        return history, future, dt, dx


def torch_iterator_prefetch(iterator, size: int):
    """
    Casts and prefetches a torch DataLoader iterator to the device.

    Args:
        iterator: iterator, torch DataLoader iterator to prefetch
        size: int, size of the prefetch buffer

    Returns:
        iterator: iterator with prefetch buffer
    """
    map_fn = lambda x: tree_map(lambda y: y.numpy()[np.newaxis, ...], x)
    iterator = map(map_fn, iterator)
    return prefetch_to_device(iterator, size)


def iterator_sharded_prefetch(iterator, size: int, sharding):

    queue = collections.deque()
    map_numpy = lambda x: tree_map(lambda y: y.numpy(), x)

    def _prefetch(xs):
        return eqx.filter_shard(xs, sharding)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, map_numpy(data)))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)


def collate_helper(x):
    return np.asarray(x)[np.newaxis, ...]


def numpy_collate(batch):
    return tree_map(collate_helper, default_collate(batch))


def jax_collate(batch):
    """
    Collate function that converts a batch of numpy arrays to a batch of jax arrays. Use with torch DataLoader.
    """
    if isinstance(batch[0], np.ndarray):
        return jnp.stack(batch)[jnp.newaxis, ...]
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [jax_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)[jnp.newaxis, ...]


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class JaxLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=jax_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
