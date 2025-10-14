import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=16)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--activation", type=str, default="sin")
parser.add_argument("--epochs", type=int, default=20000)
parser.add_argument("--dataset", type=str, default="diffusion_42x42")
parser.add_argument("--prefix", type=str, default="")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--loss", type=str, default="nmse")
parser.add_argument("--init_lr", type=float, default=1e-3)
parser.add_argument("--final_lr", type=float, default=2e-5)
parser.add_argument("--decay_steps", type=int, default=1000)
parser.add_argument("--decay_rate", type=float, default=0.985)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument(
    "--autodecoder",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use autodecoder",
)
parser.add_argument("--max_step", type=int, default=25)
parser.add_argument(
    "--decoder_arch", type=str, default="hyper", choices=["mlp", "hyper"]
)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)


args = parser.parse_args()
latent_dim = args.latent_dim
width_scale = args.width
activation = args.activation
epochs = args.epochs
dataset_name = args.dataset
prefix = args.prefix
seed = args.seed
loss = args.loss
init_lr = args.init_lr
final_lr = args.final_lr
decay_steps = args.decay_steps
decay_rate = args.decay_rate
num_samples = args.num_samples
autodecoder = args.autodecoder
max_step = args.max_step
arch = args.decoder_arch
normalize = args.normalize

from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax as optx
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader

from PHIROM.modules.models import CROMOffline, DecoderArchEnum
from PHIROM.pde.diffusion import *
from PHIROM.training.baseline import CROMAutoDecoderTrainer, CROMOfflineTrainer
from PHIROM.training.callbacks import CheckpointCallback
from PHIROM.utils.serial import load_model, make_CROMOffline, save_model

name = f"{dataset_name}_seed={seed}_loss={loss}_nt={max_step}_n={num_samples}_ac={activation}_ld={latent_dim}_ws={width_scale}_ep={epochs}"
batch_size = 1250
if prefix:
    name = prefix + "_" + name

dataset_name = "diffusion_42x42"

crop_bnd = False

if autodecoder:
    dataset_train = DiffusionDatasetTorch(
        "data/diffusion_42x42.h5", max_step, 1, 0, (0, num_samples), True, False
    )
else:
    raise ValueError("Not supported")
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

MEAN, STD = dataset_train.compute_mean_std_fields()
nx = dataset_train.x.shape[0]
ny = dataset_train.y.shape[0]

hyperparams = {
    "latent_dim": latent_dim,
    "num_sensors": nx * ny,
    "field_dim": 1,
    "spatial_dim": 2,
    "mean_field": MEAN if normalize else None,
    "std_field": STD if normalize else None,
    "activation": activation,
}

if arch == "mlp":
    arch = DecoderArchEnum.MLP
    hyperparams["width_scale"] = width_scale
    hyperparams["decoder_arch"] = arch
elif arch == "hyper":
    arch = DecoderArchEnum.HYPER
    hyperparams["decoder_arch"] = arch
    hyperparams["width"] = width_scale * 1
    hyperparams["n_layers"] = 3
    hyperparams["input_scale"] = 1.0

if activation in ["softplus", "elu", "swish", "tanh"]:
    mean_x, std_x = dataset_train.compute_mean_std_coords()
    hyperparams["mean_x"] = mean_x
    hyperparams["std_x"] = std_x
elif activation == "sin":
    min_x, max_x = dataset_train.compute_min_max_coords()
    hyperparams["min_x"] = min_x
    hyperparams["max_x"] = max_x

key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

model = CROMOffline(**hyperparams, key=subkey)
scheduler = optx.schedules.exponential_decay(
    init_lr, decay_steps, decay_rate, end_value=final_lr, staircase=True
)
optimizer = optx.adam(scheduler)

if autodecoder:
    method = "AD"
else:
    method = "AE"

evolve = "CROM"

path = os.path.join(dataset_name, method, arch, evolve)
path_experiment = os.path.join("CROM_experiments", path, name)
Path(path_experiment).mkdir(parents=True, exist_ok=True)
path_checkpoint = os.path.join(path_experiment, "checkpoints")
callbacks = [CheckpointCallback(path_checkpoint, name, hyperparams, True, 500)]


key, subkey = jax.random.split(key)

if autodecoder:
    scheduler_latent = optx.schedules.exponential_decay(
        init_lr, decay_steps, decay_rate, end_value=final_lr, staircase=False
    )
    optimimzer_latent = optx.adam(scheduler_latent)
    trainer = CROMAutoDecoderTrainer(
        model,
        optimizer,
        optimimzer_latent,
        evolve_time=False,
        loss=loss,
        callbacks=callbacks,
        key=subkey,
        num_trajectories=num_samples,
        num_time_steps=max_step,
        latent_dim=latent_dim,
    )
else:
    raise ValueError("Not implemented")
model, opt_state, history = trainer.fit(loader_train, epochs=epochs, warm_start=True)

save_model(os.path.join(path_experiment, "model.eqx"), hyperparams, model, None)
history["loss_reconstruction"] = np.array(history["loss_reconstruction"])
history["loss_time_stepping"] = np.array(history["loss_time_stepping"])
np.savez(os.path.join(path_experiment, "history.npz"), **history)

if autodecoder:
    l = np.array(trainer.latent_memory)
    np.save(os.path.join(path_experiment, "latent_memory.npy"), l)
