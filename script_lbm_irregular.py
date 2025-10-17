import argparse
import os

os.environ["EQX_ON_ERROR"] = "nan"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


from datetime import datetime
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax as optx
import xlb
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader
from xlb.operator.macroscopic import SecondMoment

from PHIROM.modules.models import DecoderArchEnum, NodeROM
from PHIROM.pde.lbm_irreg import *
from PHIROM.pde.data_utils import JaxLoader, NumpyLoader
from PHIROM.training.baseline import IrregularDINoTrainer
from PHIROM.training.callbacks import (
    CheckpointCallback,
    NODEUnrollingEvaluationCallback,
    NODEUnrollingEvaluationCallbackXLB,
)
from PHIROM.training.train import IrregularPhiROMTrainer, NodeTrainingModeEnum
from PHIROM.utils.experiment_utils import *
from PHIROM.utils.serial import load_model, make_CROMOffline, save_model

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=4)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--activation", type=str, default="sin")
parser.add_argument("--node_activation", type=str, default="swish")
parser.add_argument("--node_width", type=int, default=16)
parser.add_argument("--epochs", type=int, default=35000)
parser.add_argument("--dataset", type=str, default="cylinder_population")
parser.add_argument("--prefix", type=str, default="")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--loss", type=str, default="nmse")
# parser.add_argument("--evolve_mode", type=str, default="label_label", choices=["label_label", "pred_pred", "pred_label"])
parser.add_argument(
    "--ode_solver", type=str, default="bosh3", choices=["bosh3", "dopri5", "euler"]
)
parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--max_ode_steps", type=int, default=None)
parser.add_argument(
    "--dino",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Train Data-Driven only (DINo)",
)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--gamma_decay_rate", type=float, default=0.99)
parser.add_argument(
    "--gamma_epochs", type=int, default=10, help="Scheduling gamma decay epochs"
)
parser.add_argument("--final_gamma", type=float, default=0.0)
parser.add_argument("--final_lr", type=float, default=1e-6)
parser.add_argument("--decay_steps", type=int, default=50)
parser.add_argument("--decay_rate", type=float, default=0.985)
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument(
    "--autodecoder",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use autodecoder",
)
parser.add_argument("--max_step", type=int, default=100)
parser.add_argument("--evolve_start", type=int, default=0)
parser.add_argument(
    "--decoder_arch", type=str, default="hyper", choices=["mlp", "hyper"]
)
parser.add_argument(
    "--node_arch",
    type=str,
    default="hyper_concat",
    choices=[
        "mlp",
        "hyper",
        "hyperV2",
        "hyper_add_v1",
        "hyper_add_v2",
        "hyper_multip_v1",
        "hyper_multip_v2",
        "hyper_add_v3",
        "hyper_bias",
        "hyper_add_v4",
        "hyper_multip_v4",
        "hyper_concat",
    ],
)
parser.add_argument(
    "--node_training_mode",
    type=str,
    default=NodeTrainingModeEnum.JACOBIAN_INVERSE,
    choices=[
        NodeTrainingModeEnum.JACOBIAN_PSI,
        NodeTrainingModeEnum.JACOBIAN_INVERSE,
        NodeTrainingModeEnum.ZERO,
        NodeTrainingModeEnum.LABELS,
    ],
)
parser.add_argument("--batch_size", type=int, default=300)
parser.add_argument("--learning_rate_decoder", type=float, default=1e-3)
parser.add_argument("--learning_rate_node", type=float, default=-1)
parser.add_argument("--learning_rate_latent", type=float, default=-1)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--pinn", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--solver_steps", type=int, default=1)
parser.add_argument("--loss_lambda", type=float, default=0.8)

args = parser.parse_args()
latent_dim = args.latent_dim
width = args.width
activation = args.activation
node_activation = args.node_activation
epochs = args.epochs
dataset_name = args.dataset
prefix = args.prefix
seed = args.seed
loss = args.loss
DINO = args.dino
final_lr = args.final_lr
decay_steps = args.decay_steps
decay_rate = args.decay_rate
gamma = args.gamma
gamma_decay_rate = args.gamma_decay_rate
gamma_epochs = args.gamma_epochs
final_gamma = args.final_gamma
num_samples = args.num_samples
autodecoder = args.autodecoder
max_step = args.max_step
evolve_start = args.evolve_start
max_split = 0
split_start = 0
arch = args.decoder_arch
ode_solver = args.ode_solver
adaptive = args.adaptive
max_ode_steps = args.max_ode_steps
node_activation = args.node_activation
node_width = args.node_width
node_arch = args.node_arch
batch_size = args.batch_size
learning_rate_decoder = args.learning_rate_decoder
learning_rate_node = args.learning_rate_node
learning_rate_latent = args.learning_rate_latent
normalize = args.normalize
solver_steps = args.solver_steps
loss_lambda = args.loss_lambda

node_training_mode = args.node_training_mode

if "ins=5" in dataset_name:
    start_time = 0
else:
    start_time = 300
print("start_time:", start_time)
paramed = True
path = f"./data/{dataset_name}.h5"
test_dataset_path = f"./data/{dataset_name}_test.h5"

if paramed:
    param_dim = 1
else:
    param_dim = 0
    if node_arch == "hyper" or node_arch == "hyperV2":
        raise ValueError("Hyper node not supported for non-parametric datasets")

if autodecoder:
    if not DINO:
        dataset_train = IrregularLBMDatasetTorch(
            path, start_time, max_step, indices=(0, num_samples), paramed=paramed
        )
    else:
        dataset_train = IrregularLBMTrajDatasetTorch(
            path, start_time, max_step, indices=(0, num_samples), paramed=paramed
        )
    dataset_validation = IrregularLBMTrajDatasetTorch(
        test_dataset_path, start_time, max_step + 50, paramed=paramed, indices=(0, 8)
    )
    subdataset_train = IrregularLBMTrajDatasetTorch(
        path, start_time, max_step + 50, paramed=paramed, indices=[0, 10, 20, 30]
    )
else:
    raise NotImplementedError("AE Not implemented")

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
decay_steps = len(loader_train) * decay_steps

print(
    f"Training on {dataset_name} dataset - num batches: {len(loader_train)} - num samples: {num_samples} - max step: {max_step}"
)
print(f"Decay every {decay_steps} steps with rate {decay_rate}")

if paramed:
    MEAN_NODE_ARGS = dataset_train.node_args.mean(axis=0).numpy()
    STD_NODE_ARGS = dataset_train.node_args.std(axis=0).numpy()
else:
    MEAN_NODE_ARGS = None
    STD_NODE_ARGS = None

path, name = get_path_and_name(args)
if not DINO:
    name = "SOLVER_STEPS_" + str(solver_steps) + "_" + name

MEAN, STD = dataset_train.compute_mean_std_fields()
nx = dataset_train.x.shape[0]
ny = dataset_train.y.shape[0]

hyperparams = {
    "latent_dim": latent_dim,
    "num_sensors": nx * ny,
    "field_dim": 9 if not DINO else 2,
    "spatial_dim": 2,
    "mean_field": MEAN if normalize else None,
    "std_field": STD if normalize else None,
    "activation": activation,
    "node_kwargs": {
        "node_arch": node_arch,
        "activation": node_activation,
        "depth": 3,
        "width": node_width,
        "param_size": param_dim,
        "solver": ode_solver,
        "adaptive": adaptive,
        "max_steps": max_ode_steps,
        "mean_params": MEAN_NODE_ARGS,
        "std_params": STD_NODE_ARGS,
        "initializer": None,
    },
}

if arch == "mlp":
    arch = DecoderArchEnum.MLP
    hyperparams["width_scale"] = width
    hyperparams["decoder_arch"] = arch
    hyperparams["n_layers"] = 6
    hyperparams["final_activation"] = None
elif arch == "hyper":
    arch = DecoderArchEnum.HYPER
    hyperparams["decoder_arch"] = arch
    hyperparams["width"] = width
    hyperparams["n_layers"] = 4
    hyperparams["input_scale"] = 1.0
    hyperparams["std_coords"] = True

if activation in ["elu", "softplus", "tanh"]:
    mean_x, std_x = dataset_train.compute_mean_std_coords()
    hyperparams["mean_x"] = mean_x
    hyperparams["std_x"] = std_x
elif activation == "sin":
    min_x, max_x = dataset_train.compute_min_max_coords()
    print(min_x, max_x)
    hyperparams["min_x"] = min_x
    hyperparams["max_x"] = max_x

key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

model, model_state = eqx.nn.make_with_state(NodeROM)(**hyperparams, key=subkey)

path_experiment = os.path.join("NODE_experiments", path, name)
path_checkpoint = os.path.join(path_experiment, "checkpoints")
Path(path_experiment).mkdir(parents=True, exist_ok=True)

key, subkey = jax.random.split(key)


print(hyperparams)


if not DINO:
    backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, compute_backend=backend
    )
    xlb.init(
        velocity_set=velocity_set,
        default_backend=backend,
        default_precision_policy=precision_policy,
    )
    u_max = dataset_train.u_max
    grid_shape = dataset_train.x.shape[0], dataset_train.y.shape[0]
    grid = grid_factory(grid_shape, compute_backend=backend)
    cylinder_diameter = dataset_train.cylinder_diameter
    xlb_macro = Macroscopic(
        velocity_set=velocity_set,
        compute_backend=backend,
        precision_policy=precision_policy,
    )
    xlb_second_moment = SecondMoment(
        velocity_set=velocity_set,
        compute_backend=backend,
        precision_policy=precision_policy,
    )
    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    cyliner_radius = cylinder_diameter // 2
    X = X.flatten()
    Y = Y.flatten()
    X = jnp.stack([X, Y], axis=-1)
    indices = jnp.where(
        (X[:, 0] - 2.0 * cylinder_diameter) ** 2
        + (X[:, 1] - 2.0 * cylinder_diameter) ** 2
        < cyliner_radius**2
    )
    evolve_fn = lbm_residual_builder(
        u_max,
        grid,
        grid_shape,
        cylinder_diameter,
        velocity_set,
        xlb_macro,
        solver_steps,
    )
    callbacks = [
        CheckpointCallback(path_checkpoint, name, hyperparams, True, 200),
        NODEUnrollingEvaluationCallbackXLB(
            dataset_validation,
            xlb_macro,
            max_step,
            max_step + 50,
            100,
            plot_results=False,
            plot_dir=path_experiment,
            dict_key_prefix="validation_unrolling",
            batch_size=8,
        ),
        NODEUnrollingEvaluationCallbackXLB(
            subdataset_train,
            xlb_macro,
            max_step,
            max_step + 50,
            100,
            plot_results=True,
            plot_dir=path_experiment,
            dict_key_prefix="train_unrolling",
            batch_size=8,
        ),
    ]
else:
    evolve_fn = None
    callbacks = [
        CheckpointCallback(path_checkpoint, name, hyperparams, True, 200),
        NODEUnrollingEvaluationCallback(
            dataset_validation,
            max_step,
            max_step + 50,
            100,
            plot_results=False,
            plot_dir=path_experiment,
            dict_key_prefix="validation_unrolling",
            batch_size=8,
        ),
        NODEUnrollingEvaluationCallback(
            subdataset_train,
            max_step,
            max_step + 50,
            100,
            plot_results=True,
            plot_dir=path_experiment,
            dict_key_prefix="train_unrolling",
            batch_size=8,
        ),
    ]


if DINO:
    scheduler = optx.schedules.exponential_decay(
        learning_rate_decoder,
        decay_steps,
        decay_rate,
        end_value=final_lr,
        staircase=True,
    )
    optimizer = optx.adam(scheduler)
    scheduler_node = optx.schedules.exponential_decay(
        learning_rate_node, decay_steps, decay_rate, end_value=final_lr, staircase=True
    )
    optimizer_node = optx.adam(scheduler_node)
    scheduler_latent = optx.schedules.exponential_decay(
        learning_rate_latent,
        decay_steps,
        decay_rate,
        end_value=final_lr,
        staircase=True,
    )
    assert (
        learning_rate_latent > 0
    ), "Learning rate for latent variable must be positive"
    optimizer_latent = optx.adam(scheduler_latent)
    trainer = IrregularDINoTrainer(
        model=model,
        model_state=model_state,
        optimizer=optimizer,
        optimizer_node=optimizer_node,
        optimizer_latent=optimizer_latent,
        loss=loss,
        evolve_fn=evolve_fn,
        evolve_start=evolve_start,
        max_evolve_split=max_split,
        split_start=split_start,
        random_split=False,
        num_trajectories=num_samples,
        num_time_steps=max_step,
        latent_dim=latent_dim,
        callbacks=callbacks,
        gamma=gamma,
        gamma_decay_rate=gamma_decay_rate,
        gamma_decay_epochs=gamma_epochs,
        final_gamma=final_gamma,
        key=subkey,
    )
else:
    scheduler = (
        optx.schedules.exponential_decay(
            learning_rate_decoder,
            decay_steps,
            decay_rate,
            end_value=final_lr,
            staircase=True,
        )
        if decay_rate < 1.0
        else learning_rate_decoder
    )
    print(scheduler)
    optimizer = optx.adamw(scheduler)
    if learning_rate_node > 0:
        scheduler_node = (
            optx.schedules.exponential_decay(
                learning_rate_node,
                decay_steps,
                decay_rate,
                end_value=final_lr,
                staircase=True,
            )
            if decay_rate < 1.0
            else learning_rate_node
        )
        optimizer_node = optx.adamw(scheduler_node)
    else:
        optimizer_node = None
    if learning_rate_latent > 0:
        scheduler_latent = (
            optx.schedules.exponential_decay(
                learning_rate_latent,
                decay_steps,
                decay_rate,
                end_value=final_lr,
                staircase=True,
            )
            if decay_rate < 1.0
            else learning_rate_latent
        )
        optimizer_latent = optx.adamw(scheduler_latent)
    else:
        optimizer_latent = None
    trainer = IrregularPhiROMTrainer(
        model=model,
        model_state=model_state,
        optimizer=optimizer,
        optimizer_node=optimizer_node,
        optimizer_latent=optimizer_latent,
        node_training_mode=node_training_mode,
        loss=loss,
        evolve_fn=evolve_fn,
        evolve_start=evolve_start,
        num_trajectories=num_samples,
        num_time_steps=max_step,
        latent_dim=latent_dim,
        callbacks=callbacks,
        gamma=gamma,
        key=subkey,
        xlb_macro=xlb_macro,
        xlb_second_moment=xlb_second_moment,
        mask_indices=None,
        loss_lambda=loss_lambda,
    )


model, model_state, opt_state, history = trainer.fit(
    loader_train, epochs=epochs, warm_start=True
)

model = eqx.nn.inference_mode(model, True)
save_model(os.path.join(path_experiment, "model.eqx"), hyperparams, model, model_state)
history["loss_reconstruction"] = np.array(history["loss_reconstruction"])
history["loss_time_stepping"] = np.array(history["loss_time_stepping"])
np.savez(os.path.join(path_experiment, "history.npz"), **history)

if autodecoder:
    l = np.array(trainer.latent_memory)
    np.save(os.path.join(path_experiment, "latent_memory.npy"), l)
