"""
utilities for saving and loading models.
"""

import json
import os

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..modules.models import CROMOffline, NodeROM


def save_model(filename, hyperparams, model, state):
    with open(filename, "wb") as f:
        hyperparams = jax.tree.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x, hyperparams
        )
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, (model, state))


def save_opt_state(filename, opt_state):
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, opt_state)


def load_model(filename, make):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        for key in list(hyperparams.keys()):
            if key.endswith("_numpy"):
                hyperparams[key[:-6]] = jax.numpy.array(
                    hyperparams[key],
                )
                del hyperparams[key]
        hyperparams = jax.tree.map(
            lambda x: jax.numpy.array(x) if isinstance(x, list) else x,
            hyperparams,
            is_leaf=lambda x: isinstance(x, list),
        )
        hyperparams = jax.tree.map(
            lambda x: jnp.array(x) if isinstance(x, float) else x,
            hyperparams,
            is_leaf=lambda x: isinstance(x, float),
        )
        (model, state) = make(key=jr.PRNGKey(0), **hyperparams)
        model, state = eqx.tree_deserialise_leaves(f, (model, state))
        model = eqx.nn.inference_mode(model, True)
        return model, state


def make_CROMOffline(key, activation=jnn.elu, **hyperparams):
    if "activation" not in hyperparams:
        hyperparams["activation"] = activation
    return eqx.nn.make_with_state(CROMOffline)(key=key, **hyperparams)


def make_NodeROM(key, activation=jnn.elu, **hyperparams):
    if "activation" not in hyperparams:
        hyperparams["activation"] = activation
    return eqx.nn.make_with_state(NodeROM)(key=key, **hyperparams)
