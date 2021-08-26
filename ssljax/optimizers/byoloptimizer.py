from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import utils as outils
from optax._src.transform import EmaState, _bias_correction, _update_moment
from ssljax.optimizers import lars


def byol_optimizer(learning_rate, decay_rate):
    """
    Optimizer that applies LARS to online network and
    updates target network by exponential moving average.

    Assumes that parameters are partitioned into
    "online" and "target" groups.
    Args:
        learning_rate: for lars
        decay rate: for ema

    TODO: There are potentially cleaner implementations.
    1. Use combine.chain() to execute lars (while masking byol_ema) then byol_ema
    current version has EMA trailing by one step. Sketch of improved implementation:

    optax.chain(
        [
            optax.multi_transform({"branch_0": identity, "branch_1":lars}),
            optax.multi_transform({"branch_0": byol_ema, "branch_1": identity}),
        ]
    )

    2. Use masking. The problem here is that we do not have access to state at this level:

    mask = jax.tree_map(lambda x: x in treedef_children(state.params["branch_0"])))
    optax.chain(
        [optax.masked(lars, mask), optax.masked(byol_ema, !mask)]
    )
    """
    param_labels = ("branch_0", "branch_1")
    return optax.multi_transform(
        {"branch_0": byol_ema(decay_rate), "branch_1": lars(learning_rate)},
        param_labels,
    )


def byol_ema(
    decay: float, debias: bool = True, accumulator_dtype: Optional[Any] = None
) -> base.GradientTransformation:
    """
    Update target network as an exponential moving average of the online network.

    Args:
        decay: the decay rate for the exponential moving average.
        debias: whether to debias the transformed gradient.
        accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
            then the `dtype` is inferred from `params` and `updates`.
    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        return EmaState(
            count=jnp.zeros([], jnp.int32),
            ema=jax.tree_map(
                lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params
            ),
        )

    def update_fn(updates, state, params=None):
        del params
        # branch_0 is online, branch_1 is target
        _update_moment(state.params["branch_0"], state.ema, decay, order=1)
        count_inc = outils.safe_int32_increment(state.count)
        if debias:
            new_ema = _bias_correction(new_ema, decay, count_inc)
        state_ema = outils.cast_tree(new_ema, accumulator_dtype)

        return new_ema, EmaState(count=count_inc, ema=state_ema)

    return base.GradientTransformation(init_fn, update_fn)


def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment."""
    return jax.tree_multimap(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments
    )


def _cosine_decay(
    global_step: jnp.ndarray, max_steps: int, initial_value: float
) -> jnp.ndarray:
    """
    Taken from 
    Simple implementation of cosine decay from TF1.
    This is used in BYOL optimizer to manage tau parameter.
    """
    global_step = jnp.minimum(global_step, max_steps)
    cosine_decay_value = 0.5 * (1 + jnp.cos(jnp.pi * global_step / max_steps))
    decayed_learning_rate = initial_value * cosine_decay_value
    return decayed_learning_rate
