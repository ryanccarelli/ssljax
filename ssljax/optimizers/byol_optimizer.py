from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import utils as outils
from optax._src.transform import EmaState, _bias_correction, _update_moment
from ssljax.optimizers.optimizers import lars


def byol_optimizer(
    learning_rate,
    decay_rate,
    debias: bool = True,
    accumulator_dtype: Optional[Any] = None,
):
    """
    Optimizer that applies LARS to online network and
    updates target network by exponential moving average.

    Target branch is assumed to have params label "branch_0".
    Online branch is assumed to have params label "branch_1".
    Args:
        learning_rate: for lars
        decay rate: for ema
    """
    param_labels = ("branch_0", "branch_1")
    """
    return optax.multi_transform(
        {
            "branch_0": byol_ema(decay_rate, debias, accumulator_dtype),
            "branch_1": lars(learning_rate),
        },
        param_labels,
    )
    """

    #mask_fn = lambda p: jax.tree_map(lambda x: x in p["branch_0"], p)
    return optax.chain([
        optax.masked(byol_ema(decay_rate, debias, accumulator_dtype), {"branch_0":False, "branch_1":True}),
        optax.masked(lars(learning_rate), {"branch_0":True, "branch_1":False})
    ])


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

    accumulator_dtype = outils.canonicalize_dtype(accumulator_dtype)

    def init_fn(params):
        return EmaState(
            count=jnp.zeros([], jnp.int32),
            ema=jax.tree_map(
                lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params
            ),
        )

    def update_fn(updates, state, params=None):
        del params
        new_ema = state.ema
        new_ema["branch_0"] = _update_moment(
            updates["branch_0"], state.ema["branch_0"], decay, order=1
        )
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
