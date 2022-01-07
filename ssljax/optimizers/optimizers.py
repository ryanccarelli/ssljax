from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import optax
from optax import (adabelief, adagrad, adam, adamw, dpsgd, fromage, lamb,
                   noisy_sgd, radam, rmsprop, sgd, yogi)
from optax._src import base, combine, transform
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate, lars
from ssljax.core import register

__all__ = ["Optimizer"]


class Optimizer:
    pass


def zerog():
    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


def sgdw(
    learning_rate: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """
    A variant of the canonical Stochastic Gradient Descent optimizer with weight decay.

    References:
        Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

    Args:
        learning_rate: this is a fixed global scaling factor.
        momentum: (default `None`), the `decay` rate used by the momentum term,
            when it is set to `None`, then momentum is not used at all.
        nesterov (default `False`): whether nesterov momentum is used.
        accumulator_dtype: optional `dtype` to be used for the accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay (default 1e-4): strength of the weight decay regularization.
        mask: optional; a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.
    Returns:
        A `GradientTransformation`.
    """
    return combine.chain(
        (
            transform.trace(
                decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype
            )
            if momentum is not None
            else base.identity()
        ),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )


# Manually register everything without creating subclasses
optimizers = {
    "adabelief": adabelief,
    "adagrad": adagrad,
    "adam": adam,
    "adamw": adamw,
    "dpsgd": dpsgd,
    "fromage": fromage,
    "lamb": lamb,
    "noisy_sgd": noisy_sgd,
    "radam": radam,
    "rmsprop": rmsprop,
    "sgd": sgd,
    "yogi": yogi,
    "lars": lars,
    "zerog": zerog,
    "sgdw": sgdw,
}


for name, func in optimizers.items():
    register(Optimizer, name)(func)
