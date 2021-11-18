import jax
import jax.numpy as jnp
import optax
from optax import (adabelief, adagrad, adam, adamw, dpsgd, fromage, lamb,
                   noisy_sgd, radam, rmsprop, sgd, yogi)
from optax._src.alias import lars
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
}


for name, func in optimizers.items():
    register(Optimizer, name)(func)
