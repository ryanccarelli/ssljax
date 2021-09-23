from optax import (adabelief, adagrad, adam, adamw, dpsgd, fromage, lamb,
                   noisy_sgd, radam, rmsprop, sgd, yogi)
from optax._src.alias import lars
from ssljax.core.utils import register

__all__ = ["Optimizer"]


class Optimizer:
    pass


# Manually put register everything without creating subclasses
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
}

for name, func in optimizers.items():
    register(Optimizer, name)(func)
