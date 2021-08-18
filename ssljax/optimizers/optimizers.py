from optax import (adabelief, adagrad, adam, adamw, dpsgd, fromage, lamb,
                   noisy_sgd, radam, rmsprop, sgd, yogi)
from optax._src.alias import lars
from ssljax.core.utils import Registrable


class Optimizer(Registrable):
    pass


# Manually put register everything without creating subclasses
Registrable._registry[Optimizer] = {
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
