import logging
import random

import jax
import numpy
from jax import random as jax_random

logger = logging.getLogger(__name__)
__all__ = [
    "prepare_environment",
]


def prepare_environment(config) -> jax.numpy.DeviceArray:
    """
    Set the random seeds.

    Args:
    config: The config object.

    Returns (jax.numpy.DeviceArray): The jax RNG generator.
    """

    # Get the seed values from the config.
    # TODO(gabeorlanski): Replace the pop and cast with a `pop_int` function
    seed = config.env.seed if ("env" in config and config.env.seed) else 0
    numpy_seed = (
        config.env.numpy_seed if ("env" in config and config.env.numpy_seed) else 0
    )
    jax_seed = config.env.jax_seed if ("env" in config and config.env.jax_seed) else 0

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)

    # Set the jax seed and return it. If the jax seed is None, default to 0.
    return jax_random.PRNGKey(jax_seed if jax_seed is not None else 0)


def wrap_func_in_class_call(name, func, BaseClass=object):
    """ Wraps a function in a class with the function as __call__

    Args:
        name(str): Name of the new class
        func(Callable): A function which will be the mapped to __call__
        BaseClass(Class, <optional>): Base class of the new function
    """
    newclass = type(name, (BaseClass,), {"__call__": lambda self, *args: func(*args)})
    return newclass
