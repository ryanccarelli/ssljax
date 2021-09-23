import logging

from optax import (constant_schedule, cosine_decay_schedule,
                   cosine_onecycle_schedule, exponential_decay,
                   linear_onecycle_schedule, piecewise_constant_schedule,
                   piecewise_interpolate_schedule, polynomial_schedule)
from ssljax.core.utils.register import register
from ssljax.train.scheduler.ema import ema

logger = logging.getLogger(__name__)

logger.error(f"{__name__}: THESE ARE PLACEHOLDERS!!")


class Scheduler:
    """
    Schedulers are used to alter the value of a parameter over time.
    Learning rate schedulers must subclass Scheduler.
    """

    pass


schedulers = {
    "constant": constant_schedule,
    "cosine_decay": cosine_decay_schedule,
    "cosine_onecycle": cosine_onecycle_schedule,
    "exponential_decay": exponential_decay,
    "linear_onecycle": linear_onecycle_schedule,
    "piecewise_constant": piecewise_constant_schedule,
    "piecewise_interpolate": piecewise_interpolate_schedule,
    "polynomial": polynomial_schedule,
    "ema": ema,
}

for name, func in schedulers.items():
    register(Scheduler, name)(func)
