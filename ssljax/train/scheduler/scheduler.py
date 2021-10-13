import logging

import jax.numpy as jnp
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


@register(Scheduler, "BYOL")
def byol_schedule(
    batch_size: int, base_learning_rate: float, total_steps: int, warmup_steps: int,
) -> float:
    """
    Cosine learning rate scheduler for BYOL.
    """

    def schedule(global_step):
        # Compute LR & Scaled LR
        scaled_lr = base_learning_rate * batch_size / 256.0
        learning_rate = (
            global_step.astype(jnp.float32) / int(warmup_steps) * scaled_lr
            if warmup_steps > 0
            else scaled_lr
        )

        # Cosine schedule after warmup.
        return jnp.where(
            global_step < warmup_steps,
            learning_rate,
            _cosine_decay(
                global_step - warmup_steps, total_steps - warmup_steps, scaled_lr
            ),
        )

    return schedule


def _cosine_decay(
    global_step: jnp.ndarray, max_steps: int, initial_value: float
) -> jnp.ndarray:
    """Simple implementation of cosine decay from TF1."""
    global_step = jnp.minimum(global_step, max_steps)
    cosine_decay_value = 0.5 * (1 + jnp.cos(jnp.pi * global_step / max_steps))
    decayed_learning_rate = initial_value * cosine_decay_value
    return decayed_learning_rate
