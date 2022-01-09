import jax
import jax.numpy as jnp
from typing import Mapping

from ssljax.core import register
from ssljax.losses.loss import Loss
import flax.linen as nn

@register(Loss, "dino_loss")
def dino_loss(
    outs: Mapping[str, Mapping[str, jnp.ndarray]],
    tau_t: float,
    tau_s: float = 0.1,
) -> jnp.ndarray:
    """
    Sharpening followed by
    """
    teacher_out = outs["0"]["0"]
    student_out = outs["1"]["1"]

    student_out = student_out/tau_s
    teacher_out = nn.softmax(teacher_out/tau_t)
