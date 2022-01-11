import jax
import jax.numpy as jnp
from typing import Mapping, List

from ssljax.core import register
from ssljax.losses.loss import Loss
import flax.linen as nn

@register(Loss, "dino_loss")
def dino_loss(
    outs: Mapping[str, Mapping[str, List[jnp.ndarray]]],
    tau_t: float,
    tau_s: float = 0.1,
) -> jnp.ndarray:
    """
    Compute sharpened loss over views.
    """

    teacher_out = outs["0"]["0"]["proj"]
    student_out = outs["1"]["1"]["proj"]

    student_out = student_out/tau_s
    teacher_out = nn.softmax(teacher_out/tau_t)

    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # skip case where student and teacher same view
                continue
            loss = jnp.sum(-q * nn.log_softmax(student_out[v], axis=-1), axis=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms
    # TODO: centering
    return total_loss
