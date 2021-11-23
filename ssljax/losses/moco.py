from typing import Mapping

import jax.numpy as jnp
from optax.loss import sigmoid_binary_cross_entropy
from ssljax.core import register
from ssljax.losses.loss import Loss


@register(Loss, "moco_loss")
def moco_infonce_loss(
    outs: Mapping[str, Mapping[str, jnp.ndarray]],
    tau: float = 0.2,
) -> jnp.ndarray:
    """
    Compute MoCo v3 loss.

    Args:
        outs (Mapping[str, Mapping[str, jnp.ndarray]]): model output
    """

    def _contrastive_loss(q, k, tau):
        """
        Compute InfoNCE contrastive loss.

        Args:
            q (jnp.ndarray): query
            k (jnp.ndarray): key
            tau (float): temperature parameter
        """
        q = jnp.linalg.norm(q, ord=2, axis=1)
        k = jnp.linalg.norm(k, ord=2, axis=1)
        logits = jnp.einsum("nc,mc->nm", [q, k]) / tau
        labels = jnp.arange(logits.shape[0], dtype=jnp.float32)
        return sigmoid_binary_cross_entropy(logits, labels) * 2 * tau

    # outs["i"]["j"] indicates output of branch i applied to pipeline j
    return _contrastive_loss(outs["0"]["0"], outs["1"]["1"], tau) + _contrastive_loss(
        outs["0"]["1"], outs["1"]["0"], tau
    )
