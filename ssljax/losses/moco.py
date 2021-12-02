from typing import Mapping, Optional, Text

import jax.numpy as jnp
from jax.tree_util import tree_leaves
from optax import sigmoid_binary_cross_entropy
from ssljax.core import register
from ssljax.losses.loss import Loss


@register(Loss, "moco_loss")
def moco_infonce_loss(
    outs: Mapping[str, Mapping[str, jnp.ndarray]],
    tau: float = 0.2,
    reduction: Optional[Text] = "mean",
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
        q_norm = jnp.linalg.norm(q, ord=2, axis=1)
        k_norm = jnp.linalg.norm(k, ord=2, axis=1)
        q = q / q_norm
        k = q / k_norm
        logits = jnp.einsum("nc,mc->nm", q, k) / tau
        labels = jnp.arange(logits.shape[0], dtype=jnp.float32)
        return sigmoid_binary_cross_entropy(logits, labels) * 2 * tau

    assert all(
        isinstance(x, jnp.ndarray) for x in tree_leaves(outs)
    ), "loss functions act on jnp.arrays"

    # outs["i"]["j"] indicates output of branch i applied to pipeline j
    loss = _contrastive_loss(outs["0"]["0"], outs["1"]["1"], tau) + _contrastive_loss(
        outs["0"]["1"], outs["1"]["0"], tau
    )
    if reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "none" or reduction is None:
        return loss
    else:
        raise ValueError(f"Incorrect reduction mode {reduction}")
