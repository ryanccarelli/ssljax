import jax.numpy as jnp
from optax.loss import sigmoid_binary_cross_entropy
from ssljax.core import register
from ssljax.losses.loss import Loss


@register(Loss, "infonce_loss")
def infonce_loss(kp: jnp.ndarray, kq: jnp.ndarray, q: jnp.ndarray, tau: float):
    """
    InfoNCE contrastive loss.

    Args:
        kp (jnp.ndarray): positive key
        kn (jnp.ndarray): negative keys
        q (jnp.ndarray): query
        tau (float): temperature
    """
    tau = jnp.array(tau)
    positive = jnp.exp(jnp.dot(q, kp) / tau)
    negative = jnp.sum(jnp.exp(jnp.dot(q, kq) / tau), axis=1)
    return 2 * tau * -jnp.log(positive / (positive + negative))


def moco_infonce_loss(q, k, tau):
    q = jnp.linalg.norm(q, ord=2, axis=1)
    k = jnp.linalg.norm(k, ord=2, axis=1)
    # TODO: concat_all_gather in torch implementation
    logits = jnp.einsum("nc,mc->nm", [q, k]) / tau
    labels = jnp.arange(logits.shape[0], dtype=jnp.float32)
    return sigmoid_binary_cross_entropy(logits, labels) * 2 * tau


def cross_entropy_loss(logits, labels):
    # TODO
    pass
