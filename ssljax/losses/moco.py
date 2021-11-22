import jax.numpy as jnp
from optax.loss import sigmoid_binary_cross_entropy
from ssljax.core import register
from ssljax.losses.loss import Loss


@register(Loss, "moco_loss")
def moco_infonce_loss(q, k, tau):
    def _contrastive_loss(q, k, tau):
        q = jnp.linalg.norm(q, ord=2, axis=1)
        k = jnp.linalg.norm(k, ord=2, axis=1)
        logits = jnp.einsum("nc,mc->nm", [q, k]) / tau
        labels = jnp.arange(logits.shape[0], dtype=jnp.float32)
        return sigmoid_binary_cross_entropy(logits, labels) * 2 * tau

    return _contrastive_loss(q[0], k[1], tau) + _contrastive_loss(q[1], k[0], tau)
