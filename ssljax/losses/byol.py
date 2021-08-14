from ssljax.core.utils import Registrable


@Loss.register("regression")
class BYOLLoss:
    """
    Cosine similarity regression loss.
    """
    normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
    return jnp.sum((normed_x - normed_y)**2, axis=-1)



def l2_normalize(
        x: jnp.ndarray,
        axis: Optional[int] = None,
        epsilon: float = 1e-12,
) -> jnp.ndarray:
    """l2 normalize a tensor on an axis with numerical stability."""
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm



