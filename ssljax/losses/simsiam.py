from typing import Mapping, Optional, Text

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from omegaconf import OmegaConf
from ssljax.core import register
from ssljax.losses.loss import Loss
from ssljax.models.mlp import MLP


@register(Loss, "simsiam_loss")
def simsiam_loss(
    outs: Mapping[str, Mapping[str, jnp.ndarray]],
) -> jnp.ndarray:
    """
    SimSiam loss.
    """

    view1 = outs[0][0]["head"]
    view2 = outs[0][1]["head"]

    pred1 = outs[0][0]["pred"]
    pred2 = outs[0][1]["pred"]

    # ground truth uses torch.nn.CosineSimilarity(dim=1)
    loss = (
        -(
            cosinesimilarity(pred1, stop_gradient(view2)).mean()
            + cosinesimilarity(pred2, stop_gradient(view1)).mean()
        )
        * 0.5
    )
    return loss


# TODO: move w/ byol cossim into common folder
# after merge with open byol dev branch
def cosinesimilarity(x: jnp.ndarray, y: jnp.ndarray):
    assert isinstance(
        x, jnp.ndarray
    ), f"inputs are of type {type(x)} but must be type jnp.ndarray"
    assert isinstance(
        y, jnp.ndarray
    ), f"inputs are of type {type(y)} but must be type jnp.ndarray"
    normed_x, normed_y = (
        l2_normalize(x, axis=-1),
        l2_normalize(y, axis=-1),
    )
    loss = jnp.sum((normed_x - normed_y) ** 2, axis=-1)
    return loss


def l2_normalize(
    x: jnp.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
    """
    l2 normalize a tensor on an axis with numerical stability.

    Args:
        x (jnp.ndarray):
    """
    assert isinstance(x, jnp.ndarray), "loss functions act on jnp.arrays"
    assert isinstance(axis, int), "axis must be int"
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm
