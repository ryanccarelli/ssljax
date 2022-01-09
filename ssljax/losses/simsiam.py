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
    # TODO: pass this from config file
    predconfig = {
        "layer_dims": [512, 2048],
        "activation_name": "relu",
        "dropout_prob": 0.0,
        "batch_norm": True,
        "batch_norm_params": {
            "use_running_average": True,
            "momentum": 0.1,
            "epsilon": 1e-5,
        },
        "dtype": "float32",
    }

    mlp = MLP(OmegaConf.create(predconfig))

    view1 = outs[0][0]
    view2 = outs[0][1]

    pred1 = mlp(view1)
    pred2 = mlp(view2)

    # gt uses nn.CosineSimilarity(dim=1)
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
