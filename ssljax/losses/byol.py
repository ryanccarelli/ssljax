# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# THIS CODE HAS BEEN MODIFIED FROM ITS SOURCE
from typing import Optional, Text

import jax
import jax.numpy as jnp
from ssljax.core import register
from ssljax.losses.loss import Loss

@register(Loss, "byol_regression_loss")
def byol_regression_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Cosine similarity regression loss.
    """
    assert isinstance(x, jnp.ndarray), "loss functions act on jnp.arrays"
    assert isinstance(y, jnp.ndarray), "loss functions act on jnp.arrays"
    normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
    return jnp.sum((normed_x - normed_y) ** 2, axis=-1)

@register(Loss, "byol_softmax_cross_entropy_loss")
def byol_softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    reduction: Optional[Text] = "mean",
) -> jnp.ndarray:
    """Computes softmax cross entropy given logits and one-hot class labels.

    Args:
        logits: Logit output values.
        labels: Ground truth one-hot-encoded labels.
        reduction: Type of reduction to apply to loss.

    Returns:
        Loss value. If `reduction` is `none`, this has the same shape as `labels`;
        otherwise, it is scalar.

    Raises:
        ValueError: If the type of `reduction` is unsupported.
    """
    assert isinstance(logits, jnp.ndarray), "loss functions act on jnp.arrays"
    assert isinstance(labels, jnp.ndarray), "loss functions act on jnp.arrays"
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    if reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "none" or reduction is None:
        return loss
    else:
        raise ValueError(f"Incorrect reduction mode {reduction}")

@register(Loss, "l2_normalize")
def l2_normalize(
    x: jnp.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
    """
    l2 normalize a tensor on an axis with numerical stability.

    Args:
        x (jnp.mdarray):
    """
    assert isinstance(x, jnp.ndarray), "loss functions act on jnp.arrays"
    assert isinstance(axis, int), "axis must be int"
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm
