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
from typing import Mapping, Optional, Text

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from ssljax.core import register
from ssljax.losses.loss import Loss


@register(Loss, "cosine_similarity")
def cosine_similarity(
    outs: Mapping[str, Mapping[str, jnp.ndarray]], reduction: Optional[Text] = "mean"
) -> jnp.ndarray:
    """
    Cosine similarity regression loss.

    Args:
        outs (Mapping[str, Mapping[str, jnp.ndarray]]): model output
        reduction (str): Type of reduction to apply to batch.
    """
    assert all(
        isinstance(x, jnp.ndarray) for x in tree_leaves(outs)
    ), "loss functions act on jnp.arrays"
    # outs["i"]["j"] indicates output of branch i applied to pipeline j
    normed_x, normed_y = (
        l2_normalize(outs["0"]["0"], axis=-1),
        l2_normalize(outs["1"]["1"], axis=-1),
    )
    loss = jnp.sum((normed_x - normed_y) ** 2, axis=-1)
    if reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "none" or reduction is None:
        return loss
    else:
        raise ValueError(f"Incorrect reduction mode {reduction}")


@register(Loss, "cross_entropy")
def cross_entropy(
    outs: Mapping[str, Mapping[str, jnp.ndarray]], reduction: Optional[Text] = "mean",
) -> jnp.ndarray:
    """
    Computes softmax cross entropy given logits and one-hot class labels.

    Args:
        outs (Mapping[str, Mapping[str, jnp.ndarray]]): model output
        reduction (str): Type of reduction to apply to batch.

    Returns:
        Loss value. If `reduction` is `none`, this has the same shape as `labels`;
        otherwise, it is scalar.

    Raises:
        ValueError: If the type of `reduction` is unsupported.
    """
    assert all(
        isinstance(x, jnp.ndarray) for x in tree_leaves(outs)
    ), "loss functions act on jnp.arrays"
    # outs["i"]["j"] indicates output of branch i applied to pipeline j
    loss = -jnp.sum(outs["1"]["1"] * jax.nn.log_softmax(outs["0"]["0"]), axis=-1)
    if reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "none" or reduction is None:
        return loss
    else:
        raise ValueError(f"Incorrect reduction mode {reduction}")


def l2_normalize(
    x: jnp.ndarray, axis: Optional[int] = None, epsilon: float = 1e-12,
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
