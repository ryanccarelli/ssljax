from typing import Mapping, Optional, Text

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from ssljax.core import register
from ssljax.losses.loss import Loss


@register(Loss, "barlow_twins_loss")
def barlow_twins_loss(
    outs: Mapping[str, Mapping[str, Mapping[str, jnp.ndarray]]],
    batch_size: int,
    lambd: float,
    reduction: Optional[Text] = "mean",
) -> jnp.ndarray:
    """
    Barlow twins loss function.
    Adapted from https://github.com/facebookresearch/barlowtwins/blob/82b7c4539fdedc9f0f123a40f027451a4063c7e5/main.py#L187

    Args:
        outs (Mapping[str, Mapping[str, jnp.ndarray]]): model output
        reduction (str): Type of reduction to apply to batch.
        batch_size (int): Size of batch
        lambd (float): hyperparameter weighting loss terms
    """

    assert all(
        isinstance(x, jnp.ndarray) for x in tree_leaves(outs)
    ), "loss functions act on jnp.arrays"

    # outs["i"]["j"] indicates output of branch i applied to pipeline j
    # NOTE: to reproduce, output of pred must be batchnormed
    # TODO: extra dimensions! (batch_size, 1, 1, mlp_out_dim)
    pred0 = jnp.squeeze(outs["0"]["0"]["pred"])
    pred1 = jnp.squeeze(outs["1"]["1"]["pred"])
    c = pred0.T @ pred1
    c = jax.lax.div(c, float(batch_size))

    # we had some discussion about whether this introduces
    # redundant sync and computation of loss on every device
    c = jax.lax.psum(c, axis_name="batch")
    on_diag = jnp.sum(jnp.power(jnp.add(jnp.diag(c), 1), 2))
    off_diag = jnp.sum(jnp.power(off_diagonal(c), 2))

    loss = on_diag + lambd * off_diag

    return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    flat = jnp.reshape(x, [-1])[:-1]
    off_diagonals = jnp.reshape(flat, (n - 1, n + 1))[:, 1:]
    off_diag = jnp.reshape(off_diagonals, [-1])
    return off_diag
