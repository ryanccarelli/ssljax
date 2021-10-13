from typing import Callable, List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
from ssljax.models.model import Model
from ssljax.optimizers.optimizers import adam


@register(Model, "MLP")
class MLP(nn.Module):
    """
    Flax implementation of multilayer perceptron.

    Args:
        layer_dims(List[int]): list indicating number of neurons in each layer
        dtype: jnp datatype
        dropout_prob(float): dropout rate hyperparameteri
        batch_norm(bool): whether to use batchnorm between layers
    """

    layer_dims: List[int]
    dtype: jax._src.numpy.lax_numpy._ScalarMeta = jnp.float32
    dropout_prob: float = 0.0
    batch_norm: bool = False
    batch_norm_params: dict = {}
    activation: Callable = "relu"

    def setup(self):
        if self.activation == "relu":
            self.activation = nn.relu
        layers = []
        if self.batch_norm:
            for layer in self.layer_dims[:-1]:
                layers.append(nn.Dense(layer, dtype=self.dtype))
                layers.append(nn.BatchNorm(**self.batch_norm_params))
                layers.append(self.activation)
                layers.append(nn.Dropout(rate=self.dropout_prob))
        else:
            for layer in self.layer_dims[:-1]:
                layers.append(nn.Dense(layer, dtype=self.dtype))
                layers.append(self.activation)
                layers.append(nn.Dropout(rate=self.dropout_prob))
        layers.append(nn.Dense(self.layer_dims[-1], dtype=self.dtype))
        self.layers = layers

    @nn.compact
    def __call__(self, x, train=False):
        for layer in self.layers:
            if isinstance(layer, flax.linen.stochastic.Dropout):
                x = layer(x, deterministic=True)
            else:
                x = layer(x)
        return x
