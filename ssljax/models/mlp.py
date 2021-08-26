from typing import Callable, List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
from ssljax.optimizers.optimizers import adam

# from ssljax.models.model import Model


class StackedMLP(nn.Module):
    def setup(self):
        self.mlp1 = MLP(layer_dims=[500, 200, 10])

    @nn.compact
    def __call__(self, x, train=False):
        x = self.mlp1(x, train)
        return x


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
    activation: Callable = nn.relu

    def setup(self):
        layers = []
        if self.batch_norm:
            for layer in self.layer_dims[:-1]:
                layers.append(nn.Dense(layer, dtype=self.dtype))
                layers.append(self.activation)
                layers.append(nn.BatchNorm())
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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (2000,))
    model = StackedMLP()
    learning_rate = 1e-3
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(k2, x, k3)["params"],
        tx=adam(learning_rate),
    )
    print(state.params)
