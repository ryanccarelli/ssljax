from typing import List, Callable
import flax
import flax.linen as nn
import jax.numpy as jnp
import jax
#from ssljax.models.model import Model


class MLP(nn.Module):
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
                x = layer(x, deterministic=not train)
            else:
                x = layer(x)
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (2000,))
    model = MLP(layer_dims=[500, 200, 10])
    params = model.init(k2, x)
    out = model.apply(params, x, train=True, rngs={"dropout": k3})
    print(out)
