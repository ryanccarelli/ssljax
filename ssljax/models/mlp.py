from typing import List, Callable
import flax.linen as nn
import jax.numpy as jnp
import jax
from ssljax.models.model import Model


class MLP(Model):
    layer_dims: List[int]
    dtype: jax._src.numpy.lax_numpy._ScalarMeta = jnp.float32
    dropout_prob: float = 0.0
    batch_norm: bool = False
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for layer in self.layer_dims[:-1]:
            x = nn.Dense(layer, dtype=self.dtype)(x)
            x = self.activation(x)
        out = nn.Dense(self.layer_dims[-1], dtype=self.dtype)(x)
        return out


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (2000,))
    model = MLP(layer_dims=[500, 200, 10])
    params = model.init(k2, x)
    out = model.apply(params, x)
    print(out)
