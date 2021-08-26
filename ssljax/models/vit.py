import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


class VIT(nn.Module):
    """
    Flax implementation of vision transformer.

    Args:

    """

    def setup(self):
        pass

    @nn.compact
    def __call__(self, x):
        pass


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (256, 256, 3))
    model = VIT()
    params = model.init(k2, x)
    out = model.apply(params, x, train=True)
    print(out)
