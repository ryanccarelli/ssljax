from ssljax.models.model import Model

@register(Model, "Center")
class Center(Model):
    """
    Flax implementation of centering layer.
    Required for DINO model, see update_center in https://github.com/facebookresearch/dino.

    Needs to hold
    """
    momentum: float
    def setup(self):
        self.center = jnp.zeros(1)

    @nn.compact
    def __call__(self, x):
        x = jnp.sum(x, dim=0, keepdims=True)
