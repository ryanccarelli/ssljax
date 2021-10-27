from ssljax.models.model import Model


@register(Model, "Center")
class Center(Model):
    """
    Flax implementation of centering layer.
    Required for DINO model, see update_center in https://github.com/facebookresearch/dino.

    The output of the
    teacher network is centered with a mean computed over the batch.
    Each networks outputs a K dimensional feature that is normalized
    with a temperature softmax over the feature dimension. Their
    similarity is then measured with a cross-entropy loss.
    """

    momentum: float

    def setup(self):
        self.center = jnp.zeros(1)

    @nn.compact
    def __call__(self, x):
        x = jnp.sum(x, dim=0, keepdims=True)
