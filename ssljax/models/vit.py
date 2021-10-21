import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from vit_jax import checkpoint, models, utils
from vit_jax.configs import models as models_config
from ssljax.models.model import Model

@register(Model, "VIT")
class VIT(nn.Module):
    """
    Flax implementation of vision transformer.
    We use ViT implementations maintained by https://github.com/google-research/vision_transformer.

    # TODO: optionally remove softmax layer for L2 loss

    Args:
        config (str): Vision transformer implementation. Choose from {ViT-B_32, Mixer-B-16}
    """

    config: str
    num_classes: int

    def setup(self):
        model_config = models_config.MODEL_CONFIGS[self.config]
        if self.config.startswith("Mixer"):
            self.model = models.MlpMixer(num_classes=self.num_classes, **model_config,)
        elif self.config.startswith("ViT"):
            self.model = models.VisionTransformer(
                num_classes=self.num_classes, **model_config,
            )
        else:
            raise KeyError(
                "invalid config, valid configs include: ViT-B_32, Mixer-B-16"
            )

    @nn.compact
    def __call__(self, x):
        return self.model(x)
