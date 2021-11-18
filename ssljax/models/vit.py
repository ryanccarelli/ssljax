import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from scenic.projects.baselines.vit import ViT
from ssljax.core import register
from ssljax.models.model import Model


@register(Model, "ViT")
class ViT(Model):
    """
    Flax implementation of a vision transformer.
    We wrap the ViT model in `<scenic> https://github.com/google-research/scenic`_.

    Args:
        config (omegaconf.DictConfig): configuration
    """

    config: DictConfig

    def setup(self):
        self.model = ViT(**self.config)

    @nn.compact
    def __call__(self, x):
        return self.model(x)
