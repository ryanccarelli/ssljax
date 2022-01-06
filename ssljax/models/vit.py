import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from omegaconf import DictConfig
from scenic.projects.baselines.vit import ViT as vit
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
        # scenic constructs patch_sizes in backend
        patches = ml_collections.ConfigDict()
        size = self.config.patch_size
        del self.config.patch_size
        patches.size = [int(size), int(size)]
        self.model = vit(**self.config, patches=patches)

    @nn.compact
    def __call__(self, x, train: bool=True):
        return self.model(x, train=train)
