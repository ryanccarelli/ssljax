import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.baselines.vit import ViT
from ssljax.models.model import Model
from omegaconf import DictConfig

@register(Model, "ViT")
class ViT(nn.Module):
    """
    Flax implementation of vision transformer.
    We wrap the ViT model in https://github.com/google-research/scenic

    Args:
        config (ssljax.core.config): OmegaConf
    """

    config: DictConfig

    def setup(self):
        self.model = ViT(**self.config)

    @nn.compact
    def __call__(self, x):
        return self.model(x)
