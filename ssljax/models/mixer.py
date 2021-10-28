import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.baselines.mixer import ViT
from ssljax.models.model import Model
from omegaconf import DictConfig

@register(Model, "Mixer")
class Mixer(nn.Module):
    """
    Flax implementation of MLP Mixer.
    We wrap the Mixer model in https://github.com/google-research/scenic

    Args:
        config (ssljax.core.config): OmegaConf
    """

    config: DictConfig

    def setup(self):
        self.model = Mixer(**self.config)

    @nn.compact
    def __call__(self, x):
        return self.model(x)
