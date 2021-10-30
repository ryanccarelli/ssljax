import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from scenic.projects.baselines.mixer import Mixer
from ssljax.core.utils import register
from ssljax.models.model import Model


@register(Model, "Mixer")
class Mixer(Model):
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
