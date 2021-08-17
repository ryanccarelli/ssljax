"""
We want to support two configurations:
    1. contrastive learning by negative pairs (eg SIMCLR)
    2. non-contrastive (eg BYOL, SimSiam)

The non-contrastive approaches replace negative pairs with:
    1. a learnable predictor
    2. a stop-gradient

Support
stop gradient on target but not on predictor
both ema of bodies and bodies with shared parameters

"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from ssljax.config import FromParams
from ssljax.models import Body, Head


class BaseSSL(ModelBase, nn.Module, FromParams):
    """
    Base class implementing self-supervised model.

    Args:
        config (json/yaml?): model specification
    """

    def __init__(self, config):
        self.config = config
        self.head = fromParams(Head, self.config.pop("head"))
        self.body = fromParams(Body, self.config.pop("body"))

    def __call__(self, x):
        """
        Forward pass head and body.
        """
        raise NotImplementedError

    def freeze_head():
        raise NotImplementedError

    def freeze_body():
        raise NotImplementedError

    def is_frozen():
        raise NotImplementedError
