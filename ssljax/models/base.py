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


class BaseSSL(FromParams, nn.Module):
    """
    Base class implementing self-supervised model.

    Args:
        config (json/yaml?): model specification
        params (str): path to model parameters
    """

    def __init__(self, config, params: None):
        self.config = config
        self.params = params
        # TODO: read config
        self.head = self._setup_head(config)
        self.body = self._setup_body(config)
        self.loss = self._setup_loss(config)
        self.optimizer = self._setup_optimizer(config)
        self.trainer = self._setup_trainer(config)

    def __call__():
        # this takes the place of vissl forward
        raise NotImplementedError

    # inherit from FromParams
    def from_params():
        raise NotImplementedError

    def _setup_head():
        """
        Read from config and instantiate head

        Returns:
            list
        """
        raise NotImplementedError

    def _setup_body():
        # read from config and instantiate body
        raise NotImplementedError

    def _setup_loss():
        # read from config and instantiate loss
        raise NotImplementedError

    def freeze_head():
        raise NotImplementedError

    def freeze_body():
        raise NotImplementedError

    def is_frozen():
        raise NotImplementedError

    @jax.jit
    def apply_model():
        raise NotImplementedError
