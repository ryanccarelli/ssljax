# Base SSL class here
import jax
import jax.numpy as jnp
from flax import linen as nn


class BaseSSL:
    """
    Base class implementing self-supervised model.

    Args:
        config (json/yaml?): model specification
    """

    def __init__(self, config, params):
        self.config = config
        # read config

        self.head = self._setup_head()
        self.body = self._setup_body()
        self.loss = self._setup_loss()

    def parse_config():
        """
        Mostly we want to know if params are in config so we can declare from pretrained
        """
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
        pass
