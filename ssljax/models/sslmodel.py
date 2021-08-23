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
from ssljax.augment import BaseAugment
from ssljax.config import FromParams
from ssljax.models import Model


class SSLModel(Model, nn.Module, FromParams):
    """
    Base class implementing self-supervised model.

    Args:
        config (json/yaml?): model specification
    """

    # TODO: in the case of multiple heads and bodies
    # do we have here lists?

    def setup(config, head, body):
        self.head = head
        self.body = body
        self.branches = []
        for branch in len(self.body):
            # iterate over first element of head and body?
            pass

    def __call__(self, x):
        """
        Forward pass head and body.

        Args:
            x(tuple(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        # TODO: this is terrible
        outs = []
        for bindex, branch in enumerate(branches):
            xtemp = x
            for layer in branch:
                xtemp = layer(xtemp)
            outs[bindex] = xtemp
        return outs

    def freeze_head():
        raise NotImplementedError

    def freeze_body():
        raise NotImplementedError

    def is_frozen():
        raise NotImplementedError
