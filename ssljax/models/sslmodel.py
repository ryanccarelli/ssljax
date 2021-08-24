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
from ssljax.core.utils.register import get_from_register
from ssljax.models import Model


class SSLModel(Model, nn.Module, FromParams):
    """
    Base class implementing self-supervised model.

    Args:
        config (json/yaml?): model specification
    """

    def setup(self, config):
        # we want branches to be a nested dict where
        # each entry is a branch
        # then a forward pass is simply executing each
        # entry of the dict and returning the
        # tuple of outs
        # we also want to indicate groups of parameters
        # that share the same optimizer or are optimized
        # wrt one another
        self.branches = get_from_register(config.branches)

    @nn.compact
    def __call__(self, x):
        """
        Forward pass head and body.

        Args:
            x(tuple(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        # TODO:
        # using self.variable, self.param initialize the variables
        # from each branch separately and store in param dict?
        outs = {}
        for branchkey, branch in self.branches.items():
            out = x.copy()
            for layerkey, layer in branch.items():
                out = layer(out)
            outs[branchkey] = out
        return outs

    def freeze_head(self):
        raise NotImplementedError

    def freeze_body(self):
        raise NotImplementedError

    def is_frozen(self):
        raise NotImplementedError
