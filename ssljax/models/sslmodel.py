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
import collections

import jax
import jax.numpy as jnp
from flax import linen as nn
from ssljax.augment import BaseAugment
from ssljax.config import FromParams
from ssljax.core.utils.register import get_from_register
from ssljax.models import Branch, Model


class SSLModel(Model):
    """
    Base class implementing self-supervised model.

    Args:
        config (json/yaml?): model specification
    """

    def setup(self, config):
        # branch implements optax.multi_transform
        self.branches = get_from_register(config.branches)
        assert all(
            (isinstance(x, Branch) for x in self.branches)
        ), "self.branches must be a list of branches"

    @nn.compact
    def __call__(self, x):
        """
        Forward pass branches.

        Args:
            x(tuple(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        outs = collections.OrderedDict()
        for index, (branchkey, branch) in enumerate(self.branches.items()):
            out = x[index].copy()
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
