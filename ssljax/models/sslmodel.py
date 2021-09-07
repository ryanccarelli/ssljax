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
from ssljax.augment import Augment
from ssljax.core.utils.register import get_from_register, register
from ssljax.models.branch.branch import Branch
from ssljax.models.model import Model

@register(Model, "SSLModel")
class SSLModel(Model):
    """
    Base class implementing self-supervised model.

    A self-supervised model consists of a set of branches
    that are executed in parallel on a list of augmented inputs,
    returning a list of branch outs.

    Args:
        config (json/yaml?): model specification
    """

    def setup(self, config):
        self.branches = []
        for branch_idx, branch_params in config.model.branches.items():
            branch = get_from_register(Branch, branch_params.name)(branch_params.params)
            self.branches.append(branch)

    def __call__(self, x):
        """
        Forward pass branches.

        Args:
            x(tuple(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        outs = []
        # implement as a map
        def executebranch(_x, branch):
            _x = branch(_x)
            return _x

        # use enumerate
        outs = map(
            lambda a, b: executebranch(a, b),
            x,
            self.branches,
        )
        return list(outs)

    def freeze_head(self):
        raise NotImplementedError

    def freeze_body(self):
        raise NotImplementedError

    def is_frozen(self):
        raise NotImplementedError
