import collections
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from omegaconf import DictConfig
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
        config (ssljax.conf.config): model specification
    """

    config: DictConfig

    def setup(self):
        branch = []
        for branch_idx, branch_params in self.config.model.branches.items():
            b = Branch(branch_params)
            branch.append(b)
        self.branch = branch

    def __call__(self, x):
        """
        Forward pass branches.

        Args:
            x(tuple(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        outs = []
        # x = jnp.split(x, x.shape[-1], axis=-1)
        # x = [jnp.squeeze(y, axis=-1) for y in x]

        for idx, b in enumerate(self.branch):
            outs.append(b(x[..., idx]))

        return outs

    def freeze_head(self, branch_name):
        raise NotImplementedError

    def freeze_body(self, branch_name):
        raise NotImplementedError

    def is_frozen(self):
        raise NotImplementedError
