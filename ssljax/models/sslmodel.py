import collections
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from omegaconf import DictConfig
from ssljax.augment import Augment
from ssljax.core import get_from_register, register
from ssljax.models.branch.branch import Branch
from ssljax.models.model import Model


@register(Model, "SSLModel")
class SSLModel(Model):
    """
    Base class implementing a self-supervised model.
    A self-supervised model consists of a list of branches
    that are executed in parallel to process augmented views of inputs.

    This class is used by ``ssljax.core.utils.register``.

    Args:
        config (ssljax.conf.config): model specification
    """

    config: DictConfig

    def setup(self):
        branch = {}
        for idx, branch_params in self.config.model.branches.items():
            b = Branch(branch_params.stages)
            branch[idx] = {"branch": b, "pipelines": branch_params.pipelines}
        self.branch = branch

    def __call__(self, x):
        """
        Forward pass branches.

        Args:
            x(dict(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        outs = {}

        for key, (branch, pipelines) in self.branch.items():
            for pipeline in pipelines:
                outs[key][pipeline] = branch(x[pipeline])

        return outs

    def freeze_head(self, branch_name):
        raise NotImplementedError

    def freeze_body(self, branch_name):
        raise NotImplementedError

    def is_frozen(self):
        raise NotImplementedError
