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
        pipelines = {}
        for idx, branch_params in self.config.model.branches.items():
            b = Branch(branch_params.stages)
            branch[str(idx)] = b
            pipelines[str(idx)] = branch_params.pipelines
        self.branch = branch
        self.pipelines = pipelines

    def __call__(self, x):
        """
        Forward pass branches.

        Args:
            x(dict(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        outs = {}
        for key, val in self.branch.items():
            add = {}
            for pipeline in self.pipelines[key]:
                add[pipeline] = val(x[pipeline])
            outs[key] = add

        return outs

    def detach_module(self, branch, module):
        """
        Detach a module (for inference).

        Args:
            branch (str): branch key
            module (str): module key

        Example:
            TODO
        """
        # overwrite call?
        raise NotImplementedError

    def is_frozen(self):
        """
        Returns:
            bool: true if any module is frozen
        """
        raise NotImplementedError
