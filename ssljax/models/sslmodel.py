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
        modules, branches, pipelines = {}, {}, {}

        for module_key, module_params in self.config.modules.items():
            modules[module_key] = get_from_register(Model, module_params.name)(module_params.params, name=module_key)

        for branch_key, branch_params in self.config.model.branches.items():
            stop_gradient = branch_params.stop_gradient
            pipelines[str(branch_key)] = branch_params.pipelines
            stages = {key: modules[val] for key, val in branch_params.items() if key not in ["stop_gradient", "pipelines"]}
            branches[str(branch_key)] = Branch(stages=stages, stop_gradient=stop_gradient)
            # add back pipelines or init fails
            branch_params.pipelines = pipelines[str(branch_key)]

        self.branches = branches
        self.pipelines = pipelines

    def __call__(self, x):
        """
        Forward pass branches.

        Args:
            x(dict(jnp.array)): each element of x represents
                raw data mapped through a different augmentation.Pipeline
        """
        outs = {}
        for key, val in self.branches.items():
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
