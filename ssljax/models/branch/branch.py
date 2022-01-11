import flax.linen as nn
import jax.lax
from ssljax.core import get_from_register, register
from ssljax.models.model import Model
from omegaconf import DictConfig


class Branch(Model):
    """
    A Branch is a nn.Module that is executed in parallel with other branches.
    Branches sequentially execute stages, typically  a model body (eg. ResNet, ViT),
    then optionally a model head and predictor (eg. MLP).

    This class is used by ``ssljax.core.utils.register``.

    Args:
        stages (dict): dictionary containing modules indexed by name
        stop_gradient (bool): whether gradients will propagate through this branch
        intermediate (list): if dict

    Returns:
        outs: Mapping[str, jnp.ndarray]
    """

    stages: dict
    stop_gradient: bool = False
    intermediate: list or None = None

    def __call__(self, x):
        outs = {}
        for key, val in self.stages.items():
            finalkey = key
            x = val(x)
            if self.intermediate and (key in self.intermediate):
                outs[key] = x
        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)
        outs[finalkey] = x
        return outs


# lol
register(Branch, "Branch")(Branch)
