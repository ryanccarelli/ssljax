import flax.linen as nn
import jax.lax
from ssljax.core.utils import get_from_register, register
from ssljax.models.model import Model


class Branch(Model):
    """
    A Branch is a nn.Module that is executed in parallel with other branches.
    Branches sequentially execute stages, typically  a model body (eg. ResNet, ViT),
    then optionally a model head and predictor (eg. MLP).

    Args:
        config (hydra.OmegaConf): config file at config.model.branches.i where i is branch index
    """

    config: dict

    def setup(self, **args):
        self.stop_gradient = self.config["stop_gradient"]
        stages = []
        for stage_name, stage_params in self.config.items():
            if stage_name != "stop_gradient":
                stages.append(
                    get_from_register(Model, stage_params.module)(
                        name=stage_name, **stage_params.params
                    )
                )

        self.stages = stages

    def __call__(self, x):
        for stage in self.stages:
            x = stage(x)
        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)
        return x


# lol
register(Branch, "Branch")(Branch)
