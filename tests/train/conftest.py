import flax.linen as nn
import hydra
import pytest
from hydra import compose, initialize
from ssljax.augment.augmentation.augmentation import (Augmentation,
                                                      AugmentationDistribution)
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.core.utils import register
from ssljax.models.branch.branch import Branch


@pytest.fixture
def cputestconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="cpu_conf.yaml")
    return cfg


@register(Branch, "CPUOnlineBranch")
class CPUOnlineBranch(Branch):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x


@register(Branch, "CPUTargetBranch")
class CPUTargetBranch(Branch):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x


class Identity(Augmentation):
    """
    Map image by identity.
    """

    def __call__(self, x, rng):
        return x


@register(Pipeline, "CPUPipeline")
class CPUPipeline(Pipeline):
    def __init__(self):
        super().__init__([AugmentationDistribution([Identity()])])
