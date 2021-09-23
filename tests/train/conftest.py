import flax.linen as nn
import pytest
from hydra import compose, initialize
from ssljax.augment.augmentation.augmentation import (Augmentation,
                                                      AugmentationDistribution)
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.core.utils import register
from ssljax.models.branch.branch import Branch


@pytest.fixture
def cputestconfig():
    initialize(config_path="../train/conf")
    cfg = compose(config_name="cpu_conf.yaml")
    return cfg


@register(Branch, "CPUOnlineBranch")
class CPUOnlineBranch(Branch):
    @nn.compact
    def __call__(self, x):
        print("online1", x.shape)
        x = nn.Dense(1)(x)
        print("online2", x.shape)
        x = nn.Dense(1)(x)
        print("online3", x.shape)
        return x


@register(Branch, "CPUTargetBranch")
class CPUTargetBranch(Branch):
    def setup(self):
        self.linear = nn.Dense(1)

    def __call__(self, x):
        print("target1", x.shape)
        x = self.linear(x)
        print("target2", x.shape)
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
