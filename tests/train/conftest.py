import flax.linen as nn
import pytest
from hydra import compose, initialize
from ssljax.models.branch.branch import Branch
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.augment.augmentation.augmentation import Augmentation, AugmentationDistribution
from ssljax.core.utils import register


@pytest.fixture
def cputestconfig():
    initialize(config_path="../train/conf")
    cfg = compose(config_name="cpu_conf.yaml")
    return cfg


@register(Branch, "CPUOnlineBranch")
class CPUOnlineBranch(Branch):
    def setup(self):
        self.linear = nn.Dense(1)

    @nn.compact
    def __call__(self, x):
        x = self.linear(x)
        x = self.linear(x)
        return x


@register(Branch, "CPUTargetBranch")
class CPUTargetBranch(Branch):
    def setup(self):
        self.linear = nn.Dense(1)

    @nn.compact
    def __call__(self, x):
        x = self.linear(x)
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
