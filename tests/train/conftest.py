import flax.linen as nn
import pytest
from hydra import compose, initialize
from ssljax.models.branch import Branch


@pytest.fixture
def cputestconfig():
    initialize(config_path="../train/conf/null_conf")
    cfg = compose(config_name="config")
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
