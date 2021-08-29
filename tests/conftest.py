import flax.linen as nn
import pytest
from ssljax.models.branch.branch import Branch
from ssljax.models.sslmodel import SSLModel


@pytest.fixture
def mocksslmodel():
    return MockSSLModel()


class MockSSLModel(SSLModel):
    def setup(self):
        self.branches = [LinearBranch(), LinearBranch()]


class LinearBranch(Branch):
    def setup(self):
        self.dense = nn.Dense(features=10)

    @nn.compact
    def __call__(self, x):
        x = self.dense(x)
        return x


class IdBranch(Branch):
    def setup(self):
        pass

    @nn.compact
    def __call_(self, x):
        return x
