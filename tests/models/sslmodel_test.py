import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from flax.training import train_state
from omegaconf import OmegaConf
from ssljax.models.branch.branch import Branch
from ssljax.models.sslmodel import SSLModel


class SSLModelTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.mocksslmodel = MockSSLModel(config=OmegaConf.create())

    def test_withid(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jnp.array(jax.random.normal(k1, (2000, 2)))
        params = self.mocksslmodel.init(k2, x)
        out = self.mocksslmodel.apply(params, x)
        assert len(out) == 2


class MockSSLModel(SSLModel):
    def setup(self):
        self.branch = [LinearBranch(), LinearBranch()]


class LinearBranch(Branch):
    def setup(self):
        self.dense = nn.Dense(features=10)

    @nn.compact
    def __call__(self, x):
        x = self.dense(x)
        return x
