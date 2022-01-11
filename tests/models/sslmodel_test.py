# test ssljax/models/sslmodel.py

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape
from flax.training import train_state
from jax.tree_util import tree_leaves
from omegaconf import DictConfig, OmegaConf
from ssljax.models.branch.branch import Branch
from ssljax.models.sslmodel import SSLModel


# tests
class TestSSLModel:
    def test_setup_call(self, mocksslmodel):
        key = jax.random.PRNGKey(0)
        k1, k2, k3, _ = jax.random.split(key, 4)
        x = {
            "0": jnp.array(jax.random.normal(k1, (2000,))),
            "1": jnp.array(jax.random.normal(k2, (2000,))),
        }
        params = mocksslmodel.init(k3, x)

        # assert params correctly index branches
        # internals of branches tested in branch_test.py
        assert all(isinstance(x, jnp.ndarray) for x in tree_leaves(params))

        # assert that outs has the correct structure
        # {branch: {pipeline: array, ...}, ...}
        out = mocksslmodel.apply(params, x)
        assert all(x in out for x in ["0", "1"])
        assert "0" in out["0"]
        assert "0" not in out["1"]
        assert "1" in out["1"]
        assert "1" not in out["0"]
        assert all(isinstance(x, jnp.ndarray) for x in tree_leaves(out))
        for x in tree_leaves(out):
            assert_shape(x, (10,))

    def test_detach(self, mocksslmodel):
        pass


# fixtures
@pytest.fixture
def mocksslmodel():
    return MockSSLModel(config=OmegaConf.create())


# utilities
class MockSSLModel(SSLModel):
    config: DictConfig

    def setup(self):
        self.branches = {
            "0": LinearBranch(stages={}),
            "1": LinearBranch(stages={}),
        }
        self.pipelines = {"0": ["0"], "1": ["1"]}


class LinearBranch(Branch):
    def setup(self):
        self.dense = nn.Dense(features=10)

    @nn.compact
    def __call__(self, x):
        x = self.dense(x)
        return x
