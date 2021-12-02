# test ssljax/models/branch/branch.py

import jax.numpy as jnp
import jax.random
import pytest
from chex import assert_rank
from omegaconf import DictConfig
# TODO: fix this
from ssljax.models.branch.branch import Branch


# how to mock config here?
#
@pytest.mark.parametrize("stop_gradient", [True, False])
class TestBranch:
    def test_setup_call(self, stop_gradient):
        config = DictConfig(
            {
                "stop_gradient": True,
                "body": {"module": "MLP", "params": {"layer_dims": [1]}},
                "head": {"module": "MLP", "params": {"layer_dims": [1]}},
            }
        )
        branch = Branch(config=config)
        key = jax.random.PRNGKey(0)
        k1, _ = jax.random.split(key)
        x = jnp.ones((20,), jnp.float16)
        params = branch.init(k1, x)
        out = branch.apply(params, x)
        assert_rank(out, 1)
