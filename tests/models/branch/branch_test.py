# test ssljax/models/branch/branch.py

import jax.numpy as jnp
import jax.random
import pytest
from chex import assert_rank
from omegaconf import DictConfig
# TODO: fix this
from ssljax.core import get_from_register
from ssljax.models.model import Model
from ssljax.models.branch.branch import Branch


# how to mock config here?
@pytest.mark.parametrize("stop_gradient", [True, False])
class TestBranch:
    def test_setup_call(self, stop_gradient):
        config = DictConfig(
            {
                "body": {"name": "MLP", "params": {"layer_dims": [1], "activation_name": "relu", "dtype": "float32"}},
                "head": {"name": "MLP", "params": {"layer_dims": [1], "activation_name": "relu", "dtype": "float32"}},
            }
        )
        modules = {}
        for module_key, module_params in config.items():
            modules[module_key] = get_from_register(Model, module_params.name)(module_params.params, name=module_key)
        branch = Branch(stages=modules, stop_gradient=stop_gradient)
        key = jax.random.PRNGKey(0)
        k1, _ = jax.random.split(key)
        x = jnp.ones((20,), jnp.float16)
        params = branch.init(k1, x)
        out = branch.apply(params, x)
        assert isinstance(out, dict)
        assert all([isinstance(key, str) for key in out.keys()])
        assert all([isinstance(val, jnp.ndarray) for val in out.values()])
        for val in out.values():
            assert_rank(val, 1)
