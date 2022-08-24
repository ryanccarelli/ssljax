# test ssljax/models/mlp.py

import jax.numpy as jnp
import jax.random
import pytest
from chex import assert_shape
from ssljax.models.mlp import MLP
from omegaconf import OmegaConf


# tests
@pytest.mark.parametrize("layer_dims", [[1, 1], [1]])
@pytest.mark.parametrize("batch_norm", [True, False])
# TODO: use_running_average=False case fails with
# mutability complaint
@pytest.mark.parametrize("batch_norm_params", [{"use_running_average": True}])
@pytest.mark.parametrize("activation_name", ["relu", "gelu"])
@pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("dtype", ["float32"])
class TestMLP:
    def test_setup_call(
        self,
        layer_dims,
        batch_norm,
        batch_norm_params,
        activation_name,
        dropout_prob,
        dtype,
    ):
        mlp = MLP(
            OmegaConf.create(
                {
                    "layer_dims":layer_dims,
                    "batch_norm":batch_norm,
                    "batch_norm_params":batch_norm_params,
                    "activation_name":activation_name,
                    "dropout_prob":dropout_prob,
                    "dtype":dtype,
                }
            )
        )
        key = jax.random.PRNGKey(0)
        k1, _ = jax.random.split(key)
        x = jnp.ones((100,))
        params = mlp.init(k1, x)
        out = mlp.apply(params, x)
        # assertions
        assert_shape(out, (1,))
