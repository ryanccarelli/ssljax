# test ssljax/models/resnet.py
from typing import Any

import flax.optim
import jax.numpy as jnp
import jax.random
import optax
import pytest
from chex import assert_rank
from flax.training import train_state
from ssljax.models.resnet import ResNet
from omegaconf import OmegaConf


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale


# popular named resnet configurations
@pytest.mark.gpu
@pytest.mark.parametrize(
    "num_outputs",
    [None, 10],
)
@pytest.mark.parametrize("num_filters", [64])
@pytest.mark.parametrize("num_layers", [5, 8, 9, 11, 14, 18, 26, 34, 50, 101, 152, 200])
class TestResnet:
    def test_setup_call(self, num_outputs, num_filters, num_layers):
        resnet = ResNet(
            OmegaConf.create(
                {
                    "num_outputs":num_outputs,
                    "num_filters":num_filters,
                    "num_layers":num_layers,
                }
            )
        )
        key = jax.random.PRNGKey(0)
        k1, _ = jax.random.split(key)
        x = jnp.ones((1, 224, 224, 3), jnp.float16)
        params = resnet.init(k1, x)
        tx = optax.sgd(
            learning_rate=0.1,
            momentum=0.1,
            nesterov=True,
        )
        state = TrainState.create(
            apply_fn=resnet.apply,
            params=params["params"],
            tx=tx,
            batch_stats=params["batch_stats"],
            dynamic_scale=None,
        )
        out, state = state.apply_fn(
            {"params": params["params"], "batch_stats": state.batch_stats},
            x,
            mutable=["batch_stats"],
        )
        # assertions
        if num_outputs:
            assert out.shape[1] == num_outputs
        else:
            assert isinstance(out, dict)
