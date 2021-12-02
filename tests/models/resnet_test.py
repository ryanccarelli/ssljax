# test ssljax/models/resnet.py
from typing import Any

import jax.numpy as jnp
import jax.random
import pytest
from chex import assert_rank
from ssljax.models.resnet import ResNet
import optax
from flax.training import train_state
import flax.optim

class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

# popular named resnet configurations
@pytest.mark.parametrize(
    "stage_sizes,block_cls_name",
    [
        ([2, 2, 2, 2], "ResNetBlock"),
        ([3, 4, 6, 3], "ResNetBlock"),
        ([3, 4, 6, 3], "BottleneckResNetBlock"),
        ([3, 4, 23, 3], "BottleneckResNetBlock"),
        ([3, 8, 36, 3], "BottleneckResNetBlock"),
        ([3, 24, 36, 3], "BottleneckResNetBlock"),
    ],
)
@pytest.mark.parametrize("num_filters", [64])
@pytest.mark.parametrize("num_classes", [None, 10])
class TestResnet:
    def test_setup_call(self, stage_sizes, block_cls_name, num_filters, num_classes):
        resnet = ResNet(
            stage_sizes=stage_sizes,
            num_classes=num_classes,
            num_filters=num_filters,
            block_cls_name=block_cls_name,
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
        out, state = state.apply_fn({"params": params["params"], "batch_stats": state.batch_stats}, x, mutable=["batch_stats"])
        # assertions
        assert len(out.shape) == 2
