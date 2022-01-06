from chex import assert_equal_shape
from ssljax.models import MultiCrop
from flax.training import train_state
import optax
import flax.optim
import pytest
import jax.random
import jax.numpy as jnp
from typing import Any


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

@pytest.mark.gpu
class TestMulticrop:
    def test_byol_setup(self, multi_resolution_data_list, minimal_vit):
        # all images
        key = jax.random.PRNGKey(0)
        k1, k2, _ = jax.random.split(key, num=3)
        x = jnp.ones((1, 224, 224, 3), jnp.float16)
        multicrop = MultiCrop(minimal_vit)
        params = multicrop.init(k1, x, train=False)
        print(params)
        tx = optax.sgd(
            learning_rate=0.1,
            momentum=0.1,
            nesterov=True,
        )
        state = TrainState.create(
            apply_fn=multicrop.apply,
            params=params["params"],
            tx=tx,
            batch_stats=params["batch_stats"],
            dynamic_scale=None,
        )
        multicrop_out, state = state.apply_fn(
            {"params": params["params"], "batch_stats": state.batch_stats},
            x,
            mutable=["batch_stats"],
            rngs={"dropout": k1},
        )

        # single image
        singlecrop = minimal_vit
        params = singlecrop.init(k2, x)
        tx = optax.sgd(
            learning_rate=0.1,
            momentum=0.1,
            nesterov=True,
        )
        state = TrainState.create(
            apply_fn=singlecrop.apply,
            params=params["params"],
            tx=tx,
            batch_stats=params["batch_stats"],
            dynamic_scale=None,
        )
        singlecrop_out, state = state.apply_fn(
            {"params": params["params"], "batch_stats": state.batch_stats},
            x,
            mutable=["batch_stats"],
        )

        assert_equal_shape(singlecrop_out, multicrop_out)
