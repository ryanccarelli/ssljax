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
    dynamic_scale: flax.optim.DynamicScale

@pytest.mark.gpu
class TestMulticrop:
    def test_byol_setup(self, multi_resolution_data_list, minimal_vit):
        # all images
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        x = jnp.ones((1, 224, 224, 3), jnp.float16)
        multicrop = MultiCrop(minimal_vit)
        init_rngs = {"params": k1, "dropout": k2}
        params = multicrop.init(init_rngs, x, train=True)
        multicrop_outs = multicrop.apply(params, x, rngs={"dropout": jax.random.PRNGKey(2)})

        # single image
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        singlecrop = minimal_vit
        init_rngs = {"params": k1, "dropout": k2}
        params = singlecrop.init(init_rngs, x, train=True)
        singlecrop_outs = singlecrop.apply(params, x, rngs={"dropout": jax.random.PRNGKey(2)})

        assert_equal_shape([singlecrop_outs, multicrop_outs])
