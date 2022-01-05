from chex import assert_equal_shape
from ssljax.models import MultiCrop
from flax.training import train_state
import optax
import flax.optim
import pytest
import jax.random


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

@pytest.mark.gpu
@pytest.mark.parametrize(
    "model",
    [minimal_resnet, minimal_vit],
)
class TestMulticrop:
    def test_byol_setup(self, multi_resolution_data_list, model):
        # call on all images
        key = jax.random.PRNGKey(0)
        k1, _ = jax.random.split(key)
        x = jnp.ones((1, 224, 224, 3), jnp.float16)
        multicrop = MultiCrop(model)
        params = multicrop.init(k1, x)
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
        out, state = state.apply_fn(
            {"params": params["params"], "batch_stats": state.batch_stats},
            x,
            mutable=["batch_stats"],
        )


        # output shape fromarray call matches list of arrays call
        assert_equal_shape(
