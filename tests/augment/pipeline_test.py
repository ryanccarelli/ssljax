import pytest
import jax.numpy as jnp
import jax.random
from ssljax.augment.pipeline.pipeline import Pipeline
from omegaconf import DictConfig


@pytest.fixture
def image():
    return jnp.ones((1, 224, 244, 3), dtype=jnp.float32)

class TestPipeline:
    def test_init_call(self, image):
        config = DictConfig({"Identity": {"params":{}}, "Identity": {"params":{}}})
        pipe = Pipeline(config)
        rng = jax.random.PRNGKey(0)
        outs = pipe(x=image, rng=rng)
        assert jnp.equal(outs, image).all()
