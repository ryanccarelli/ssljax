import pytest
import jax.numpy as jnp
import jax.random
from ssljax.augment.augmentation import RandomFlip, GaussianBlur, ColorTransform, Solarize, Clip, Identity, CenterCrop


@pytest.fixture
def image():
    return jnp.ones((1, 224, 244, 3), dtype=jnp.float32)

class TestAugment:
    @pytest.mark.parametrize("augmentation", [RandomFlip, GaussianBlur, ColorTransform, Solarize, Clip, Identity, CenterCrop])
    def test_call(self, augmentation, image):
        rng = jax.random.PRNGKey(0)
        aug = augmentation(prob=1.0)
        out = aug(x=image, rng=rng)
