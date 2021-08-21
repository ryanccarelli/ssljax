from ssljax.core.utils import utils
import random
import numpy as np
import jax


def test_prepare_environment():
    rng = utils.prepare_environment({})

    # Generate first values
    before_random_value = random.randint(0, 1000)
    before_np_val = np.random.randint(0, 1000)
    rng, before_key = jax.random.split(rng)
    before_jax_value = jax.random.uniform(rng)

    # Second prepare environment
    rng = utils.prepare_environment({})

    # Generate Second values
    after_random_value = random.randint(0, 1000)
    after_np_val = np.random.randint(0, 1000)
    rng, after_key = jax.random.split(rng)
    after_jax_value = jax.random.uniform(rng)

    assert before_random_value == after_random_value
    assert before_np_val == after_np_val
    assert (before_key == after_key).all()
    assert before_jax_value == after_jax_value
