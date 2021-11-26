# we rely on Optax https://github.com/google-research/scenic
# to maintain:
# adabelief, adagrad, adam, adamw, dpsgd, fromage, lamb, noisy_sgd,
# radam, rmsprop, sgd, yogi, lars, zerog
import jax.numpy as jnp
import pytest
from optax._src import update
from ssljax.optimizers.optimizers import zerog


# following tests in optax
@pytest.mark.parametrize("opt", [zerog])
class TestOptimizers:
    # this test relies on initial_params remaining fixed
    # to test other optimizers see
    # https://github.com/deepmind/optax/blob/ab298f39a57c72ed03b45698e7f2d0d101ebbd71/optax/_src/alias_test.py#L44
    def test_zerog(self, opt):
        op = opt()
        initial_params = jnp.array([-1.0, 10.0, 1.0])
        updates = jnp.array([10.0, 2.0, 3.0])
        state = op.init(initial_params)
        updates, state = op.update(updates, state, initial_params)
        params = update.apply_updates(initial_params, updates)
        assert jnp.array_equal(params, initial_params)
