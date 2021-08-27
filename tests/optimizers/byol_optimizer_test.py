import pytest

import chex
import jax.numpy as jnp
import numpy as np
from ssljax.optimizers.byol_optimizer import byol_optimizer, byol_ema
from absl.testing import parameterized


class BYOLTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.init_params = (jnp.array([1.,2.]), jnp.array([3.,4.]))
        self.per_step_updates = (jnp.array([500.,5.]), jnp.array([300.,3.]))

    @chex.all_variants()
    def test_byol_ema(self):
        # this is expected to skip pmap variant test (must configure multiple devices)
        values = {"branch_0":jnp.array([5.0, 7.0]), "branch_1":jnp.array([3.0, 4.0])}
        decay = 0.9
        d = decay

        ema = byol_ema(decay=decay, debias=False)
        state = ema.init(values)  # init zeros

        transform_fn = self.variant(ema.update)
        mean, state = transform_fn(values, state)
        assert np.isclose(mean["branch_0"][0], (1 - d) * values["branch_0"][0], atol=1e-4)

    @chex.all_variants()
    def test_byol_optimizer(self):
        values = {"branch_0":jnp.array([5.0, 7.0]), "branch_1":jnp.array([3.0, 4.0])}
        lr = 0.1
        decay = 0.9
        d = decay

        tx = byol_optimizer(learning_rate=lr, decay_rate=d, debias=False)
        state = tx.init(values)  # init zeros

        transform_fn = self.variant(tx.update)
        mean, state = transform_fn(values, state)

        assert 1 == 2
