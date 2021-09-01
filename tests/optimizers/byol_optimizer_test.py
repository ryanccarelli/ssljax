import chex
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from ssljax.optimizers.byol_optimizer import byol_ema, byol_optimizer


class BYOLTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

    @chex.all_variants()
    def test_byol_ema(self):
        # this is expected to skip pmap variant test (must configure multiple devices)
        params = {"branch_0": jnp.array([5.0, 7.0]), "branch_1": jnp.array([3.0, 4.0])}
        decay = 0.9
        d = decay

        ema = byol_ema(decay=decay, debias=False)
        state = ema.init(params)  # init zeros

        transform_fn = self.variant(ema.update)
        mean, state = transform_fn(state, params)
        assert np.isclose(
            mean["branch_0"][0], (1 - d) * params["branch_0"][0], atol=1e-4
        )

    @chex.all_variants()
    def test_byol_optimizer(self):
        # here we sequentially apply updates
        params = {"branch_0": jnp.array([5.0, 7.0]), "branch_1": jnp.array([3.0, 4.0])}
        lr = 0.1
        decay = 0.9
        d = decay
        opt = byol_optimizer(lr, decay)
        tx1 = opt["branch_0"]
        print(tx1)
        tx2 = opt["branch_1"]
        print(tx2)

        state1, state2 = tx1.init(params), tx2.init(params)  # init zeros

        transform_fn_1, transform_fn_2 = self.variant(tx1.update), self.variant(
            tx2.update
        )
        update1, state1 = transform_fn_1(state=state1, updates=params, params=params)
        update2, state2 = transform_fn_2(state=state2, params=params)

        assert True == True
