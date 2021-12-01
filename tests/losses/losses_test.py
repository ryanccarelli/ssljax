import jax.numpy as jnp
import pytest
from chex import assert_rank
from ssljax.losses.byol import byol_regression_loss, byol_softmax_cross_entropy
from ssljax.losses.moco import moco_infonce_loss


class TestLosses:
    @pytest.mark.parametrize(
        "fn", [moco_infonce_loss, byol_regression_loss, byol_softmax_cross_entropy]
    )
    def test_returns(self, fn):
        # (embedding, batch)
        outs = {"0": {"0": jnp.ones((10, 10))}, "1": {"1": jnp.ones((10, 10))}}
        loss = fn(outs)
        # reduction transforms rank 1 -> 0
        assert_rank(loss, 0)
