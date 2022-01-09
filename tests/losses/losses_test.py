import jax.numpy as jnp
import pytest
from chex import assert_rank
from ssljax.losses.byol import cosine_similarity, cross_entropy
from ssljax.losses.moco import infonce_loss


class TestLosses:
    @pytest.mark.parametrize(
        "fn", [infonce_loss, cosine_similarity, cross_entropy]
    )
    def test_returns(self, fn):
        # (embedding, batch)
        outs = {
            "0": {"0": jnp.ones((10, 10)), "1": jnp.ones((10, 10))},
            "1": {"0": jnp.ones((10, 10)), "1": jnp.ones((10, 10))},
        }
        loss = fn(outs)
        # reduction transforms rank 1 -> 0
        assert_rank(loss, 0)
