import jax.numpy as jnp
import pytest
from chex import assert_rank
from ssljax.losses.byol import cosine_similarity, cross_entropy
from ssljax.losses.moco import infonce_loss
from ssljax.losses.dino import dino_loss


class TestLosses:
    @pytest.mark.parametrize(
        "fn", [infonce_loss, cosine_similarity, cross_entropy]
    )
    def test_returns(self, fn):
        # (embedding, batch)
        outs = {
            "0": {"0": {"head": jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}, "1": {"head":jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}},
            "1": {"0": {"head": jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}, "1": {"head":jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}},
        }
        loss = fn(outs)
        # reduction transforms rank 1 -> 0
        assert_rank(loss, 0)

    # separate because must schedule tau_t
    # TODO: combine with test_returns
    def test_dino(self):
        outs = {
            "0": {"0": {"head": jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}, "1": {"head":jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}},
            "1": {"0": {"head": jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}, "1": {"head":jnp.ones((10, 10)), "pred": jnp.ones((10, 10)), "proj": jnp.ones((10, 10))}},
        }
        loss = dino_loss(outs, tau_t=0.4)
        # reduction transforms rank 1 -> 0
        assert_rank(loss, 0)
