# multicrop wrapper, inspired by https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/utils.py#L594

import jax.numpy as jnp
import flax.linen as nn
from ssljax.models.model import Model

class MultiCrop(Model):
    """
    Performs one forward pass for each resolution in inputs. All inputs of each
    resolution are clubbed. Outputs returned concatenated. Note that unlike in
    the reference implementation, we do not apply the head.

    Adapted from: https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/utils.py#L594

    Image format is assumed to be (x, y, c), we check for differences in y.

    Args:
        module (nn.Module): model (typically backbone)
    """

    module: nn.Module

    @nn.compact
    def __call__(self, x, train=True):
        # x needs to be a list of jnp.ndarray
        # handle single input case
        if not isinstance(x, list):
            x = [x]
        idx_crops = jnp.cumsum(
            jnp.unique(
                jnp.array([i.shape[-2] for i in x]),
                return_counts=True,
        )[1], 0)

        start_idx = 0
        output = None
        for end_idx in idx_crops:
            _out = self.module(jnp.concatenate(x[start_idx: end_idx]), train=train)
            if output:
                output = jnp.concatenate((output, _out))
            else:
                output = _out
            start_idx = end_idx

        return output
