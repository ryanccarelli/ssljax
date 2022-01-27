import jax
import jax.numpy as jnp
from ssljax.augment.augmentation.augmentation import Augmentation
from ssljax.core import get_from_register
from omegaconf import DictConfig
from functools import partial


class Pipeline(Augmentation):
    """
    A Pipeline is a composition of Augmentations.

    Args:
        config (DictConfig): config at pipelines.branches.i where i is pipeline index
    """

    def __init__(self, config):
        aug_list = [
            get_from_register(Augmentation, x)(**y.params) for x, y in config.items()
        ]

        assert all([isinstance(t, Augmentation) for t in aug_list]), (
            f"all elements in input list must be of"
            f" type ssljax.augment.Augmentation"
        )
        self.pipeline = aug_list

    def __len__(self):
        return len(self.pipeline)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.pipeline:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def __call__(self, x, rng):
        if not isinstance(x, list):
            x = [x]
        for aug in self.pipeline:
            rng, _ = jax.random.split(rng)
            aug = partial(aug, rng=rng)
            x = list(map(aug, x))
        x = list(map(lambda v: jax.lax.stop_gradient(v), x))

        return x[0] if len(x) == 1 else x
