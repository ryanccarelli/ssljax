import jax
import jax.numpy as jnp
from ssljax.augment.augmentation.augmentation import (Augmentation,
                                                      AugmentationDistribution)
from ssljax.core.utils import get_from_register


class Pipeline(Augmentation):
    """
    A Pipeline is a composition of AugmentationDistribution.

    Args:
        config (hydra.OmegaConf): config at pipelines.branches.i where i is branch index
    """

    def __init__(self, config):
        aug_dist_list = [
            AugmentationDistribution([get_from_register(Augmentation, x)(**y.params)])
            for x, y in config.items()
        ]

        assert all([isinstance(t, AugmentationDistribution) for t in aug_dist_list]), (
            f"all elements in input list must be of"
            f" type ssljax.augment.AugmentationDistribution"
        )
        self.pipeline = aug_dist_list

    def __len__(self):
        return len(self.pipeline)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.pipeline:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def __call__(self, x, rng):
        for aug_distribution in self.pipeline:
            rng, _ = jax.random.split(rng)
            aug = aug_distribution.sample(rng)
            x = aug(x, rng)
        x = jax.lax.stop_gradient(x)
        return x
