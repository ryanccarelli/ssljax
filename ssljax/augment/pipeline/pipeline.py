# base class for augmentations here

# augmentations will be used by the trainer
import jax
import jax.numpy as jnp

from ssljax.augment.augmentation.augmentation import Augmentation, AugmentationDistribution


class Pipeline(Augmentation):
    """
    A Pipeline is a composition of AugmentationDistribution.

    Args:
        augmentations (list): sequence of AugmentationDistribution to be sampled in sequence
    """

    def __init__(self, augmentation_distributions):
        assert all([isinstance(t, AugmentationDistribution) for t in augmentation_distributions]), (
            f"all elements in input list must be of"
            f" type ssljax.augment.AugmentationDistribution"
        )
        self.pipeline = augmentation_distributions

    def __len__(self):
        return len(self.pipeline)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.pipeline:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def __call__(self, x, rng):
        # assert isinstance(x, jnp.array), f"argument of type {type(x)} must be __."
        for aug_distribution in self.pipeline:
            rng, _ = jax.random.split(rng)
            aug = aug_distribution.sample(rng)
            x = aug(x, rng)
        return x

    def save(self, path):
        """
        save pipeline to disk

        Args:
            path (str): save path on disk
        """
        raise NotImplementedError
