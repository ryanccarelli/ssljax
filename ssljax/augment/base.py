# base class for augmentations here
# augmentations will be used by the trainer
import jax.numpy as jnp


class Augmentation:
    """
    An augmentation is a function applied to images.
    """

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class Pipeline(Augmentation):
    """
    A Pipeline is a composition of Augmentations.

    Args:
        augmentlist (list): sequence of Augmentation to be composed.
            List of `ssljax.augment.Augmentation` objects
    """

    def __init__(self, augmentlist):
        assert all([isinstance(t, Augmentation) for t in augmentlist]), (
            f"All elements in input list must be of"
            f" type ssljax.augment.Augmentation"
        )
        self.pipeline = augmentlist

    def __len__(self):
        return len(self.pipeline)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.pipeline:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def __call__(self, x):
        # this function has side effects
        # modifies the tile in place, but also returns the modified tile
        # need to do this for dask distributed
        assert isinstance(x, jnp.array), f"argument of type {type(x)} must be __."
        for t in self.pipeline:
            aug = t()
            x = aug(x)
        return x

    @classmethod
    def sample(cls):
        """
        Sample a pipeline from elements of self.pipeline.

        """
        raise NotImplementedError

    def save(self, path):
        """
        save pipeline to disk

        Args:
            path (str): save path on disk
        """
        raise NotImplementedError


# TODO: implement a class to sample augmentations
