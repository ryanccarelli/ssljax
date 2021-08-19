# base class for augmentations here

# augmentations will be used by the trainer
import jax.numpy as jnp


class Pipeline(Augmentation):
    """
    A Pipeline is a composition of Augmentations.
    It is often necessary to sample augmentations and generate a
    new pipeline on each forward pass.

    Args:
        augmentations (list, dict): sequence of Augmentation to be composed. If
    """

    def __init__(self, augmentations, mode="deterministic", rng=False):
        assert all([isinstance(t, Augmentation) for t in augmentations]), (
            f"All elements in input list must be of"
            f" type ssljax.augment.Augmentation"
        )
        self.pipeline = augmentations
        self.mode = mode
        self.rng = rng

    def __len__(self):
        return len(self.pipeline)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.pipeline:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def __call__(self, x):
        assert isinstance(x, jnp.array), f"argument of type {type(x)} must be __."
        if mode == "deterministic":
            for t in self.pipeline:
                rng, _ = jax.random.split(self.rng)
                x = t(x, rng)
            return x
        elif mode == "withoutreplacement":
            raise NotImplementedError
        elif mode == "withreplacement":
            raise NotImplementedError
        else:
            raise KeyError(f"mode is {mode} but must be in \{'deterministic','withreplacement','withoutreplacement'\}"

    def sample(cls, rng):
        """
        Sample a pipeline from elements of self.pipeline.

        Args:
            rng (jnp.ndarray): jax rng
        """
        # should overwrite call
        raise NotImplementedError

    def save(self, path):
        """
        save pipeline to disk

        Args:
            path (str): save path on disk
        """
        raise NotImplementedError


# TODO: implement a class to sample augmentations
