from flax import linen as nn


class Model(nn.Module):
    """
    Base class for a ssljax model used by ``ssljax.core.utils.register``.
    Wraps flax.linen.Module.
    """

    def __call__(self, **args):
        raise NotImplementedError
