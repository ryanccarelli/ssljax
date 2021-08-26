from flax import linen as nn
from ssljax.core.utils import register


class Model(nn.Module):
    """
    Base class for a ssljax model.
    """

    def __call__(self, **args):
        raise NotImplementedError


class Branch(Model):
    """
    Base class for a ssljax branch.
    A branch holds explicit references to parameter groups.
    """

    def setup(self, **args):
        pass
