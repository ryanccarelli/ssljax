from flax import linen as nn
from ssljax.core.utils import Registrable


class Model(Registrable, nn.Module):
    """
    Abstract class for a ssljax  model.
    """

    def __call__(self, **args):
        raise NotImplementedError
