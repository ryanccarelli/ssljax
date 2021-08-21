import logging

logger = logging.getLogger(__name__)

logger.error(f"{__name__}: THESE ARE PLACEHOLDERS!!")


class LossBase:
    """
    Base class for loss function.
    """

    def __init__(self, lossfunction):
        self.lossfunction = lossfunction

    def __call__(self, x):
        return self.lossfunction(x)


def lossbase(lossfunction):
    """
    lossbase is a decorator that wraps loss functions in the
    LossBase class.
    """

    lossclass = LossBase(lossfunction)
    return lossclass
