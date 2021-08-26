import logging
from ssljax.core.utils import register

logger = logging.getLogger(__name__)

logger.error(f"{__name__}: THESE ARE PLACEHOLDERS!!")


# TODO: should this be done instead in
# the same way as Optimizer?


class Loss:
    """
    Base class for loss function.
    """

    pass


losses = {
    # "byol_regression": regression_loss,
    # "byol_softmax_cross_entropy": byol_softmax_cross_entropy,
}

for name, func in losses.items():
    register(Loss, func)
