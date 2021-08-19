# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging

# TODO(gabeorlanski): Change these to reflect init files
from ssljax.augment.base import AugmentBase
from ssljax.config import Config
from ssljax.data import DataloaderBase
from ssljax.losses import LossBase
from ssljax.models import ModelBase
from ssljax.optimizers import OptimizerBase
from ssljax.train import MeterBase, SchedulerBase

logger = logging.getLogger(__name__)


class TaskBase:
    """
    Abstract class for a task.

    A task constructs and holds:
        - model
        - optimizer
        - dataset
        - dataloader
        - loss
        - meters
        - augmentations
        - scheduler
            - lr

    Args:
        config (Config): The config to get parameters from.


    """

    def __init__(self, config:Config):

        super().__init__()
        self.config = config
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.dataloader = self._get_dataloader()
        self.meter = self._get_meter()
        self.augment = self._get_augment()

    #####################################################################
    # Functions the children class must implement                       #
    #####################################################################
    def _get_optimizer(self) -> OptimizerBase:
        """
        Initialize the optimizer. This must be implemented by child tasks.

        Returns (OptimizerBase): The optimizer to use for the task.
        """
        raise NotImplementedError()

    def _get_dataloader(self) -> DataloaderBase:
        """
        Initialize the dataloader. This must be implemented by child tasks.

        Returns (DataloaderBase): The dataloader to use for the task.
        """
        raise NotImplementedError()

    def _get_scheduler(self) -> SchedulerBase:
        """
        Initialize the scheduler. This must be implemented by child tasks.

        Returns (SchedulerBase): The scheduler to use for the task.
        """
        raise NotImplementedError()

    def _get_loss(self) -> LossBase:
        """
        Initialize the loss calculator. This must be implemented by child tasks.

        Returns (LossBase): The loss to use for the task.
        """
        raise NotImplementedError()

    def _get_meter(self) -> MeterBase:
        """
        Initialize the metrics. This must be implemented by child tasks.

        Returns (MeterBase): The metrics to use for the task.
        """
        raise NotImplementedError()

    def _get_model(self) -> ModelBase:
        """
        Initialize the model for this task. This must be implemented by child
        tasks.

        Returns (ModelBase): The model to use for this task.
        """
        raise NotImplementedError()

    def _get_augment(self) -> AugmentBase:
        """
       Initialize the augment for this task. This must be implemented by child
       tasks.

       Returns (AugmentBase): The augment to use for this task.
       """
        raise NotImplementedError()
    
