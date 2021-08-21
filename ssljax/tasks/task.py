# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging

# TODO(gabeorlanski): Change these to reflect init files
from ssljax.augment.base import Pipeline
from ssljax.config import Config
from ssljax.data import Dataloader
from ssljax.losses import Loss
from ssljax.models import Model
from ssljax.optimizers import Optimizer
from ssljax.train import Meter, Scheduler

logger = logging.getLogger(__name__)


class Task:
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

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.meter = self._get_meter()
        self.pipeline = self._get_pipeline()
        self.dataloader = self._get_dataloader()

    # Functions the children class must implement
    def _get_model(self) -> Model:
        """
        Initialize the model for this task. This must be implemented by child
        tasks.

        Returns (ModelBase): The model to use for this task.
        """
        raise NotImplementedError()

    def _get_loss(self) -> Loss:
        """
        Initialize the loss calculator. This must be implemented by child tasks.

        Returns (LossBase): The loss to use for the task.
        """
        raise NotImplementedError()

    def _get_optimizer(self) -> Optimizer:
        """
        Initialize the optimizer. This must be implemented by child tasks.

        Returns (OptimizerBase): The optimizer to use for the task.
        """
        raise NotImplementedError()

    def _get_scheduler(self) -> Scheduler:
        """
        Initialize the scheduler. This must be implemented by child tasks.

        Returns (SchedulerBase): The scheduler to use for the task.
        """
        raise NotImplementedError()

    def _get_meter(self) -> Meter:
        """
        Initialize the metrics. This must be implemented by child tasks.

        Returns (MeterBase): The metrics to use for the task.
        """
        raise NotImplementedError()

    def _get_pipeline(self) -> Pipeline:
        """
        Initialize the augment for this task. This must be implemented by child
        tasks.

        Returns (AugmentBase): The augment to use for this task.
        """
        raise NotImplementedError()

    def _get_dataloader(self) -> Dataloader:
        """
        Initialize the dataloader. This must be implemented by child tasks.

        Returns (DataloaderBase): The dataloader to use for the task.
        """
        raise NotImplementedError()
