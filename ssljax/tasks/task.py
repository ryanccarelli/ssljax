# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging

# TODO(gabeorlanski): Change these to reflect init files
from ssljax.augment.base import AugmentBase
# from ssljax.config import FromParams
from ssljax.core import register, FromParams
from ssljax.data import DataloaderBase, DatasetBase
from ssljax.losses import LossBase
from ssljax.models import ModelBase, SSLBase
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
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.dataset = self._get_dataset()
        self.dataloader = self._get_dataloader()
        self.meter = self._get_meter()
        self.augment = self._get_augment()

    def _get_optimizer(self) -> OptimizerBase:
        """

        :return:
        """
        raise NotImplementedError()

    # def _get_dataset(self) -> DatasetBase:
    #     raise NotImplementedError()

    def _get_dataloader(self) -> DataloaderBase:
        raise NotImplementedError()

    def _get_scheduler(self) -> SchedulerBase:
        raise NotImplementedError()

    def _get_loss(self) -> LossBase:
        raise NotImplementedError()

    def _get_meter(self) -> MeterBase:
        raise NotImplementedError()

    def _get_model(self) -> ModelBase:
        raise NotImplementedError()

    def _get_augment(self) -> AugmentBase:
        raise NotImplementedError()
