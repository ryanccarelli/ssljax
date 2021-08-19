# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging

from ssljax.augment import AugmentBase
from ssljax.config import FromParams
from ssljax.core import register
from ssljax.data import DataloaderBase, DatasetBase
from ssljax.loss import LossBase
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

    def _get_optimizer(self):
        raise NotImplementedError

    def _get_dataset(self):
        raise NotImplementedError

    def _get_dataloader(self):
        raise NotImplementedError

    def _get_optimizer(self):
        raise NotImplementedError

    def _get_scheduler(self):
        raise NotImplementedError

    def _get_loss(self):
        raise NotImplementedError

    def _get_meter(self):
        raise NotImplementedError

    def _get_model(self):
        raise NotImplementedError

    def _get_augment(self):
        raise NotImplementedError


@register(TaskBase, "ssltask")
class SSLTask(TaskBase, FromParams):
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

    def _get_optimizer(self):
        return fromParams(Optimizer, self.config.pop("optimizer"))

    def _get_dataset(self):
        return fromParams(DatasetBase, self.config.pop("dataset"))

    def _get_dataloader(self):
        return fromParams(DataloaderBase, self.config.pop("dataloader"))

    def _get_optimizer(self):
        return fromParams(OptimizerBase, self.config.pop("optimizer"))

    def _get_scheduler(self):
        return fromParams(SchedulerBase, self.config.pop("scheduler"))

    def _get_loss(self):
        return fromParams(LossBase, self.config.pop("loss"))

    def _get_meter(self):
        return fromParams(MeterBase, self.config.pop("meter"))

    def _get_model(self):
        return fromParams(ModelBase, self.config.pop("model"))

    def _get_augment(self):
        return fromParams(AugmentBase, self.config.pop("augment"))
