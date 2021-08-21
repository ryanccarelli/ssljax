# TODO(gabeorlanski): Change these to reflect init files
from ssljax.augment.base import AugmentBase
# from ssljax.config import FromParams
from ssljax.core import register, FromParams
from ssljax.data import DataloaderBase, DatasetBase
from ssljax.losses import LossBase
from ssljax.models import ModelBase
from ssljax.optimizers import OptimizerBase
from ssljax.train import MeterBase, SchedulerBase
from ssljax.tasks.task import TaskBase


@register(TaskBase, "ssltask")
class SSLTask(TaskBase):
    def _get_optimizer(self) -> OptimizerBase:
        return fromParams(OptimizerBase, self.config.pop("optimizer"))

    def _get_dataloader(self) -> DataloaderBase:
        return fromParams(DataloaderBase, self.config.pop("dataloader"))

    def _get_scheduler(self) -> SchedulerBase:
        return fromParams(SchedulerBase, self.config.pop("scheduler"))

    def _get_loss(self) -> LossBase:
        return fromParams(LossBase, self.config.pop("loss"))

    def _get_meter(self) -> MeterBase:
        return fromParams(MeterBase, self.config.pop("metrics"))

    def _get_model(self) -> ModelBase:
        return fromParams(ModelBase, self.config.pop("model"))

    def _get_augment(self) -> AugmentBase:
        return fromParams(AugmentBase, self.config.pop("augment"))