# TODO(gabeorlanski): Change these to reflect init files
from ssljax.augment.base import Augment
# from ssljax.config import FromParams
from ssljax.core import FromParams, register
from ssljax.data import Dataloader, Dataset
from ssljax.losses import Loss
from ssljax.models import Model
from ssljax.optimizers import Optimizer
from ssljax.tasks.task import Task
from ssljax.train import Meter, Scheduler


@register(Task, "ssltask")
class SSLTask(Task):
    def _get_optimizer(self) -> Optimizer:
        return fromParams(Optimizer, self.config.pop("optimizer"))

    def _get_dataloader(self) -> Dataloader:
        return fromParams(Dataloader, self.config.pop("dataloader"))

    def _get_scheduler(self) -> Scheduler:
        return fromParams(Scheduler, self.config.pop("scheduler"))

    def _get_loss(self) -> Loss:
        return fromParams(Loss, self.config.pop("loss"))

    def _get_meter(self) -> Meter:
        return fromParams(Meter, self.config.pop("metrics"))

    def _get_model(self) -> Model:
        return fromParams(Model, self.config.pop("model"))

    def _get_augment(self) -> Augment:
        return fromParams(Augment, self.config.pop("augment"))
