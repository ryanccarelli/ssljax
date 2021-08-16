# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
from ssljax.losses import Loss
from ssljax.models import SSLBase
from ssljax.optimizers import Optimizer


class TaskBase(Registrable):
    """
    Abstract class for a task.

    A task prepares and holds
        - optimizer
        - dataset
        - dataloader
        - loss
        - meters
        - augmentations
    """
    pass

@Task.register("ssltask")
class SSLTask(TaskBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = None
        self.optimizer = None
        self.dataset = None
        self.dataloader = None
        self.meter = None
        self.augment = None

    @classmethod
    def from_params(cls, config):
        raise NotImplementedError

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
        SSLBase.from_params(self.config.pop("model"))
