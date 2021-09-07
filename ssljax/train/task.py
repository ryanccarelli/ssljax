# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging

from ssljax.augment.pipeline.pipeline import Pipeline
# from ssljax.augment.base import Pipeline
from ssljax.config import Config
from ssljax.core.utils import prepare_environment
from ssljax.core.utils.register import get_from_register, print_registry
from ssljax.data import Dataloader
from ssljax.losses.loss import Loss
from ssljax.models.model import Model
from ssljax.optimizers import Optimizer
from ssljax.train import Meter, Scheduler, SSLTrainer, Trainer

logger = logging.getLogger(__name__)


class Task:
    """
    Abstract class for a task.

    A task constructs and holds:
        - model
        - loss
        - optimizer
        - schedulers
        - meter
        - pipeline
        - dataloader
        - trainer

    Args:
        config (Config): The config to get parameters from.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.rng = prepare_environment(self.config)
        self.trainer = self._get_trainer()
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.meter = self._get_meter()
        self.pipelines = self._get_pipeline()
        self.dataloader = self._get_dataloader()

    def _get_trainer(self) -> Trainer:
        """
        Initialize the trainer for this task.
        """
        trainer = get_from_register(Trainer, self.config.trainer.name)
        return trainer(rng=self.rng, task=self)

    def _get_model(self) -> Model:
        """
        Initialize the model for this task. This must be implemented by child
        tasks.

        Returns (Model): The model to use for this task.
        """
        return get_from_register(Model, self.config.model.name)

    def _get_loss(self) -> Loss:
        """
        Initialize the loss calculator. This must be implemented by child tasks.

        Returns (Loss): The loss to use for the task.
        """
        print_registry()
        return get_from_register(Loss, self.config.loss)

    def _get_optimizer(self) -> Optimizer:
        """
        Initialize optimizer.

        Returns (Optimizer): The optimizers to use for the task.
        """
        return get_from_register(Optimizer, self.config.optimizer.name)

    def _get_schedulers(self) -> dict(Scheduler):
        """
        Initialize the scheduler. This must be implemented by child tasks.

        Returns (Scheduler): The scheduler to use for the task.
        """
        schedulers = {}
        for scheduler_key, scheduler_params in self.config.schedulers.items()
            scheduler = get_from_register(Scheduler, scheduler_params.name)(scheduler_params.params)
            schedulers[scheduler_key] = scheduler

        return schedulers

    def _get_meter(self) -> Meter:
        """
        Initialize the metrics. This must be implemented by child tasks.

        Returns (Meter): The metrics to use for the task.
        """
        return get_from_register(Meter, self.config.meter.name)

    def _get_pipeline(self) -> list(Pipeline):
        """
        Initialize the augment for this task. This must be implemented by child
        tasks.

        Returns (Pipeline): The augment to use for this task.
        """
        pipelines = []
        print("printing registry")
        print_registry()
        for pipeline_idx, pipeline_params in self.config.pipeline.branches.items():
            pipeline = get_from_register(Pipeline, pipeline_params.name)(**pipeline_params.params)
            pipelines.append(pipeline)

        return pipelines

    def _get_dataloader(self) -> Dataloader:
        """
        Initialize the dataloader. This must be implemented by child tasks.

        Returns (Dataloader): The dataloader to use for the task.
        """
        return get_from_register(self.config.dataloader)
