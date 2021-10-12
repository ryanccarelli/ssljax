# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging
import random
from collections import OrderedDict
from typing import Callable, Dict, List

import jax
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.core.config import Config
from ssljax.core.utils.register import get_from_register, print_registry
from ssljax.data import DataLoader
from ssljax.losses.loss import Loss
from ssljax.models.model import Model
from ssljax.optimizers import Optimizer
from ssljax.train import SSLTrainer, Trainer
from ssljax.train.metrics import Meter
from ssljax.train.postprocess import PostProcess
from ssljax.train.scheduler import Scheduler

logger = logging.getLogger(__name__)


class Task:
    """
    Abstract class for a task.

    A task is specified by the config file and constructs and holds the:
        - trainer
        - model
        - loss
        - optimizer
        - scheduler
        - meter
        - pipeline
        - dataloader
        - post-processing

    Args:
        config (ssljax.config): a hydra configuration file
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.rng = self.prepare_environment(self.config)
        self.trainer = self._get_trainer()
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizers = self._get_optimizers()
        self.schedulers = self._get_schedulers()
        self.meter = self._get_meter()
        self.pipelines = self._get_pipelines()
        self.dataloader = self._get_dataloader()
        self.post_process_funcs = self._get_post_process_list()

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
        return get_from_register(Loss, self.config.loss)

    def _get_optimizers(self) -> List[Optimizer]:
        """
        Initialize optimizer.

        Returns (Optimizer): The optimizers to use for the task.
        """

        optimizers = OrderedDict()
        for optimizer_key, optimizer_params in self.config.optimizers.branches.items():
            optimizer = get_from_register(Optimizer, optimizer_params.name)(
                **optimizer_params.params
            )
            optimizers[optimizer_key] = optimizer

        return optimizers

    def _get_schedulers(self) -> Dict[str, Scheduler]:
        """
        Initialize the scheduler. This must be implemented by child tasks.

        Returns (Scheduler): The scheduler to use for the task.
        """
        schedulers = {}
        for scheduler_key, scheduler_params in self.config.schedulers.branches.items():
            scheduler = get_from_register(Scheduler, scheduler_params.name)(
                **scheduler_params.params
            )
            schedulers[scheduler_key] = scheduler

        return schedulers

    def _get_meter(self) -> Meter:
        """
        Initialize the metrics. This must be implemented by child tasks.

        Returns (Meter): The metrics to use for the task.
        """
        return get_from_register(Meter, self.config.meter.name)

    def _get_pipelines(self) -> List[Pipeline]:
        """
        Initialize the augment for this task. This must be implemented by child
        tasks.

        Returns (Pipeline): The augment to use for this task.
        """
        pipelines = []
        for pipeline_idx, pipeline_params in self.config.pipelines.branches.items():
            pipeline = get_from_register(Pipeline, pipeline_params.name)(
                **pipeline_params.params
            )
            pipelines.append(pipeline)

        return pipelines

    def _get_dataloader(self) -> DataLoader:
        """
        Initialize the dataloader. This must be implemented by child tasks.

        Returns (Dataloader): The dataloader to use for the task.
        """
        return get_from_register(DataLoader, self.config.dataloader.name)(
            **self.config.dataloader.params
        )

    def _get_post_process_list(self) -> List[Callable]:
        print_registry()
        post_process_list = []

        for (
            post_process_idx,
            post_process_params,
        ) in self.config.post_process.funcs.items():
            post_process = get_from_register(PostProcess, post_process_params.name)(
                **post_process_params.params
            )
            post_process_list.append(post_process)

        return post_process_list

    def prepare_environment(config) -> jax.numpy.DeviceArray:
        """
        Set the random seeds.

        Args:
            config: Hydra config.

        Returns (jax.numpy.DeviceArray): A Jax PRNG object.
        """

        # Get the seed values from the config.
        seed = config.env.seed if ("env" in config and config.env.seed) else 0
        numpy_seed = (
            config.env.numpy_seed if ("env" in config and config.env.numpy_seed) else 0
        )
        jax_seed = (
            config.env.jax_seed if ("env" in config and config.env.jax_seed) else 0
        )

        if seed is not None:
            random.seed(seed)
        if numpy_seed is not None:
            numpy.random.seed(numpy_seed)

        # Set the jax seed and return it. If the jax seed is None, default to 0.
        return jax_random.PRNGKey(jax_seed if jax_seed is not None else 0)
