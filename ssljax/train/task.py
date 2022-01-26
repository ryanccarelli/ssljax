# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import jax.random
from omegaconf import DictConfig
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.core import get_from_register, prepare_environment, print_registry
from ssljax.data import ScenicData
from ssljax.losses.loss import Loss
from ssljax.models.model import Model
from ssljax.optimizers import Optimizer
from ssljax.train import Meter, Scheduler, SSLTrainer, Trainer
from ssljax.train.postprocess import PostProcess

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
        config (omegaconf.DictConfig): a hydra configuration file
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        # for internal use
        self.config = config
        self.rng = prepare_environment(self.config)
        self.schedulers = self._get_schedulers()
        # for external use
        # NOTE: schedulers must be declared before the components they schedule
        # for now before optimizers and post_process_funcs
        self.trainer = self._get_trainer()
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizers = self._get_optimizers()
        self.meter = self._get_meter()
        self.pre_pipelines = self._get_pre_pipelines()
        self.post_pipelines = self._get_post_pipelines()
        self.data = self._get_data()
        self.post_process_funcs = self._get_post_process_list()

    def _get_trainer(self) -> Trainer:
        """
        Initialize the trainer.
        """
        return get_from_register(Trainer, self.config.trainer.name)(rng=self.rng, task=self)

    def _get_model(self) -> Model:
        """
        Initialize the model.

        Returns (Model): The model to use for this task.
        """
        return get_from_register(Model, self.config.model.name)(self.config)

    def _get_loss(self) -> Loss:
        """
        Initialize the loss function.

        Returns (Loss): The loss to use for the task.
        """
        return partial(
            get_from_register(Loss, self.config.loss.name),
            **self.config.loss.params,
        )

    def _get_optimizers(self) -> List[Optimizer]:
        """
        Initialize the optimizer.

        Returns (Optimizer): The optimizers to use for the task.
        """

        optimizers = OrderedDict()
        for optimizer_key, optimizer_params in self.config.optimizers.branches.items():
            schedulers = {}
            for key, val in self.schedulers["branches"][optimizer_key].items():
                schedulers[key] = get_from_register(Scheduler, val.name)(**val.params)
            optimizer = get_from_register(Optimizer, optimizer_params.name)(
                **schedulers,
                **optimizer_params.params,
            )
            optimizers[optimizer_key] = optimizer

        return optimizers

    def _get_schedulers(self) -> Dict[str, Scheduler]:
        """
        Initialize the scheduler.

        Returns (Scheduler): The scheduler to use for the task.
        """
        schedulers = {"branches": {}, "post_process": {}}
        for scheduler_key, scheduler_params in self.config.schedulers.branches.items():
            schedulers["branches"][scheduler_key] = scheduler_params

        if "post_process" in self.config.schedulers:
            for (
                scheduler_key,
                scheduler_params,
            ) in self.config.schedulers.post_process.items():
                schedulers["post_process"][scheduler_key] = scheduler_params

        return schedulers

    def _get_meter(self) -> Meter:
        """
        Initialize the metrics.

        Returns (Meter): The metrics to use for the task.
        """
        return get_from_register(Meter, self.config.meter.name)

    def _get_pre_pipelines(self) -> Pipeline or None:
        """
        Initialize the pre-augmentations.

        Returns (Pipeline): The pre-augmentation pipeline to use for this task.
        """
        if "pre" in self.config.pipelines:
            return Pipeline(self.config.pipelines.pre.augmentations)
        else:
            return None

    def _get_post_pipelines(self) -> List[Pipeline]:
        """
        Initialize the post-augmentations.

        Returns (Pipeline): The post-augmentation pipeline to use for this task.
        """
        pipelines = []
        for pipeline_idx, pipeline_params in self.config.pipelines.branches.items():
            pipelines.append(Pipeline(pipeline_params.augmentations))

        return pipelines

    def _get_data(self):
        """
        Initialize the dataloader. This must be implemented by child tasks.

        Returns (Dataloader): The dataloader to use for the task.
        """
        _, data_rng = jax.random.split(self.rng)
        return get_from_register(ScenicData, self.config.data.pretraining.name)(
            config=self.config.data.pretraining.params,
            data_rng=data_rng,
        )

    def _get_post_process_list(self) -> List[Callable]:
        post_process_list = []
        if "post_process" in self.config:
            for (
                post_process_idx,
                post_process_params,
            ) in self.config.post_process.funcs.items():
                schedulers = {}
                for key, val in self.schedulers["post_process"][post_process_idx].items():
                    schedulers[key] = get_from_register(Scheduler, val.name)(**val.params)
                post_process = get_from_register(PostProcess, post_process_params.name)(
                    **schedulers,
                    **post_process_params.params,
                )
                post_process_list.append(post_process)
        return post_process_list
