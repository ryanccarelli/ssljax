# similar to https://github.com/facebookresearch/vissl/blob/master/vissl/trainer/train_task.py
import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Mapping

import jax
import jax.numpy as jnp
import jax.random
from clu import metric_writers
from omegaconf import DictConfig
from scenic.dataset_lib.dataset_utils import Dataset
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.core import get_from_register, prepare_environment
from ssljax.data import ScenicData
from ssljax.losses.loss import Loss
from ssljax.models.model import Model
from ssljax.optimizers import Optimizer
from ssljax.train import Scheduler, Trainer
from ssljax.train.metrics import Metric
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
        - writer
        - pipeline
        - dataloader
        - post-processing
        - metrics

    Args:
        config (omegaconf.DictConfig): a hydra configuration file
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.rng = prepare_environment(self.config)
        # get_schedulers first
        self.scheduler = self._get_scheduler()
        self.trainer = self._get_trainer()
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.writer = self._get_writer()
        self.pipeline = self._get_pipeline()
        self.data = self._get_data()
        self.post_process = self._get_post_process()

    def _get_scheduler(self) -> Mapping[str, Callable]:
        """
        Initialize the scheduler.

        Returns (Scheduler): The scheduler to use for the task.
        """
        schedulers = {}
        for component in self.config.scheduler:
            component_schedulers = {}
            for scheduler_key, scheduler_params in self.config.scheduler[
                component
            ].items():
                component_schedulers[scheduler_key] = scheduler_params
            schedulers[component] = component_schedulers

        return schedulers

    def _get_trainer(self) -> Trainer:
        """
        Initialize the trainer.
        """
        schedule = self.scheduler["trainer"] if "trainer" in self.scheduler else {}
        return get_from_register(Trainer, self.config.trainer.name)(
            rng=self.rng,
            task=self,
            **schedule,
        )

    def _get_model(self) -> Model:
        """
        Initialize the model.

        Returns (Model): The model to use for this task.
        """
        schedule = self.scheduler["model"] if "model" in self.scheduler else {}
        return get_from_register(Model, self.config.model.name)(self.config, **schedule)

    def _get_loss(self) -> Callable:
        """
        Initialize the loss function.

        Returns (Loss): The loss to use for the task.
        """
        schedule = self.scheduler["loss"] if "loss" in self.scheduler else {}
        return partial(
            get_from_register(Loss, self.config.loss.name),
            **self.config.loss.params,
            **schedule,
        )

    def _get_optimizer(self) -> List[Callable]:
        """
        Initialize the optimizer.

        Returns (Optimizer): The optimizers to use for the task.
        """

        schedule = self.scheduler["branch"] if "branch" in self.scheduler else {}
        optimizers = OrderedDict()
        for optimizer_key, optimizer_params in self.config.optimizer.branch.items():
            schedulers = {}
            for key, val in schedule[optimizer_key].items():
                schedulers[key] = get_from_register(Scheduler, val.name)(**val.params)
            optimizer = get_from_register(Optimizer, optimizer_params.name)(
                **schedulers,
                **optimizer_params.params,
            )
            optimizers[optimizer_key] = optimizer

        return optimizers

    def _get_writer(self) -> Callable:
        """
        Initialize the metrics.

        Returns (metric_writers.Writer): a writer
        """
        writer = metric_writers.create_default_writer(
            self.config.workdir, just_logging=jax.process_index() > 0, asynchronous=True
        )
        return writer

    def _get_post_process(self) -> Mapping[str, Callable]:
        schedule = (
            self.scheduler["post_process"] if "post_process" in self.scheduler else {}
        )
        post_process_dict = OrderedDict()
        for (
            post_process_idx,
            post_process_params,
        ) in self.config.post_process.funcs.items():
            schedulers = {}
            if post_process_idx in schedule:
                for key, val in schedule[post_process_idx].items():
                    schedulers[key] = get_from_register(Scheduler, val.name)(
                        **val.params
                    )
            post_process = get_from_register(PostProcess, post_process_params.name)(
                **schedulers,
                **post_process_params.params,
            )
            post_process_dict[post_process_idx] = post_process
        return post_process_dict

    def _get_pipeline(self) -> Mapping[str, Callable]:
        """
        Initialize the post-augmentations.

        Returns (Pipeline): The post-augmentation pipeline to use for this task.
        """
        pipelines = OrderedDict()
        for pipeline_idx, pipeline_params in self.config.pipeline.branch.items():
            pipelines[pipeline_idx] = Pipeline(pipeline_params.augmentations)

        return pipelines

    def _get_data(self) -> Dataset:
        """
        Initialize the dataloader. This must be implemented by child tasks.

        Returns (Dataloader): The dataloader to use for the task.
        """
        data = {}
        for data_idx, data_params in self.config.data.items():
            _, data_rng = jax.random.split(self.rng)
            data[data_idx] = get_from_register(ScenicData, data_params.name)(
                config=data_params.params,
                data_rng=data_rng,
            )
        return data

    def _get_metric(self):
        """
        Initialize metric.
        """
        metric = get_from_register(Metric, self.config.metric.name)(
            **self.config.metric.params
        )
        return metric
