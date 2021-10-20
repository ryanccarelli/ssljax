import os

import pytest
from hydra import compose, initialize
from ssljax.train.task import Task


class TestTrainer:
    def test_train_cpu(self, cputestconfig):
        task = Task(cputestconfig)
        trainer = task.trainer
        trainer.train()

    """
    def test_train_byol(self, byoltestconfig):
        task = Task(byoltestconfig)
        trainer = task.trainer
        trainer.train()
    """


def byoltestconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="byol_conf.yaml")
    return cfg
