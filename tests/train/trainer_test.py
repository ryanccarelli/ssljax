import os

import pytest
from ssljax.train.task import Task


class TestTrainer:
    def test_train_cpu(self, cputestconfig):
        task = Task(cputestconfig)
        trainer = task.trainer
        trainer.train()

    def test_train_byol(self, byoltestconfig):
        task = Task(byoltestconfig)
        trainer = task.trainer
        trainer.train()
