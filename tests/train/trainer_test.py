import os

import pytest
from ssljax.train.task import Task


class TestTrainer:
    def test_train(self, cputestconfig):
        task = Task(cputestconfig)
        trainer = task.trainer
        trainer.train()
