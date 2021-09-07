import parameterized
import pytest
from ssljax.train.task import Task
from ssljax.train.ssltrainer import SSLTrainer
from ssljax.losses import l2_normalize
from ssljax.train.scheduler.scheduler import cosine_decay_schedule
from ssljax.train.metrics.meter import SSLMeter


def test_init(cputestconfig):
    task = Task(cputestconfig)
    assert task.trainer == SSLTrainer
    assert task.model == SSLModel
    assert task.loss == l2_normalize
    assert task.optimizer == [sgd]
    assert task.scheduler == [cosine_decay_schedule]
    assert task.meter == SSLMeter
