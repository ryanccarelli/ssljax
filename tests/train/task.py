import parameterized
import pytest
from ssljax.losses import l2_normalize
from ssljax.train.metrics.meter import SSLMeter
from ssljax.train.scheduler.scheduler import cosine_decay_schedule
from ssljax.train.ssltrainer import SSLTrainer
from ssljax.train.task import Task


def test_init(cputestconfig):
    task = Task(cputestconfig)
    assert isinstance(task.trainer, SSLTrainer)
    assert isinstance(task.model, SSLModel)
    assert isinstance(task.loss, l2_normalize)
    assert isinstance(task.optimizer, [sgd])
    assert isinstance(task.scheduler, [cosine_decay_schedule])
    assert isinstance(task.meter, SSLMeter)
