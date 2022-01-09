import parameterized
import pytest
from ssljax.losses import l2_normalize
from ssljax.models.sslmodel import SSLModel
from ssljax.train.metrics.sslmeter import SSLMeter
from ssljax.train.scheduler.scheduler import cosine_decay_schedule
from ssljax.train.ssltrainer import SSLTrainer
from ssljax.train.task import Task


def test_init(basecpuconfig):
    task = Task(basecpuconfig)
    assert isinstance(task.trainer, SSLTrainer)
    assert issubclass(task.model, SSLModel)
    assert issubclass(task.meter, SSLMeter)
