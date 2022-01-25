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
    assert isinstance(task.schedulers, dict)
    assert isinstance(task.trainer, SSLTrainer)
    assert isinstance(task.model, SSLModel)
    assert isinstance(task.optimizers, dict)
    assert isinstance(task.post_process_funcs, dict)
    assert isinstance(task.pipelines, dict)
    assert isinstance(task.data, dict)
