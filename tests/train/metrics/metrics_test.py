import jax.numpy as jnp
import pytest
from ssljax.train.metrics.mse_metric import MSEMetric


def test_mse_metric():
    targets = jnp.array([1, 2, 3, 4])
    preds = jnp.array([4, 3, 2, 1])
    metric = MSEMetric()
    assert metric(preds, targets) == 5.0
    metric(preds, targets)
    assert metric.get_epoch_value(batches_seen=2) == 5.0
    metric.reset()
    assert metric.running_mse == 0.0
