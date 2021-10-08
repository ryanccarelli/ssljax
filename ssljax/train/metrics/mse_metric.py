import jax.numpy as jnp
from ssljax.core.utils import register
from ssljax.train.metrics.metric import Metric


@register(Metric, "MSE")
class MSEMetric(Metric):
    def __init__(self):
        self.running_mse = 0.0

    def __call__(self, preds, targets, *args) -> float:
        mse = jnp.mean(jnp.square(preds - targets))
        self.running_mse = self.running_mse + mse
        return float(mse.item())

    def get_epoch_value(self, batches_seen: int) -> float:
        return self.running_mse / batches_seen

    def reset(self):
        self.running_mse = 0.0
