from typing import Dict

from ssljax.core import register
from ssljax.train.metrics import Meter, Metric


@register(Meter, "SSLMeter")
class SSLMeter(Meter):
    """
    The base class for tracking metrics during a task.
    Args:
        metrics (Dict[str, Metric]): A dict where each key maps to
         a metric. In the returned dict by `Metric.__call__`, the names of
         each metric (the keys in this dict) will be used to index the metric
         values calculated (by the Metric values in this Dict).
    """

    def __init__(self, metrics: Dict[str, Metric]):
        super().__init__(self, metrics)
