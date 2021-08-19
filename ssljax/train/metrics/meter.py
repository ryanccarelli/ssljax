from collections import Counter
import logging
from typing import Callable, Dict

from ssljax.train.metrics.metric import MetricBase

logger = logging.getLogger(__name__)


# TODO(gabeorlanski): This will need to be modified to handle mean metrics or
#  list metrics
class MeterBase:
    """
    The base class for tracking metrics during a task.
    Args:
        metrics (Dict[str, MetricBase]): A dict where each key maps to
         a metric. In the returned dict by `MetricBase.__call__`, the names of
         each metric (the keys in this dict) will be used to index the metric
         values calculated (by the Metric values in this Dict).
    """

    def __init__(self, metrics: Dict[str, MetricBase]):
        self._metrics = metrics
        self._batches_seen = 0

    def __call__(self, logits, targets, weights) -> Dict[str, float]:
        """
        Calculate the metrics for
        Args:
            logits:
            targets:
            weights:

        Returns:

        """
        batch_metrics = {}

        # Go through the dictionary of metrics and calculate each one. Each
        # metric is responsible for tracking the overall epoch metrics
        for metric_name, metric in self._metrics.items():
            # Save it to the batch metrics dict so that we can return it.
            batch_metrics[metric_name] = metric(logits, targets, weights)

        self._batches_seen += 1

        return batch_metrics

    def get_epoch_metrics(self) -> Dict[str, float]:
        """
        Get the metrics for the END of the epoch.

        Returns (Dict[str, float]): The average metrics for the epoch.
        """

        epoch_metrics = {}

        # Go through each metric
        for metric_name, metric in self._metrics.items():
            epoch_metrics[metric_name] = metric.get_epoch_value(self._batches_seen)

            # Because this is the end of the epoch, we reset the metrics for the
            # next epoch.
            metric.reset()

        # Reset the number of batches seen because it is the end of the epoch.
        self._batches_seen = 0

        return epoch_metrics
