import logging
from collections import Counter
from typing import Callable, Dict

from ssljax.core.utils import register

logger = logging.getLogger(__name__)


class Meter:
    def __init__(self, metrics):
        self._metrics = metrics
        self._batches_seen = 0

    def __call__(self, preds, targets, params) -> Dict[str, float]:
        """
        Calculate metrics
        Args:
            preds: model predictions
            targets: model targets
            params: model parameters (state.params)

        """
        batch_metrics = {}

        # Go through the dictionary of metrics and calculate each one. Each
        # metric is responsible for tracking the overall epoch metrics
        for metric_name, metric in self._metrics.items():
            # Save it to the batch metrics dict so that we can return it.
            batch_metrics[metric_name] = metric(preds, targets, params)

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
