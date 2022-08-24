import functools

from clu.metrics import (Accuracy, Average, CollectingMetric, Collection,
                         LastValue, Std)
from immutabledict import immutabledict
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.classification_model import (
    _CLASSIFICATION_METRICS, classification_metrics_function)
from ssljax.core import register


class Metric:
    """
    Abstract class representing metric functions.
    Metrics are used to track model performance.
    Metrics returns a metric function with the API:
    ```metrics_fn(outs, batch)```
    """

    pass


_CLASSIFICATION_METRICS = immutabledict(
    {
        "accuracy": (
            model_utils.weighted_correctly_classified,
            model_utils.num_examples,
        ),
        "loss": (
            model_utils.weighted_unnormalized_softmax_cross_entropy,
            model_utils.num_examples,
        ),
    }
)


def classification_metrics(target_is_onehot):
    return functools.partial(
        classification_metrics_function,
        target_is_onehot=target_is_onehot,
        metrics=_CLASSIFICATION_METRICS,
    )


metrics = {"classification": classification_metrics}

for name, func in metrics.items():
    register(Metric, name)(func)
