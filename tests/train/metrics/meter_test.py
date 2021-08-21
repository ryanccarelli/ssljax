import pytest
from overrides import overrides

from ssljax.train.metrics import Meter, Metric


class SimpleMetric(Metric):
    def __init__(self):
        self.results = []

    def __call__(self, logits, targets, weights) -> float:
        self.results.append(logits == targets)
        return 1.0 if logits == targets else 0.0

    def get_epoch_value(self, batches_seen: int) -> float:
        return sum(self.results) / batches_seen

    def reset(self):
        self.results = []


class TestMeterBase:
    @pytest.fixture()
    def logits(self):
        return [1, 2, 3, 4, 5]

    @pytest.fixture()
    def targets(self):
        return [1, 2, 4, 3, 5]

    @pytest.fixture()
    def weights(self):
        return [1, 1, 1, 1, 1]

    def setup_method(self):
        self.meter = Meter(
            {"A": SimpleMetric(), "B": SimpleMetric()}
        )

    def test_call(self, logits, targets, weights):
        results = self.meter(logits[0], targets[0], weights[0])
        assert self.meter._batches_seen == 1
        assert self.meter._metrics["A"].results == [1]
        assert self.meter._metrics["B"].results == [1]
        assert results == {
            "A": 1,
            "B": 1
        }

    def test_get_epoch_metrics(self, logits, targets, weights):
        for i in range(len(logits)):
            self.meter(logits[i], targets[i], weights[i])
            assert self.meter._batches_seen == i+1

        results = self.meter.get_epoch_metrics()
        assert self.meter._batches_seen == 0
        assert self.meter._metrics["A"].results == []
        assert self.meter._metrics["B"].results == []
        assert results == {
            "A": 0.6,
            "B": 0.6
        }
