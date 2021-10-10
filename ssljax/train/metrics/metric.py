import logging

logger = logging.getLogger(__name__)

logger.error(f"{__name__}: THESE ARE PLACEHOLDERS!!")


class Metric:
    def __call__(self, preds, targets, params) -> float:
        raise NotImplementedError()

    def get_epoch_value(self, batches_seen: int) -> float:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
