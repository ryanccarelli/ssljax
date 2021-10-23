import flax
from typing import Any
from flax.training.train_state import TrainState

class TrainState(TrainState):
  batch_stats: Any
