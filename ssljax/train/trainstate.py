import flax
from typing import Any
from flax.training.train_state import TrainState

class TrainState(TrainState):
  mutable_states: Any
