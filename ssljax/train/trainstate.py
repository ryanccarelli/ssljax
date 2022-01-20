from typing import Any

import flax
from flax.training.train_state import TrainState


class TrainState(TrainState):
    """
    flax.training.train_state with optional mutable parameters.
    """

    mutable_states: Any
    global_step: Optional[int] = 0
    accum_train_time: Optional[int] = 0
