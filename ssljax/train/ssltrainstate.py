from abc import ABC
from typing import Any, Optional

import flax
import jax.numpy as jnp
import optax
from flax import optim

from ssljax.train import Task

@flax.struct.dataclass
class SSLTrainState:
    params: Optional[Any] = None
    mutable_states: Optional[Any] = None
    opt_state: Optional[optax.OptState] = None
    global_step: Optional[int] = 0
    optimizer: Optional[optax.GradientTransformation] = None
    task: Task
    rng: Optional[jnp.ndarray] = None

    def __getitem__(self, item):
        """Make TrainState a subscriptable object."""
        return getattr(self, item)

    def get(self, keyname: str, default: Optional[Any] = None) -> Any:
        """Return the value for key if it exists otherwise the default."""
        try:
            return self[keyname]
        except KeyError:
            return default

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.optimizer.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.global_step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )