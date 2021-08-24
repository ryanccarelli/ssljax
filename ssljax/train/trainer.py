import logging

import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax import random

logger = logging.getLogger(__name__)


class Trainer:
    """
    Class to manage model training and feature extraction.
    """

    def train(self):
        raise NotImplementedError()

    def epoch(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


class SSLTrainer(Trainer):
    """
    Class to manage SSL training and feature extraction.

    Args:
        rng (jnp.DeviceArray):
        config (json): configuration file
    """

    def __init__(self, rng, task):
        self.task = task
        self.rng = rng

    def train(self):
        # setup devices
        platform = jax.local_devices()[0].platform
        # instantiate training state
        if self.config.pop("load_training_state"):
            state = self._load_training_state(self.config.pop("load_training_state"))
        else:
            init_data = jnp.ones(
                (self.task.batch_size, self.task.input_shape), jnp.float32
            )
            key, self.rng = random.split(self.rng)
            opt = self.task.optimizer
            # TODO: we need to split the train state between online and target
            state = TrainState.create(
                apply_fn=self.task.model.apply,
                params=self.task.model.init(key, init_data, self.rng)["params"],
                tx=opt,
            )
        # get parameters
        for data, _ in iter(task.dataset):
            train_data = jax.device_put(data)
            state = self.epoch(train_data, targets, grad_fn, state)

    def epoch(self, train_data, grad_fn, state):
        # TODO: should we use dataloaders
        train_data_size = len(train_data)
        steps_per_epoch = train_data_size // self.task.batch_size
        rng, self.rng = random.split(self.rng)
        perms = jax.random.permutation(rng, train_data_size)
        perms = perms[: steps_per_epoch * self.task.batch_size]
        perms = perms.reshape((steps_per_epoch, self.task.batch_size))
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_data.items()}
            batch = self.task.augment(batch)
            distributed_step = jax.pmap(self.step, axis_name="batch")
            state = distributed_step(batch, grad_fn, state)
        # TODO: call meter
        self.task.meter.get_epoch_metrics()
        return state

    @jax.jit
    def step(self, batch, state):
        """
        Compute gradients, loss, accuracy per batch
        """
        lr = self.task.scheduler.lr(state.step)
        grad_fn = jax.value_and_grad(self.task.loss)
        (aux), grad = grad_fn(state.params, batch)
        # TODO: works for flax optimizer
        # grad = jax.lax.pmean(grad, axis_name="batch")
        grad = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="batch"), grads)
        # TODO: sometimes we want to apply only to the online network and update
        # the other network by ema
        state = state.apply_gradients(grad)
        # target_params = jax.tree_multimap(
        #    lambda x, y: x + (1 - tau) * (y - x), target_params, online_params
        # )
        self.task.meter(aux)
        return state

    def eval(self):
        raise NotImplementedError

    def evalstep(self):
        raise NotImplementedError


if __name__ == "__main__":
    test = SSLTrainer(None, None)
    print(test)
