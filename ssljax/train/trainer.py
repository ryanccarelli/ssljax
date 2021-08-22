import logging

import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax import random
from ssljax.tasks import SSLTask

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

    def evalstep(self):
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

    def train(self):
        # setup devices
        platform = jax.local_devices()[0].platform
        dynamic_scale = None
        if self.config.pop("half_precision") and platform == "gpu":
            dynamic_scale = optim.DynamicScale()
        # instantiate training state
        if self.config.pop("load_training_state"):
            state = self._load_training_state(self.config.pop(""))
        else:
            # TODO: should be shape of input data
            init_data = jnp.ones((task.batch_size, task.input_size), jnp.float32)
            state = TrainState.create(
                apply_fn=self.task.model.apply,
                params=self.task.model.init(key, init_data, self.rng)["params"],
                tx=opt,
            )
        # get parameters
        lr = task.scheduler.lr(state.step)
        # TODO: If loss returns auxiliary data, pass has_aux=True
        grad_fn = jax.value_and_grad(task.loss)
        for data, targets in iter(task.dataset):
            train_data = jax.device_put(data)
            targets = jax.device_put(targets)
            task.augment(train_data)
            state = self.epoch(train_data, targets, grad_fn, state)

    def epoch(self, train_data, targets, grad_fn, state, batch_size, rng):
        # get trainingset size
        train_data_size = len(train_data)
        steps_per_epoch = train_data_size // batch_size
        perms = jax.random.permutation(rng, train_data_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        # apply augmentations here?
        batch_metrics = []
        # TODO: sync cross-device w treemap?
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_data.items()}
            state, metrics = self.step(batch, targets, grad_fn, state)
            batch_metrics.append(metrics)
        # compute mean of metrics across each batch in epoch.
        # log metrics
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }
        # TODO: log
        logger.log(
            f"train epoch: {epoch}, loss: {epoch_metrics_np['loss']}, accuracy: {epoch_metrics_np['accuracy']}"
        )
        return state

    @jax.jit
    def step(self, batch, targets, state):
        """
        Compute gradients, loss, accuracy per batch
        """
        (loss, logits), grads = grad_fn(self.state.params)
        grads = jax.lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        metrics = self.task.meter(logits, targets, weights)
        # summary = jax.tre_map(lambda x: x.mean(), train_metrics)
        return state, metrics

    def eval(self):
        raise NotImplementedError

    def evalstep(self):
        raise NotImplementedError


if __name__ == "__main__":
    test = SSLTrainer(None, None)
    print(test)
