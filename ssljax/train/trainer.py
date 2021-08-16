import logging

import flax.optim as optim
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
from ssljax.core.utils import prepare_environment
from ssljax.train.task import SSLTask

logger = logging.getLogger(__name__)


class SSLTrainer:
    """
    Class to manage model training and feature extraction.

    Prepares experiment from config file:
        - optimizer
        - loss
        - model

    Args:
        rng (jnp.DeviceArray):
        config (): configuration file
    """
    def __init__(self, rng, config):
        self.config = config
        self.rng = prepare_environment(self.config)
        self.task = self.build_task(config)
        self.model = task.model

    def train(self):
        # recover training state from checkpoint
        # TODO: parse config
        platform = jax.local_devices()[0].platform
        dynamic_scale = None
        if self.config.pop("half_precision") and platform == "gpu":
            dynamic_scale = optim.DynamicScale()
        _ = self._load_training_state(self.config.pop(""))
        # TODO: should be shape of input data
        init_data = jnp.ones((task.batch_size, TODO), jnp.float32)
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(key, init_data, self.rng)["params"],
            tx=opt,
        )
        # TODO: make this work
        for data in iter(task.dataset)
            batch = jax.device_put(data)
            task.augment(batch)
            self.epoch(batch, )

    def epoch(self, train_data, batch_size, rng):
        # get trainingset size
        train_data_size = len(train_data)
        steps_per_epoch = train_data_size // batch_size
        perms = jax.random.permutation(rng, train_data_size)
        perms = perms[:steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        # apply augmentations here?
        batch_metrics = []
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_data.items()}
            state, metrics = self.step(batch)
            batch_metrics.append(metrics)
        # compute mean of metrics across each batch in epoch.
        # log metrics
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]}
        print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
            epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
        return state

    @jax.jit
    def step(self, batch, state):
        """
        Compute gradients, loss, accuracy per batch
        """
        # get loss function
        # forward pass
        grad_fn = jax.value_and_grad(task.loss, has_aux=True)
        (loss, logits), grads = grad_fn(self.state.params)
        state = state.apply_gradients(grads=grads)
        # TODO: dynamic scale?
        metrics = # compute metrics here
        return state, metrics

    @property
    def model():
        # return subclass of BaseSSL from config
        return self._model

    @setter
    def model(config):
        # init model from config
        raise NotImplementedError

    def optimizer(config):
        # return optimizer from config
        raise NotImplementedError

    def _load_training_state(self, config):
        raise NotImplementedError

    def build_task(self, config):
        task = SSLTask().from_params(config)
