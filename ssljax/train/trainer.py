import logging
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
from ssljax.core.utils import prepare_environment

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

    def train(self):
        # recover training state from checkpoint
        # TODO: parse config
        _ = self._load_training_state(self.config.pop(""))
        model = self.model(self.config.pop("model"))
        tx = Optimizer.from_params(self.config.pop("optimizer"))
        platform = jax.local_devices()[0].platform
        dynamic_scale = None
        if config.half_precision and platform == "gpu":
            dynamic_scale = optim.DynamicScale()
        rng, key = random.split(rng)
        state = TrainState.create(
            apply_fn=model().apply,
            params=model().init(key, init_data, rng)["params"],
            tx=tx,
        )
        # get dataloaders
        # iterate dataloader
        # put data on device
        # self.epoch(batch)
        # metrics
        raise NotImplementedError

    def epoch(self, train_data, batch_size, rng):
        # get trainingset size
        train_data_size = len(train_data)
        steps = train_data_size // batch_size
        perms = jax.random.permutation(rng, train_data_size)
        # apply augmentations here?
        loss, logits, grads = self.step(batch)

    @jax.jit
    def step(self, batch):
        """
        Compute gradients, loss, accuracy per batch
        """
        # get loss function
        # forward pass
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(self.state.params)
        return loss, logits, grads

    def model(config):
        # return subclass of BaseSSL from config
        raise NotImplementedError

    def optimizer(config):
        # return optimizer from config
        raise NotImplementedError

    def _load_training_state(self, config):
        raise NotImplementedError
