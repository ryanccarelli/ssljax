import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random


class SSLTrainer:
    def __init__(self, config):
        self.config = config
        # parse config
        rng = random.PRNGKey(0)
        rng, key = random.split(rng)
        state = TrainState.create(
            apply_fn=TODO.model().apply,
            params=TODO.model().init(key, init_data, rng)["params"],
            tx=TODO,
        )

    def model():
        # return subclass of BaseSSL from config
        raise NotImplementedError

    def train(self):
        # get dataloaders
        # apply augmentations
        # put data on device
        # train step
        # append metric
        raise NotImplementedError

    @jax.jit
    def step(self):
        # get loss function
        # forward pass
        raise NotImplementedError
