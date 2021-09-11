import logging

logger = logging.getLogger(__name__)

import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training.train_state import TrainState
from jax import random
from optax._src.base import GradientTransformation
from ssljax.core.utils import register
from ssljax.optimizers.base import ParameterTransformation
from ssljax.train import Trainer


@register(Trainer, "SSLTrainer")
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
        key, self.rng = random.split(self.rng)
        params, states = self.initialize(key)
        for data, _ in iter(self.task.dataset):
            train_data = jax.device_put(data)
            params, states = self.epoch(train_data, states)

    def epoch(self, train_data, params, states):
        # TODO: should we use dataloaders
        train_data_size = len(train_data)
        steps_per_epoch = train_data_size // self.task.batch_size
        rng, self.rng = random.split(self.rng)
        perms = jax.random.permutation(rng, train_data_size)
        perms = perms[: steps_per_epoch * self.task.batch_size]
        perms = perms.reshape((steps_per_epoch, self.task.batch_size))
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_data.items()}
            batch = jax_utils.replicate(batch)
            batch = self.task.augment(batch)
            params, states = self.step(batch, params, states)
        # TODO: meter must implement distributed version
        # batch_metrics = jax.tree_multimap(lambda *xs: np.array(xs), *batch_metrics)
        # epoch_metrics ={k:np.mean(batch_metrics[k], axis=0) for k in batch_metrics}
        # self.task.meter.get_epoch_metrics()
        return params, states

    @jax.pmap
    def step(self, batch, params, states):
        """
        Compute gradients, loss, accuracy per batch
        """
        lr = None
        decay = None
        step = states[0].count
        # TODO: correctly get step from the state
        if self.task.scheduler.lr:
            lr = self.task.scheduler.lr(step)
        if self.task.scheduler.decay:
            decay = self.task.scheduler.decay(step)
        grad_fn = jax.value_and_grad(self.task.loss)
        (aux), grad = grad_fn(params, batch)
        grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="batch"), grads)
        opt = self.task.optimizers(lr, decay)

        # TODO: shadow
        def update_fn(_opt, _grads, _state, _params):
            _update, _state = _opt.update(_grads, _state, _params)
            if isinstance(_opt, GradientTransformation):
                _params = optax.apply_updates(_params, _update)
            elif isinstance(_opt, ParameterTransformation):
                _params = _update
            return _params, _state

        params, states = zip([update_fn(op, grads, states, params) for op in opt])

        # TODO: call meter (aux)
        return params, states

    def eval(self):
        raise NotImplementedError

    @jax.pmap
    def evalstep(self):
        raise NotImplementedError

    # TODO: working pmap
    def initialize(self, rng):
        # setup devices
        platform = jax.local_devices()[0].platform
        # set model_dtype
        model_dtype = jnp.float32
        if self.task.config.env.half_precision:
            if platform == "tpu":
                model_dtype = jnp.bfloat16
            else:
                model_dtype = jnp.float16
        else:
            model_dtype = jnp.float32
        # init training state
        if self.task.config.env.checkpoint:
            params, state = self._load_from_checkpoint(
                self.task.config.load_from_checkpoint
            )
        else:
            # init model
            init_shape = [self.task.config.dataloader.params.batch_size] + list(
                eval(self.task.config.env.input_shape)
            )
            init_data = jnp.ones(
                tuple(init_shape),
                model_dtype,
            )
            params = self.task.model(self.task.config).init(jnp.array(rng), init_data)

            # init optimizers
            def init_fn(op, params=params):
                """Initialize an optax optimizer"""
                return op.init(params)

            states = map(self.task.optimizers, init_fn)

            return params, states
