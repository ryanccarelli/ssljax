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
        params, states = self.epoch(params, states)

    def epoch(self, params, states):
        # TODO: should we use dataloaders
        for data, _ in iter(self.task.dataloader):
            batch = jax.device_put(data)
            rngkeys = jax.random.split(self.rng, len(self.task.pipelines) + 1)
            self.rng = rngkeys[-1]
            batch = list(
                map(
                    lambda rng, pipeline: pipeline(batch, rng),
                    rngkeys[:-1],
                    self.task.pipelines,
                )
            )
            params, states = self.step(batch, params, states)
        # TODO: meter must implement distributed version
        # batch_metrics = jax.tree_multimap(lambda *xs: np.array(xs), *batch_metrics)
        # epoch_metrics ={k:np.mean(batch_metrics[k], axis=0) for k in batch_metrics}
        # self.task.meter.get_epoch_metrics()
        return params, states

    def step(self, batch, params, states):
        """
        Compute gradients, loss, accuracy per batch
        """
        step = states[0].count
        # TODO: correctly get step from the state
        grad_fn = jax.value_and_grad(self.loss)
        print("grad fn is: ", grad_fn)
        print("params is: ", params)
        print("batch is: ", batch)
        (aux), grad = grad_fn(params, batch)
        grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name="batch"), grads)

        def update_fn(_opt, _grads, _state, _params):
            _update, _state = _opt.update(_grads, _state, _params)
            if isinstance(_opt, GradientTransformation):
                _params = optax.apply_updates(_params, _update)
            elif isinstance(_opt, ParameterTransformation):
                _params = _update
            return _params, _state

        for idx, opt in enumerate(self.task.optimizers):
            params, states[idx] = update_fn(opt, grads, states, params)

        # TODO: call meter (aux)
        return params, states

    def loss(self, params, batch):
        outs = self.model.apply(params, batch)
        loss = self.task.loss(*outs)
        print(loss)
        loss = jax.lax.pmean(loss, axis_name="batch")
        return loss

    def eval(self):
        raise NotImplementedError

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
                eval(self.task.config.dataloader.params.input_shape)
            )

            init_data = jnp.ones(tuple(init_shape), model_dtype,)

            self.model = self.task.model(config=self.task.config)
            params = self.model.init(rng, init_data)

            states = list(
                map(lambda opt: opt.init(params), self.task.optimizers.values())
            )

            return params, states
