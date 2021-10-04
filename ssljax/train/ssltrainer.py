import logging

logger = logging.getLogger(__name__)

from pathlib import Path

import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training.train_state import TrainState
from flax.training import checkpoints
from jax import random
from optax._src.base import GradientTransformation
from ssljax.core.utils import register
from ssljax.optimizers.base import ParameterTransformation
from ssljax.train import Trainer

CHECKPOINTSDIR = Path("outs/checkpoints/")
CHECKPOINTSDIR.mkdir(parents=True, exist_ok=True)


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
        self.p_step = jax.pmap(self.step, axis_name="batch")

    def train(self):
        key, self.rng = random.split(self.rng)
        params, states = self.initialize(jax.random.split(key, jax.device_count()))
        # TODO: iterate over epochs
        for epoch in range(self.task.config.env.epochs):
            params, states = self.epoch(params, states)
            for idx, state in enumerate(states):
                checkpoints.save_checkpoint(
                    target=state,
                    step=epoch,
                    prefix=f"checkpoint_{idx}_",
                    **self.task.config.env.save_checkpoint.params,
                )

    def epoch(self, params, states):
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
            batch = jnp.stack(batch, axis=-1)
            batch = jax_utils.replicate(batch)
            params, states = self.p_step(batch, params, states)
        # TODO: meter must implement distributed version
        # batch_metrics = jax.tree_multimap(lambda *xs: np.array(xs), *batch_metrics)
        # epoch_metrics ={k:np.mean(batch_metrics[k], axis=0) for k in batch_metrics}
        # self.task.meter.get_epoch_metrics()
        return params, states

    def step(self, batch, params, states):
        """
        Compute gradients, loss, accuracy per batch
        """
        if "dynamic_scale" in self.task.config.env:
            grad_fn = jax.jit(
                optim.DynamicScale(
                    **self.task.config.env.dynamic_scale.params
                ).value_and_grad(self.loss, has_aux=False)
            )
            # optim.DynamicScale returns a DynamicScaleResult object
            dyn_scale, is_fin, loss, grad = grad_fn(params, batch)
        else:
            grad_fn = jax.value_and_grad(self.loss, has_aux=False)
            grad = grad_fn(params, batch)
        grad = jax.lax.pmean(grad, axis_name="batch")

        def update_fn(_opt, _grads, _state, _params):
            _update, _state = _opt.update(_grads, _state, _params)
            if isinstance(_opt, GradientTransformation):
                if "dynamic_scale" in self.task.config.env:
                    _params = optax.apply_updates(_params, _update)
                else:
                    _params = optax.apply_updates(_params, _update[1])
            elif isinstance(_opt, ParameterTransformation):
                _params = _update
            return _params, _state

        for idx, opt in self.task.optimizers.items():
            update, states[idx] = update_fn(opt, grad, states[idx], params)
        # TODO: call meter (aux)
        return params, states

    def loss(self, params, batch):
        outs = self.model.apply(params, batch)
        loss = self.task.loss(*outs)
        loss = jnp.mean(loss)
        return loss

    def eval(self):
        raise NotImplementedError

    def evalstep(self):
        raise NotImplementedError

    def initialize(self, rng):
        @jax.pmap
        def get_initial_params(rng):
            init_shape = (
                [self.task.config.dataloader.params.batch_size]
                + list(eval(self.task.config.dataloader.params.input_shape))
                + [len(self.task.config.model.branches)]
            )
            init_data = jnp.ones(
                tuple(init_shape),
                model_dtype,
            )
            params = self.model.init(rng, init_data)
            return params

        # setup devices
        platform = jax.local_devices()[0].platform
        # set model_dtype
        model_dtype = jnp.float32
        # TODO: cast different parts of model to different precisions
        if self.task.config.env.half_precision:
            if platform == "tpu":
                model_dtype = jnp.bfloat16
            else:
                model_dtype = jnp.float16
        else:
            model_dtype = jnp.float32
        # init model
        self.model = self.task.model(config=self.task.config)
        # init training state
        if self.task.config.env.restore_checkpoint.params:
            params, state = checkpoints.restore_checkpoint(
                target=self.model,
                **self.task.config.env.restore_checkpoint,
            )
        else:
            # init model
            params = get_initial_params(rng)
            states = list(
                map(lambda opt: opt.init(params), self.task.optimizers.values())
            )
            return params, states
