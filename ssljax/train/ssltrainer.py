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
from ssljax.core.utils import register, get_from_register
from ssljax.optimizers.base import ParameterTransformation
from ssljax.optimizers import Optimizer
from ssljax.train import Trainer
from typing import Callable
from functools import reduce
import flax
from flax import traverse_util
from ssljax.optimizers.utils import add_prefix_to_dict_keys, flattened_traversal

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn
label_fn = map_nested_fn(lambda k, _: k)


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
        state = self.initialize()
        state = jax_utils.replicate(state)
        state = self.epoch(state)


    def epoch(self, state):
        step = 0
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
            # TODO: p_step
            state = self.p_step(state, batch)

            # post process
            for fun in self.task.post_process_funcs:
                state = state.replace(params=fun(state.params))

        # TODO: meter must implement distributed version
        # batch_metrics = jax.tree_multimap(lambda *xs: np.array(xs), *batch_metrics)
        # epoch_metrics ={k:np.mean(batch_metrics[k], axis=0) for k in batch_metrics}
        # self.task.meter.get_epoch_metrics()
        return state

    def step(self, state, batch):
        """
        Compute gradients, loss, accuracy per batch
        """
        logger.info("In step",)
        # TODO. Enable dynamic scaling.  use flax/wmt as reference for this.
        if "dynamic_scale" in self.task.config.env and False:
            grad_fn = jax.jit(
                optim.DynamicScale(
                    **self.task.config.env.dynamic_scale.params
                ).value_and_grad(self.loss, has_aux=False)
            )
            # optim.DynamicScale returns a DynamicScaleResult object
            dyn_scale, is_fin, loss, grad = grad_fn(state.params, batch)
        else:
            grad_fn = jax.value_and_grad(self.loss, has_aux=False)
        loss_values, grads = grad_fn({'params': state.params}, batch)
        # TODO. What the hell is this inconsistency here? Grad function doesn't
        # work without params having root key as 'params' and then grads
        # require us to strip off the 'params' key?
        new_state = state.apply_gradients(grads=grads['params'])
        return new_state

    def loss(self, params, batch):
        outs = self.model.apply(params, batch)
        loss = self.task.loss(*outs)
        loss = jnp.mean(loss)
        return loss

    def eval(self):
        raise NotImplementedError

    def evalstep(self):
        raise NotImplementedError


    def initialize(self):

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

        self.model = self.task.model(config=self.task.config)



        init_shape = (
            [self.task.config.dataloader.params.batch_size]
            + list(eval(self.task.config.dataloader.params.input_shape))
            + [len(self.task.config.model.branches)]
        )

        init_data = jnp.ones(tuple(init_shape), model_dtype,)
        params = jax.jit(self.model.init)(self.rng, init_data)


        # TODO. Restore checkpoint
        if self.task.config.env.checkpoint:
            params, state = self._load_from_checkponit(
                self.task.config.load_from_checkpoint
            )

        # TODO. Add support for optimizer collection and multiple optimizer.
        # masked requires zero gradients.  This is a workaround that needs to be fixed.
        zerog = get_from_register(Optimizer, "zerog")

        opt_collect = []
        # TODO. Should we make this a config parameter? Or extract from params?
        prefix_branch = lambda x : 'branches_' + str(x)
        for opt_idx, opt in self.task.optimizers.items():
            opt_collect.append(
                optax.masked(
                    opt,
                    mask=flattened_traversal(lambda path, _:
                                             path.split('/')[0] == prefix_branch(opt_idx))))

        multi_tx = optax.chain(*opt_collect)
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params['params'].unfreeze(),
            tx=multi_tx
        )
        return state

