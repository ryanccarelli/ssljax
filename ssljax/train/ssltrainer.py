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

def add_prefix_to_dict_keys(d, prefix="branches_"):
    return {prefix+str(k): v for k, v in d.items()}


def flattened_traversal(fn):
  def mask(data):
      flat = {'/'.join(k): v for k, v in traverse_util.flatten_dict(data).items()}
      x = traverse_util.unflatten_dict({tuple(k.split('/')): fn(k, v) for k, v in flat.items()})
      return  x
  return mask


# def flattened_traversal(fn):
#   def mask(data):
#     flat = traverse_util.flatten_dict(data)
#     return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
#   return mask



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
        state = jax_utils.replicate(self.initialize())
        new_state = self.epoch(state)

        # post process
        for fun in self.task.post_process_funcs:
            state.params = fun(state.params)


    def epoch(self, state):
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
        # TODO: meter must implement distributed version
        # batch_metrics = jax.tree_multimap(lambda *xs: np.array(xs), *batch_metrics)
        # epoch_metrics ={k:np.mean(batch_metrics[k], axis=0) for k in batch_metrics}
        # self.task.meter.get_epoch_metrics()
        return state

    def step(self, state, batch):
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
            dyn_scale, is_fin, loss, grad = grad_fn(state.params, batch)
        else:
            grad_fn = jax.value_and_grad(self.loss, has_aux=False)
        grads = grad_fn(state.params, batch)

        new_state = state.apply_gradients(grads=grads)
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
        print("prefix", add_prefix_to_dict_keys(self.task.optimizers))
        print("opt,", self.task.optimizers[0])
        multi_tx = optax.chain(
            optax.masked(self.task.optimizers[0],
                         mask=flattened_traversal(lambda path, _: path.split('/')[0] == 'branches_0')),
            optax.masked(zerog(),
                         mask=flattened_traversal(lambda path, _: path.split('/')[0] != 'branches_0')))

        print("multi_tx", multi_tx)

        print("params in multi opt,", multi_tx.init(params['params'].unfreeze()))
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params['params'].unfreeze(),
            tx=multi_tx
        )
        return state.replace(params=params['params'])

