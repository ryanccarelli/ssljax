import logging

logger = logging.getLogger(__name__)

from functools import reduce
from pathlib import Path
from typing import Callable

import flax
import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils, traverse_util
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import random
from optax._src.base import GradientTransformation
from ssljax.core.utils import get_from_register, register
from ssljax.optimizers import Optimizer
from ssljax.optimizers.base import ParameterTransformation
from ssljax.optimizers.utils import (add_prefix_to_dict_keys,
                                     flattened_traversal)
from ssljax.train import Trainer
from tensorboardX import GlobalSummaryWriter

CHECKPOINTSDIR = Path("outs/checkpoints/")
CHECKPOINTSDIR.mkdir(parents=True, exist_ok=True)
TBDIR = Path("outs/tensorboard/")
TBDIR.mkdir(parents=True, exist_ok=True)

writer = GlobalSummaryWriter(TBDIR)


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
        for epoch in range(self.task.config.env.epochs):
            state = self.epoch(state)
            checkpoints.save_checkpoint(
                target=state,
                step=epoch,
                prefix="checkpoint_",
                **self.task.config.env.save_checkpoint.params,
            )

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
            state, loss = self.p_step(state, batch)

            # post process
            for fun in self.task.post_process_funcs:
                state = state.replace(params=fun(state.params))

            writer.add_scalar("loss", np.array(loss))
        return state

    def step(self, state, batch):
        """
        Compute gradients, loss, accuracy per batch
        """
        # get losses
        if "dynamic_scale" in self.task.config.env:
            # optim.DynamicScale returns a DynamicScaleResult object
            grad_fn = jax.jit(
                optim.DynamicScale(
                    **self.task.config.env.dynamic_scale.params
                ).value_and_grad(self.loss, has_aux=False)
            )
            # optim.DynamicScale returns a DynamicScaleResult object
            dyn_scale, is_fin, loss, grad = grad_fn(state.params, batch)
        else:
            grad_fn = jax.value_and_grad(self.loss, has_aux=False)

        loss, grad = self.accumulate_gradients(grad_fn, batch, {"params": state.params})

        state = state.apply_gradients(grads=grad["params"])
        return state, loss

    def accumulate_gradients(self, grad_fn, batch, params):
        if self.task.config.env.accum_steps > 1:
            assert (
                batch.shape[0] % self.task.config.env.accum_steps == 0
            ), f"Bad accum_steps {self.task.config.env.accum_steps} for batch size {batch.shape[0]}"
            step_size = batch.shape[0] // self.task.config.env.accum_steps
            if "dynamic_scale" in self.task.config.env:
                dyn_scale, is_fin, loss, grad = grad_fn(params, batch)
            else:
                loss, grad = grad_fn(params, batch)

            def acc_grad_and_loss(i, loss_and_grad):
                btch = jax.lax.dynamic_slice(
                    batch, (i * step_size, 0, 0), (step_size,) + batch.shape[1:]
                )
                if "dynamic_scale" in self.task.config.env:
                    dyn_scale, is_fin, loss_i, grad_i = grad_fn(params, btch)
                else:
                    loss_i, grad_i = grad_fn(params, btch)
                grad_i = jax.lax.pmean(grad_i, axis_name="batch")
                return (
                    loss + loss_i,
                    jax.tree_multimap(lambda x, y: x + y, grad, grad_i),
                )

            loss, grad = jax.lax.fori_loop(
                1, self.task.config.env.accum_steps, acc_grad_and_loss, (loss, grad)
            )
            return jax.tree_map(
                lambda x: x / self.task.config.env.accum_steps, (loss, grad)
            )
        else:
            return grad_fn(params, batch)

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
        if self.task.config.env.restore_checkpoint.params:
            state = checkpoints.restore_checkpoint(
                target=self.model, **self.task.config.env.restore_checkpoint,
            )

        opt_collect = []
        # TODO. Should we make this a config parameter? Or extract from params?
        prefix_branch = lambda x: "branch_" + str(x)
        for opt_idx, opt in self.task.optimizers.items():
            opt_collect.append(
                optax.masked(
                    opt,
                    mask=flattened_traversal(
                        lambda path, _: path.split("/")[0] == prefix_branch(opt_idx)
                    ),
                )
            )

        multi_tx = optax.chain(*opt_collect)
        state = TrainState.create(
            apply_fn=self.model.apply, params=params["params"].unfreeze(), tx=multi_tx
        )
        return state
