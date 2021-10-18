import logging

logger = logging.getLogger(__name__)

from functools import partial, reduce
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
from jax import random
from optax._src.base import GradientTransformation
from ssljax.core.utils import (add_prefix_to_dict_keys, flattened_traversal,
                               get_from_register, register)
from ssljax.optimizers import Optimizer
from ssljax.optimizers.base import ParameterTransformation
from ssljax.train.trainer import Trainer
from ssljax.train.trainstate import TrainState
from tensorboardX import GlobalSummaryWriter

CHECKPOINTSDIR = Path("outs/checkpoints/")
CHECKPOINTSDIR.mkdir(parents=True, exist_ok=True)
TBDIR = Path("outs/tensorboard/")
TBDIR.mkdir(parents=True, exist_ok=True)

writer = GlobalSummaryWriter(TBDIR)


@register(Trainer, "SSLTrainer")
class SSLTrainer(Trainer):
    """
    Implements self-supervised training and inference.

    Args:
        rng (jnp.DeviceArray): A Jax PRNG key.
        task (ssljax.train.task.Task): A task object.
    """

    def __init__(self, rng, task):
        self.task = task
        self.rng = rng
        self.p_step = jax.pmap(self.step, axis_name="batch")
        self.bn = None

    def train(self):
        """
        Train model.
        """
        key, self.rng = random.split(self.rng)
        state = self.initialize()
        state = jax_utils.replicate(state)
        for epoch in range(self.task.config.env.epochs):
            print(f"epoch: {epoch}")
            state = self.epoch(state)
            checkpoints.save_checkpoint(
                target=state,
                step=epoch,
                prefix="checkpoint_",
                **self.task.config.env.save_checkpoint.params,
            )

    def epoch(self, state):
        """
        Train over one iteration of data.

        Args:
            state (flax.training.train_state.TrainState): model state
        """
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
            writer.add_scalar("loss", np.array(loss).mean())
        return state

    def step(self, state, batch):
        """
        Compute and apply gradient.

        Args:
            state (flax.training.train_state.TrainState): model state
            batch (jnp.array): a single data batch
        """
        # get losses
        has_aux = False
        if state.batch_stats:
            has_aux = True
            mutable_keys = ["batch_stats"]
            loss_fn = partial(self.loss, mutable_keys=mutable_keys)
        else:
            loss_fn = self.loss

        if "dynamic_scale" in self.task.config.env:
            # optim.DynamicScale returns a DynamicScaleResult object
            grad_fn = jax.jit(
                optim.DynamicScale(
                    **self.task.config.env.dynamic_scale.params
                ).value_and_grad(loss_fn, has_aux=has_aux)
            )
        else:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)

        if has_aux:
            loss, aux, grad = self.accumulate_gradients(
                grad_fn,
                batch,
                {"params": state.params, "batch_stats": state.batch_stats},
                has_aux=has_aux,
                mutable_keys=mutable_keys,
            )
            aux = (jax.lax.pmean(aux, axis_name="batch"),)
        else:
            loss, grad = self.accumulate_gradients(
                grad_fn,
                batch,
                {"params": state.params, "batch_stats": state.batch_stats},
                has_aux=has_aux,
                mutable_keys=mutable_keys,
            )
            aux = None
        loss, grad = (
            jax.lax.pmean(loss, axis_name="batch"),
            jax.lax.pmean(grad, axis_name="batch"),
        )
        state = state.apply_gradients(
            grads=grad["params"],
            batch_stats=aux[0].unfreeze(),
        )
        return state, loss

    def accumulate_gradients(
        self, grad_fn, batch, params, has_aux=False, mutable_keys=None
    ):
        """
        Split a batch into sub-batches and compute gradients in each sub-batch.

        Args:
            grad_fn (Callable[..., Tuple[Any, Any]]): result of opt.value_and_gradient
            batch (jnp.array): a single data batch
            params (flax.core.frozen_dict.FrozenDict[str, Any]): model parameters
        """
        if self.task.config.env.accum_steps > 1:
            assert (
                batch.shape[0] % self.task.config.env.accum_steps == 0
            ), f"Bad accum_steps {self.task.config.env.accum_steps} for batch size {batch.shape[0]}"
            step_size = batch.shape[0] // self.task.config.env.accum_steps
            if "dynamic_scale" in self.task.config.env:
                dyn_scale, is_fin, (loss_and_aux), grad = grad_fn(params, batch)
            else:
                loss_and_aux, grad = grad_fn(params, batch)
            if has_aux:
                (loss, aux) = loss_and_aux
            else:
                loss = loss_and_aux

            def acc_grad_and_loss(i, loss_and_grad):
                btch = jax.lax.dynamic_slice(
                    batch, (i * step_size, 0, 0), (step_size,) + batch.shape[1:]
                )
                if "dynamic_scale" in self.task.config.env:
                    dyn_scale, is_fin, loss_i, grad_i = grad_fn(params, btch)
                else:
                    loss_i, grad_i = grad_fn(params, btch)
                grad_i = jax.lax.pmean(grad_i, axis_name="batch")
                if has_aux:
                    loss_i, aux_i = loss_i
                    return (
                        loss + loss_i,
                        aux + aux_i,
                        jax.tree_multimap(lambda x, y: x + y, grad, grad_i),
                    )
                else:
                    return (
                        loss + loss_i,
                        jax.tree_multimap(lambda x, y: x + y, grad, grad_i),
                    )

            loss, grad = jax.lax.fori_loop(
                1, self.task.config.env.accum_steps, acc_grad_and_loss, (loss, grad)
            )
            if has_aux:
                return jax.tree_map(
                    lambda x: x / self.task.config.env.accum_steps, (loss, aux, grad)
                )
            else:
                return jax.tree_map(
                    lambda x: x / self.task.config.env.accum_steps, (loss, grad)
                )
        else:
            if "dynamic_scale" in self.task.config.env:
                dyn_scale, is_fin, (loss_and_aux), grad = grad_fn(params, batch)
            else:
                loss_and_aux, grad = grad_fn(params, batch)
            if has_aux:
                (loss, aux) = loss_and_aux
                return loss, aux, grad
            else:
                loss = loss_and_aux
                return loss, grad

    def loss(self, params, batch, mutable_keys=None):
        """
        Apply loss function. Passed to opt.value_and_grad.
        """
        # TODO: {'params': params, 'batch_stats': state.batch_stats}, mutable=['batch_stats']
        # from https://github.com/google/flax/blob/main/examples/imagenet/train.py
        # how to get the batch stats in here?
        new_state = None

        if mutable_keys:
            outs, new_state = self.model.apply(params, batch, mutable=mutable_keys)
        else:
            outs = self.model.apply(params, batch)
        loss = self.task.loss(*outs)
        loss = jnp.mean(loss)
        return loss, new_state

    def eval(self):
        raise NotImplementedError

    def evalstep(self):
        raise NotImplementedError

    def initialize(self):
        """
        Initialize platform, devices, numerical precision, model, train state.
        """
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

        init_data = jnp.ones(
            tuple(init_shape),
            model_dtype,
        )
        params = jax.jit(self.model.init)(self.rng, init_data)

        # TODO: check whether there are batchnorm params to manage
        # check whether BatchNorm_ in any leafdict

        if self.task.config.env.restore_checkpoint.params:
            state = checkpoints.restore_checkpoint(
                target=self.model,
                **self.task.config.env.restore_checkpoint,
            )

        opt_collect = []
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

        if "batch_stats" in params.keys():
            state = TrainState.create(
                apply_fn=self.model.apply,
                params=params["params"].unfreeze(),
                tx=multi_tx,
                batch_stats=params["batch_stats"].unfreeze(),
            )
        else:
            state = TrainState.create(
                apply_fn=self.model.apply,
                params=params["params"].unfreeze(),
                tx=multi_tx,
            )

        return state


def sync_batch_stats(state):
    """
    Sync batch statistics across replicas.
    Adapted from Flax: https://github.com/google/flax/blob/main/examples/imagenet/train.py
    """
    cross_replica_mean = jax.pmap(lambda x: flax.lax.pmean(x, "x"), "x")
    return state.replicate(batch_stats=cross_replica_mean(state.batch_stats))
