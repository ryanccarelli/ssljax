import logging
import cProfile, pstats

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
from ssljax.core.utils import (add_prefix_to_dict_keys, flattened_traversal,
                               get_from_register, register)
from ssljax.optimizers import Optimizer
from ssljax.train.trainer import Trainer
from ssljax.train.trainstate import TrainState
from tensorboardX import GlobalSummaryWriter
from tqdm import tqdm

CHECKPOINTSDIR = Path("outs/checkpoints/")
CHECKPOINTSDIR.mkdir(parents=True, exist_ok=True)
TBDIR = Path("outs/tensorboard/")
TBDIR.mkdir(parents=True, exist_ok=True)

writer = GlobalSummaryWriter(TBDIR)

from jax import lax
# TODO. Move to utils
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


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
        self.bn = None

    def train(self):
        """
        Train model.
        """

        from jax.lib import xla_bridge
        print("xla bridge device", xla_bridge.get_backend().platform)

        key, self.rng = random.split(self.rng)
        state, p_step = self.initialize()

        state = jax_utils.replicate(state)

        for epoch in tqdm(range(self.task.config.env.epochs)):
            state = self.epoch(state, p_step)
            checkpoints.save_checkpoint(
                target=state,
                step=epoch,
                prefix="checkpoint_",
                **self.task.config.env.save_checkpoint.params,
            )

    def epoch(self, state, p_step):
        """
        Train over one iteration of data.

        Args:
            state (flax.training.train_state.TrainState): model state
        """
        profiler = cProfile.Profile()
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
            state, loss = p_step(state, batch)
            # post process
            for idx, fun in enumerate(self.task.post_process_funcs):
                state = state.replace(params=fun(state.params))
            state = sync_batch_stats(state)
            writer.add_scalar("loss", np.array(loss).mean())
        return state

    def step(
        self,
        state,
        batch,
        has_batch_stats,
        dynamic_scale,
        has_aux,
        mutable_keys,
        loss_fn,
        dynamic_scale_params,
        accumulate_steps,
    ):
        """
        Compute and apply gradient.

        Args:
            state (flax.training.train_state.TrainState): model state
            batch (jnp.array): a single data batch
        """
        # get losses
        accumulate_gradients = partial(
            self.accumulate_gradients,
            accumulate_steps=accumulate_steps,
            has_aux=has_aux,
            mutable_keys=mutable_keys,
            dynamic_scale=dynamic_scale,
        )

        if dynamic_scale:
            # optim.DynamicScale returns a DynamicScaleResult object
            grad_fn = jax.jit(
                optim.DynamicScale(dynamic_scale_params).value_and_grad(
                    loss_fn, has_aux=has_aux
                )
            )
        else:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)

        if has_aux:
            loss, aux, grad = accumulate_gradients(
                grad_fn,
                batch,
                {"params": state.params, "batch_stats": state.batch_stats},
                has_aux=has_aux,
                mutable_keys=mutable_keys,
                dynamic_scale=dynamic_scale,
            )
            aux = (jax.lax.pmean(aux, axis_name="batch"),)
        else:
            loss, grad = accumulate_gradients(
                grad_fn,
                batch,
                {"params": state.params},
                has_aux=has_aux,
                dynamic_scale=dynamic_scale,
            )
            aux = None
        loss, grad = (
            jax.lax.pmean(loss, axis_name="batch"),
            jax.lax.pmean(grad, axis_name="batch"),
        )
        if has_aux:
            state = state.apply_gradients(
                grads=grad["params"],
                batch_stats=aux[0].unfreeze(),
            )
        else:
            state = state.apply_gradients(
                grads=grad["params"],
            )
        return state, loss

    def accumulate_gradients(
        self,
        grad_fn,
        batch,
        params,
        mutable_keys=[],
        has_aux=False,
        dynamic_scale=False,
        accumulate_steps=1,
    ):
        """
        Split a batch into sub-batches and compute gradients in each sub-batch.

        Args:
            grad_fn (Callable[..., Tuple[Any, Any]]): result of opt.value_and_gradient
            batch (jnp.array): a single data batch
            params (flax.core.frozen_dict.FrozenDict[str, Any]): model parameters
        """
        if accumulate_steps > 1:
            assert (
                batch.shape[0] % accumulate_steps == 0
            ), f"Bad accum_steps {self.task.config.env.accum_steps} for batch size {batch.shape[0]}"
            step_size = batch.shape[0] // accumulate_steps
            # TODO: why do we need to pop params here?
            if dynamic_scale:
                dyn_scale, is_fin, (loss_and_aux), grad = grad_fn(
                    {"params": params["params"]}, batch
                )
            else:
                loss_and_aux, grad = grad_fn({"params": params["params"]}, batch)
            if has_aux:
                (loss, aux) = loss_and_aux
            else:
                loss = loss_and_aux

            def acc_grad_and_loss(i, loss_and_grad):
                btch = jax.lax.dynamic_slice(
                    batch, (i * step_size, 0, 0), (step_size,) + batch.shape[1:]
                )
                if dynamic_scale:
                    dyn_scale, is_fin, loss_i, grad_i = grad_fn(
                        {"params": params["params"]}, btch
                    )
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
                1, accumulate_steps, acc_grad_and_loss, (loss, grad)
            )
            if has_aux:
                return jax.tree_map(lambda x: x / accumulate_steps, (loss, aux, grad))
            else:
                return jax.tree_map(lambda x: x / accumulate_steps, (loss, grad))
        else:
            if dynamic_scale:
                dyn_scale, is_fin, (loss_and_aux), grad = grad_fn(
                    {"params": params}, batch
                )
            else:
                loss_and_aux, grad = grad_fn(params, batch)
            if has_aux:
                (loss, aux) = loss_and_aux
                return loss, aux, grad
            else:
                loss = loss_and_aux
                return loss, grad

    def loss(self, params, batch, model_fn, mutable_keys=None):
        """
        Apply loss function. Passed to opt.value_and_grad.
        """
        # loss must return a single float (since we grad loss)
        # but here we have also new_state
        # but we need new_state to manage mutable batch_params
        new_state = None
        if mutable_keys:
            outs, new_state = model_fn.apply(params, batch, mutable=mutable_keys)
            loss = self.task.loss(*outs)
            loss = jnp.mean(loss)

            return loss, new_state
        else:
            outs = self.model.apply(params, batch)
            loss = self.task.loss(*outs)
            loss = jnp.mean(loss)
            return loss

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
            [1] # Batch size == 1
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
                batch_stats=None,
            )

        # Create init steps
        has_batch_stats = state.batch_stats is not None
        has_aux = has_batch_stats
        dynamic_scale = self.task.config.env.dynamic_scale
        mutable_keys = None
        if has_batch_stats:
            mutable_keys = ["batch_stats"]

        if mutable_keys is not None:
            loss_fn = partial(self.loss, mutable_keys=mutable_keys)
        else:
            loss_fn = self.loss
        loss_fn = partial(loss_fn, model_fn=self.model)

        if dynamic_scale:
            dynamic_scale_params = self.task.config.env.dynamic_scale.params
        else:
            dynamic_scale_params = {}

        p_step = jax.pmap(
            partial(
                self.step,
                has_batch_stats=has_batch_stats,
                dynamic_scale=dynamic_scale,
                has_aux=has_aux,
                mutable_keys=mutable_keys,
                loss_fn=loss_fn,
                dynamic_scale_params=dynamic_scale_params,
                accumulate_steps=self.task.config.env.accum_steps,
            ),
            axis_name="batch",
            donate_argnums=(0, 1),
        )

        return state, p_step
