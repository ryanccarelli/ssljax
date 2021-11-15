import logging

logger = logging.getLogger(__name__)

import time
from functools import partial
from pathlib import Path

import flax
import flax.optim as optim
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.core import freeze, unfreeze
from flax.training import checkpoints
from flax.training.checkpoints import restore_checkpoint
from jax import random
from omegaconf import DictConfig
from ssljax.core.utils import flattened_traversal, register
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
cross_replica_mean = jax.pmap(
    lambda x: lax.pmean(x, "x"),
    "x",
)


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    new_mutable_states = state.mutable_states.unfreeze()
    new_mutable_states["batch_stats"] = cross_replica_mean(state.mutable_states["batch_stats"])
    state.replace(mutable_states=new_mutable_states)
    return state


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
            save_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
            checkpoints.save_checkpoint(
                target=save_state,
                step=int(jax_utils.unreplicate(state.step)),
                prefix="checkpoint_",
                **self.task.config.env.save_checkpoint.params,
            )

    def epoch(self, state, p_step):
        """
        Train over one iteration of data.

        Args:
            state (flax.training.train_state.TrainState): model state
        """
        for data, _ in iter(self.task.dataloader):
            # TODO: need to wrap torch and tfds dataloaders so they return a consistent format
            batch = jax.device_put(data)

            self.rng, rng_step = jax.random.split(self.rng)

            rng_step = jax_utils.replicate(rng_step)
            batch = jax_utils.replicate(batch)
            start_time = time.time()
            state, loss = p_step(state, batch, rng_step)

            end_time = time.time()
            print("Exec time: ",end_time-start_time)

            # Sync batch stats across multiple devices
            if "batch_stats" in state.mutable_states.keys():
                state = sync_batch_stats(state)
            writer.add_scalar("loss", np.array(loss).mean())
        return state

    def step(
        self,
        state,
        batch,
        rng,
        mutable_keys,
        loss_fn,
        dynamic_scale,
        dynamic_scale_params,
        accumulate_steps,
    ):
        """
        Compute and apply gradient.

        Args:
            state (flax.training.train_state.TrainState): model state
            batch (jnp.array): a single data batch
        """
        rng_pre, rng = jax.random.split(rng)
        batch = self.task.pre_pipelines(batch, rng_pre)
        postpiperngs = jax.random.split(rng, len(self.task.post_pipelines) )
        batch = list(
            map(
                lambda rng, pipeline: pipeline(batch, rng),
                postpiperngs,
                self.task.post_pipelines,
                )
            )
        batch = jnp.stack(batch, axis=-1)

        accumulate_gradients = partial(
            self.accumulate_gradients,
            accumulate_steps=accumulate_steps,
            mutable_keys=mutable_keys,
            dynamic_scale=dynamic_scale,
        )

        if dynamic_scale:
            # optim.DynamicScale returns a DynamicScaleResult object
            grad_fn = jax.jit(
                optim.DynamicScale(**dynamic_scale_params).value_and_grad(
                    loss_fn, has_aux=True
                )
            )
        else:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        state_params = {"params": state.params}
        for mutable_key in mutable_keys:
            if (val := getattr(state, mutable_key, None)) is not None:
                state_params[mutable_key] = val

        (loss, grad), aux = accumulate_gradients(
            grad_fn,
            batch,
            state_params
        )
        loss, grad = (
            jax.lax.pmean(loss, axis_name="batch"),
            jax.lax.pmean(grad, axis_name="batch"),
        )


        if ("mutable_states" in aux):
            state = state.apply_gradients(
                grads=grad["params"],
                **{"mutable_states": aux["mutable_states"]},
            )
        else:
            state = state.apply_gradients(
                grads=grad["params"],
            )

        for idx, fun in enumerate(self.task.post_process_funcs):
            state = state.replace(params=fun(state.params, state.step))

        return state, loss

    def accumulate_gradients(
        self,
        grad_fn,
        batch,
        params,
        mutable_keys=[],
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
            assert (
                "batch_stats" not in params
            ), f"Batchnorm not supported when accumulating gradients"
            step_size = batch.shape[0] // accumulate_steps
            if dynamic_scale:
                dyn_scale, is_fin, (loss_and_aux), grad = grad_fn(
                    {"params": params["params"]}, batch
                )
            else:
                loss_and_aux, grad = grad_fn({"params": params["params"]}, batch)
            (loss, aux) = loss_and_aux


            def acc_grad_and_loss(i, loss_and_grad):
                btch = jax.lax.dynamic_slice(
                    batch, (i * step_size, 0, 0), (step_size,) + batch.shape[1:]
                )
                if dynamic_scale:
                    dyn_scale, is_fin, loss_and_aux, grad_i = grad_fn(
                        {"params": params["params"]}, btch
                    )
                else:
                    loss_and_aux, grad_i = grad_fn(params, btch)
                loss_i, aux = loss_and_aux
                grad_i = jax.lax.pmean(grad_i, axis_name="batch")

                return (
                    loss + loss_i,
                    jax.tree_multimap(lambda x, y: x + y, grad, grad_i),
                    aux
                )

            loss, grad, aux = jax.lax.fori_loop(
                1, accumulate_steps, acc_grad_and_loss, (loss, grad, aux)
            )
            return jax.tree_map(lambda x: x / accumulate_steps, (loss, grad)), aux
        else:
            if dynamic_scale:
                dyn_scale, is_fin, loss_and_aux, grad = grad_fn(
                    params, batch
                )
            else:
                loss_and_aux, grad = grad_fn(params, batch)

            (loss, aux) = loss_and_aux
            return (loss, grad), aux

    def loss(self, params, batch, model_fn, mutable_keys=None):
        """
        Apply loss function. Passed to opt.value_and_grad.
        """
        # loss must return a single float (since we grad loss)
        # but here we have also new_state
        # but we need new_state to manage mutable batch_params
        aux = {}
        if mutable_keys:
            outs, new_state = model_fn.apply(params, batch, mutable=mutable_keys)
            loss = self.task.loss(*outs)
            loss = jnp.mean(loss)

            aux['mutable_states'] = new_state
            return loss, aux
        else:
            outs = self.model.apply(params, batch)
            loss = self.task.loss(*outs)
            loss = jnp.mean(loss)
            return loss, aux

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
            [1]  # Batch size == 1
            + list(eval(self.task.config.env.input_shape))
            + [len(self.task.config.model.branches)]
        )

        init_data = jnp.ones(
            tuple(init_shape),
            model_dtype,
        )
        params = self.model.init(self.rng, init_data)

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

        mutable_keys = list(params.keys())
        mutable_keys.remove("params")

        mutable_states = {}
        for mutable_key in mutable_keys:
            mutable_states[mutable_key] = params[mutable_key].unfreeze()


        state_params = {"apply_fn": self.model.apply,
                        "params": params["params"].unfreeze(),
                        "tx": multi_tx,
                        "mutable_states": mutable_states
                        }


        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params["params"].unfreeze(),
            tx=multi_tx,
            mutable_states=mutable_states
        )

        # load pretrained modules
        state = load_pretrained(self.task.config.model.branches, state)

        # load checkpoint
        if self.task.config.env.restore_checkpoint.params:
            state = checkpoints.restore_checkpoint(
                target=self.model,
                **self.task.config.env.restore_checkpoint,
            )

        loss_fn = partial(self.loss, model_fn=self.model, mutable_keys=mutable_keys)

        p_step = jax.pmap(
            partial(
                self.step,
                dynamic_scale=self.task.config.env.dynamic_scale,
                mutable_keys=mutable_keys,
                loss_fn=loss_fn,
                dynamic_scale_params=self.task.config.env.dynamic_scale_params,
                accumulate_steps=self.task.config.env.accum_steps,
            ),
            axis_name="batch",
            donate_argnums=(0, 1),
        )

        return state, p_step


def load_pretrained(config, state):
    """
    Overwrite branches or modules from pretrained checkpoint indicated in
    config.model.branches.

    Args:
        config (DictConfig): SSLTrainer config at task.config.model.branches
        state (flax.training.TrainState): model state
    """

    params = unfreeze(state.params)

    for branch_key, branch in config.items():
        if "pretrained" in branch:
            replace = restore_checkpoint(
                str(Path(__file__).parents[2]) + branch["pretrained"],
                target=None,
            )
            if "model_state" in replace:
                while "model_state" in replace:
                    replace = replace["model_state"]
            elif "params" in replace:
                while "params" in replace:
                    replace = replace["params"]
            else:
                raise Exception("checkpoint file structure not recognized")
            params["branch_{branch_key}"] = replace
        for stage_key, stage in branch["stages"].items():
            if stage_key != "stop_gradient":
                if "pretrained" in stage:
                    print(str(Path(__file__).parents[2]) + stage["pretrained"])
                    replace = restore_checkpoint(
                        str(Path(__file__).parents[2]) + stage["pretrained"],
                        target=None,
                    )
                    if "model_state" in replace:
                        while "model_state" in replace:
                            replace = replace["model_state"]
                    elif "params" in replace:
                        while "params" in replace:
                            replace = replace["params"]
                    else:
                        raise Exception("checkpoint file structure not recognized")
                    params[f"branch_{branch_key}"][stage_key] = replace
    state.replace(params=freeze(params))
    return state
