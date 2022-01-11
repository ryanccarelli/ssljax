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
from ssljax.core import flattened_traversal, register
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
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x",)


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    new_mutable_states = state.mutable_states.unfreeze()
    new_mutable_states["batch_stats"] = cross_replica_mean(
        state.mutable_states["batch_stats"]
    )
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
            p_step: pmapped step function
        """

        # get training steps
        steps_per_epoch = self.task.data.meta_data.get("num_train_examples", 0) // self.task.config.data.params.batch_size
        for step in range(steps_per_epoch):
            data = next(self.task.data.train_iter)
            data = data["inputs"]
            if self.task.config.pipelines.flatten:
                # TODO: This assumes 3D format (28, 28, 1); (256, 256, 3)
                data = data.reshape(*data.shape[:-3], -1)
            batch = jax.device_put(data)

            self.rng, rng_step = jax.random.split(self.rng)
            rng_step = jax_utils.replicate(rng_step)

            start_time = time.time()
            state, loss = p_step(state, batch, rng_step)

            end_time = time.time()
            print("Step time: ", end_time - start_time)

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
    ):
        """
        Compute and apply gradient.

        Args:
            state (flax.training.train_state.TrainState): model state
            batch (jnp.ndarray): a single data batch
            rng (jnp.ndarray): PRNG key
            mutable_keys (List[str]): parameters that are mutable
            loss_fn: loss
            dynamic_scale (bool): whether to apply dynamic scaling
            dynamic_scale_params (dict): params passed to dynamic scale optimizer
        """

        state.replace(global_step = state.global_step + 1)
        print(state.global_step)
        rng_pre, rng = jax.random.split(rng)
        if self.task.pre_pipelines:
            batch = self.task.pre_pipelines(batch, rng_pre)
        postpiperngs = jax.random.split(rng, len(self.task.post_pipelines))
        batch = list(
            map(
                lambda rng, pipeline: pipeline(batch, rng),
                postpiperngs,
                self.task.post_pipelines,
            )
        )
        # batch stores views indexed by the pipeline that produced them
        batch = {str(idx): val for idx, val in enumerate(batch)}

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

        if dynamic_scale:
            dyn_scale, is_fin, loss_and_aux, grad = grad_fn(state_params, batch)
        else:
            loss_and_aux, grad = grad_fn(state_params, batch)

        (loss, aux) = loss_and_aux

        loss, grad = (
            jax.lax.pmean(loss, axis_name="batch"),
            jax.lax.pmean(grad, axis_name="batch"),
        )

        if "mutable_states" in aux:
            state = state.apply_gradients(
                grads=grad["params"], **{"mutable_states": aux["mutable_states"]},
            )
        else:
            state = state.apply_gradients(grads=grad["params"],)

        for idx, fun in enumerate(self.task.post_process_funcs):
            state = state.replace(params=fun(state.params, state.step))

        return state, loss

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
            loss = self.task.loss(outs)
            loss = jnp.mean(loss)
            aux["mutable_states"] = new_state
            return loss, aux
        else:
            outs = self.model.apply(params, batch)
            loss = self.task.loss(outs)
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

        # TODO: remove
        self.model = self.task.model

        # .meta_data["input_shape"] is (-1, H, W, C)
        data_shape = self.task.data.meta_data["input_shape"][1:]
        # add batch dimension
        data_shape = (1,) + data_shape

        init_data = jnp.ones(data_shape, model_dtype,)

        if self.task.config.pipelines.flatten:
            # TODO: This assumes 3D format (28, 28, 1); (256, 256, 3)
            init_data = init_data.reshape(*init_data.shape[:-3], -1)

        init_shape = {}
        for branch, _ in self.task.config.pipelines.branches.items():
            init_shape[str(branch)] = init_data.copy()

        params = self.model.init(self.rng, init_shape)

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

        state_params = {
            "apply_fn": self.model.apply,
            "params": params["params"].unfreeze(),
            "tx": multi_tx,
            "mutable_states": mutable_states,
        }

        state = TrainState.create(
            global_step=0,
            apply_fn=self.model.apply,
            params=params["params"].unfreeze(),
            tx=multi_tx,
            mutable_states=mutable_states,
        )

        # load pretrained modules
        state = load_pretrained(self.task.config.modules, state)

        # load checkpoint
        if self.task.config.env.restore_checkpoint.params:
            state = checkpoints.restore_checkpoint(
                target=self.model, **self.task.config.env.restore_checkpoint,
            )

        loss_fn = partial(self.loss, model_fn=self.model, mutable_keys=mutable_keys)

        p_step = jax.pmap(
            partial(
                self.step,
                dynamic_scale=self.task.config.env.dynamic_scale,
                mutable_keys=mutable_keys,
                loss_fn=loss_fn,
                dynamic_scale_params=self.task.config.env.dynamic_scale_params,
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

    for module_key, module_params in config.items():
        if "pretrained" in module_params:
            assert isinstance(module_params.pretrained, str), "pretrained models are paths type str"
            path = module_params["pretrained"]
            print(str(Path(__file__).parents[2]) + path)
            replace = restore_checkpoint(
                str(Path(__file__).parents[2]) + path,
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
            params[module_key] = replace
    state.replace(params=freeze(params))
    return state
