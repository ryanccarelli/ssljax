import logging

from ssljax.train.ssltrainstate_refactored import SSLTrainState

logger = logging.getLogger(__name__)

from pathlib import Path

import flax.optim as optim
import jax
from tensorboardX import GlobalSummaryWriter
from ssljax.train.utils import bind_rng_to_host_device

CHECKPOINTSDIR = Path("outs/checkpoints/")
CHECKPOINTSDIR.mkdir(parents=True, exist_ok=True)
TBDIR = Path("outs/tensorboard/")
TBDIR.mkdir(parents=True, exist_ok=True)

writer = GlobalSummaryWriter(TBDIR)


def train_step(
        train_state: SSLTrainState,
        batch,
        loss_fn,
        model_fn,
        config
):
    """
    Compute and apply gradient.

    Args:
        state (flax.training.train_state.TrainState): model state
        batch (jnp.ndarray): a single data batch
        rng (jnp.ndarray): PRNG key
        mutable_keys (List[str]): parameters that are mutable
        loss_fn: loss
        model_fn: model
    """

    new_rng, rng = jax.random.split(train_state.rng)

    pipeline_rng, rng = jax.random.split(rng, 2)
    pipeline_rng = bind_rng_to_host_device(pipeline_rng, "device", "device")

    pipeline_rng = jax.random.split(pipeline_rng, len(train_state.task.pipeline))
    batch = list(
        map(
            lambda rng, pipeline: pipeline(batch, rng),
            pipeline_rng,
            train_state.task.pipeline,
        )
    )
    # batch stores views indexed by the pipeline that produced them
    batch = {str(idx): val for idx, val in enumerate(batch)}

    def train_loss(params):
        """
        Apply loss function. Passed to opt.value_and_grad.
        """
        # loss must return a single float (since we grad loss)
        # but here we have also new_state
        # but we need new_state to manage mutable batch_params
        variables = {'params': params, **train_state.mutable_states}
        outs, new_state = model_fn.apply(variables, batch, mutable=train_state.mutable_states.keys())
        loss = loss_fn(outs)
        return loss, new_state

    if train_state.task.config.env.dynamic_scale:
        # optim.DynamicScale returns a DynamicScaleResult object
        grad_fn = optim.DynamicScale(**train_state.task.config.env.dynamic_scale_params).value_and_grad(
                train_loss, has_aux=True
            )
        dyn_scale, is_fin, loss_and_aux, grad = grad_fn(train_state.params, batch)
    else:
        grad_fn = jax.value_and_grad(train_loss, has_aux=True)
        loss_and_aux, grad = grad_fn(train_state.params, batch)

    (loss, aux) = loss_and_aux

    loss, grad = (
        jax.lax.pmean(loss, axis_name="device"),
        jax.lax.pmean(grad, axis_name="device"),
    )

    state = train_state.apply_gradients(
        grads=grad["params"], **{"mutable_states": aux},
    )

    for idx, fun in enumerate(train_state.task.post_process):
        state = train_state.replace(params=fun(state.params, state.step))

    return train_state, loss
