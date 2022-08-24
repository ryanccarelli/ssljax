import functools
import logging
from pathlib import Path

import flax
import flax.optim as optim
import jax
import jax.numpy as jnp
from clu import metric_writers, periodic_actions, platform
from scenic.common_lib import debug_utils
from scenic.train_lib import optimizers, train_utils
from ssljax.train import Task, Trainer
from ssljax.train.ssltrainstate_refactored import SSLTrainState
from ssljax.train.utils import bind_rng_to_host_device, register

logger = logging.getLogger(__name__)

@register(Trainer, "SSLTrainer")
def train(
    *,
    rng: jnp.ndarray,
    task: Task,
):
    """
    Main self-supervised training loop, executes a Task.

    Args:
        rng (jnp.array): An instance of jax.random.PRNGKey
        task (Task): A task that specifies the loop
    """
    lead_host = jax.process_index() == 0

    # setup cli workdir
    WORKDIR = Path(task.env.workdir) if "workdir" in task.env else Path("outs/")
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, WORKDIR, "Workdir"
    )

    batch_size = (
        (task.config.batch_size // jax.device_count())
        if "batch_size" in task.config.env
        else None
    )

    # initialize model
    input_spec = [
        (
            task.dataset.meta_data["input_shape"],
            task.dataset.meta_data.get("input_dtype", jnp.float32),
        )
    ]
    dummy_input = []
    for spec in input_spec:
        if spec is not None:
            in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
                spec, batch_size=batch_size
            )
            dummy_input.append(jnp.ones(in_st.shape, in_st.dtype))
        else:
            dummy_input.append(None)

    # create parameters in host RAM
    @functools.partial(jax.jit, backend="cpu")
    def _initialize_model(rngs):
        """Initialization function to be jitted."""
        init_model_state, init_params = task.model.init(
            rngs, *dummy_input, train=False, debug=False
        ).pop("params")
        # Set bias in the head to low value, such that loss is small initially.
        if task.config.get("init_head_bias", None) is not None:
            init_params = flax.core.unfreeze(init_params)
            init_params["output_projection"] = optimizers.tree_map_with_names(
                lambda p: jnp.full_like(p, task.config.init_head_bias),
                init_params["output_projection"],
                match_name_fn=lambda name: "bias" in name,
            )
            init_params = flax.core.freeze(init_params)
        return init_params, init_model_state

    init_rngs = rng
    if not isinstance(init_rngs, dict):
        init_rngs = {"params": rng}

    init_params, init_model_state = _initialize_model(init_rngs)

    # pop out params rng
    # TODO: why?
    init_rngs.pop("params")

    # Count number of trainable parameters:
    num_trainable_params = debug_utils.log_param_shapes(init_params)

    # Count gflops:
    if task.config.count_flops:
        variables = {"params": init_params, **init_model_state}
        flops = debug_utils.compute_flops(
            flax_model_apply_fn=functools.partial(
                task.model.apply, variables, train=False, debug=False, rngs=rngs
            ),
            input_spec=input_spec,
            fuse_multiply_add=task.config.fuse_multiply_add,
        )
        gflops = flops / (10 ** 9)
    else:
        gflops = None

    # TODO (Pika): init optimizer here, this time we should jit backend="cpu"
    optimizer = None
    rng, train_state_rng = jax.random.split(rng)

    train_state = SSLTrainState(
        params=init_params,
        mutable_states=TODO,
        opt_state=init_model_state,
        global_step=TODO,
        optimizer=optimizer,
        task=task,
        rng=train_state_rng,
    )
    start_step = train_state.global_step
    if task.config.checkpoint:
        train_state, start_step = train_utils.restore_checkpoint(WORKDIR, train_state)

    train_state = jax_utils.replicate(train_state)
    del init_params

    # training steps
    steps_per_epoch = (
        task.data.meta_data.get("num_train_examples", 0) // batch_size
    )
    total_steps = steps_per_epoch * task.config.env.epochs

    train_step_pmapped = jax.pmap(
        functools.partial(
            train_step,
            loss_fn=task.loss,
            model_fn=task.model,
            pipeline_fn=task.pipeline,
            post_process_fn=task.post_process,
            metrics_fn=task.
            config=task.config,
        ),
        axis_name="device",
        donate_argnums=(0, 1),
    )

    # TODO: dict of  eval steps to iterate?
    eval_step_pmapped = jax.pmap(
        functools.partial(
            eval_step,


    log_eval_steps = task.config.env.log_eval_steps or steps_per_epoch
    if not log_eval_steps:
        raise ValueError("'log_eval_steps' should be specified in task.config.env")
    checkpoint_steps = task.config.env.checkpoint_steps or log_eval_steps
    log_summary_steps = task.config.env.log_summary_steps or log_eval_steps

    total_eval_steps = int(
        np.ceil(task.data.meta_data["num_eval_examples"] / batch_size)
    )
    steps_per_eval = task.config.env.steps_per_eval or total_eval_steps

    train_metrics, extra_training_logs = [], []
    train_summary, eval_summary = None, None

    chrono = train_utils.Chrono(
        first_step=train_state.global_step,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        global_bs=batch_size,
        accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)),
    )

    report_progress = periodic_actions.ReportProgress(
        num_train_steps=total_steps, writer=task.writer
    )
    hooks = [report_progress]
    if task.config.env.xprof and lead_host:
        # TODO: refactor TBDIR -> WORKDIR and accept path in config
        hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=TBDIR))
    if start_step == 0:
        step0_log = {"num_trainable_params": num_trainable_params}
        if gflops:
            step0_log["gflops"] = gflops
        task.writer.write_scalars(1, step0_log)
    steps_per_epoch = (
        task.data.meta_data.get("num_train_examples", 0)
        // batch_size
    )

    for step in range(total_steps):
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            # inner loop here
            train_batch = next(task.data.train_iter)
            if task.config.pipelines.flatten:
                train_batch = jnp.ravel(train_batch)
                # batch = batch.reshape(*batch.shape[:-3], -1)
            # notice p_step refactor to return metrics
            train_state, loss, metrics = train_step_pmapped(
                train_state,
                train_batch,
            )
            train_metrics.append(metrics)
            # TODO: track learning rate
            extra_training_logs.append({"learning_rate": None})
        for h in hooks:
            h(step)
        chrono.pause()
        if (log_summary_steps == 1) or (step == total_steps):
            if lead_host:
                chrono.tick(step, writer=task.writer)
            train_summary = train_utils.log_train_summary(
                step=step,
                train_metrics=jax.tree_map(
                    train_utils.unreplicate_and_get, train_metrics
                ),
                writer=task.writer,
            )
            train_metrics, extra_training_logs = [], []
            if (step % task.config.log_eval_steps == 1) or (step == total_steps):
                with report_progress.timed("eval"):
                    eval_metrics = []
                    # populate and eval
                    train_state = train_utils.sync_model_state_across_replicas(
                        train_state
                    )
                for _ in range(steps_per_eval):
                    eval_batch = next(task.data.valid_iter)
                    if task.config.pipelines.flatten:
                        eval_batch = jnp.ravel(eval_batch)
                        # eval_batch = batch.reshape(*eval_batch.shape[:-3], -1)
                    e_metrics, _ = eval_step_pmapped(train_state, eval_batch)
                    eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
                eval_summary = train_utils.log_eval_summary(
                    step=step, eval_metrics=eval_metrics, writer=task.writer
                )
            task.writer.flush()
            del eval_metrics
        if (
            (step % checkpoint_steps == 0 and step > 0) or (step == total_steps)
        ) and task.config.checkpoint:
            with report_progress.timed("checkpoint"):
                train_state = train_utils.sync_model_state_across_replicas(train_state)
                if lead_host:
                    train_state.replace(accum_train_time=chrono.accum_train_time)
                    train_utils.save_checkpoint(WORKDIR, train_state)
        chrono.resume()

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return train_state, train_summary, eval_summary


def train_step(
    train_state: SSLTrainState,
    batch,
    loss_fn,
    model_fn,
    pipeline_fn,
    post_process_fn,
    metrics_fn,
    config,
):
    """
    Compute and apply gradient.

    Args:
        state (flax.training.train_state.TrainState): model state
        batch (jnp.ndarray): a single data batch
        loss_fn: loss
        model_fn: model
        post_process_fn: post_process
        metrics_fn: metrics
    """

    # TODO: why do we split here?
    new_rng, rng = jax.random.split(train_state.rng)

    pipeline_rng, rng = jax.random.split(rng, 2)
    pipeline_rng = bind_rng_to_host_device(pipeline_rng, "device", "device")

    pipeline_rng = jax.random.split(pipeline_rng, len(pipeline_fn))
    batch = list(
        map(
            lambda rng, pipeline: pipeline(batch, new_rng),
            pipeline_rng,
            pipeline_fn,
        )
    )
    # batch stores views indexed by the pipeline that produced them
    # TODO: why str(idx) instead of name passed in config?
    # instead zip with pipeline
    batch = {str(idx): val for idx, val in enumerate(batch)}

    def train_loss(params):
        """
        Apply loss function. Passed to opt.value_and_grad.
        """
        # loss must return a single float (since we grad loss)
        # but here we have also new_state
        # but we need new_state to manage mutable batch_params
        variables = {"params": params, **train_state.mutable_states}
        outs, new_state = model_fn.apply(
            variables, batch, mutable=train_state.mutable_states.keys()
        )
        loss = loss_fn(outs)
        return loss, new_state, outs

    if config.env.dynamic_scale:
        # optim.DynamicScale returns a DynamicScaleResult object
        grad_fn = optim.DynamicScale(**config.env.dynamic_scale_params).value_and_grad(
            train_loss, has_aux=True
        )
        dyn_scale, is_fin, loss_and_aux, grad = grad_fn(train_state.params, batch)
    else:
        grad_fn = jax.value_and_grad(train_loss, has_aux=True)
        loss_and_aux, grad = grad_fn(train_state.params, batch)

    (loss, aux) = loss_and_aux
    (new_state, outs) = aux

    loss, grad = (
        jax.lax.pmean(loss, axis_name="device"),
        jax.lax.pmean(grad, axis_name="device"),
    )

    state = train_state.apply_gradients(
        grads=grad["params"],
        **{"mutable_states": aux},
    )

    for idx, fun in enumerate(post_process_fn):
        state = train_state.replace(params=fun(state.params, state.step))

    # metrics_fn is 
    metrics = metrics_fn(outs, batch)

    return train_state, loss, metrics
