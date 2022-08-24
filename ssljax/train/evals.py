def classification_eval_step(
    train_state, batch, loss_fn, model_fn, pipeline_fn, post_process_fn, config
):
    """

    Args:
        train_state (ssljax.train.TrainState):
        batch (jnp.ndarray):
        loss_fn:
        model_fn:
        pipeline_fn:
    """
    metrics = None
    other = None

    return metrics, other
