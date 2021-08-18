# utilities for schedulers
import jax.numpy as jnp


def target_ema(
    global_step: jnp.ndarray, base_ema: float, max_steps: int
) -> jnp.ndarray:
    decay = _cosine_decay(global_step, max_steps, 1.0)
    return 1.0 - (1.0 - base_ema) * decay
