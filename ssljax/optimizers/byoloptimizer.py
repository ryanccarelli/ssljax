import optax
from ssljax.optimizers import lars


def byoloptimizer(learning_rate, tau_learning_rate):
    """
    Optimizer that applies LARS to online network and
    updates target network by exponential moving average.

    Assumes that parameters are partitioned into
    "online" and "target" groups.
    Args:
        learning_rate:
        tau_learning_rate:
            TODO: tau must be managed by its own cosine scheduler

    """
    # parameters are partitioned in SSLModel through the Branch class
    param_labels = ("target", "online")
    return optax.multi_transform({"target": ema, "online": lars}, param_labels)

def byol_ema(tau) -> base.GradientTransformation:
    """
    Exponential moving average with cosine decay as implemented
    in Bootstrap Your Own Latent.
    """
    def init_fn(params):
        pass

    def update_fn(updates, state, params=None):
        # from state.params (flax.core.frozen_dict.FrozenDict) we get a dict
        # and this dict needs to distinguish the target and online params
        target_params = state.params[]
        online_params = state.params[]
        # optax.incremental_update
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)

def _cosine_decay(global_step: jnp.ndarray, max_steps: int, initial_value: float) -> jnp.ndarray:
    """
    Simple implementation of cosine decay from TF1.
    This is used in BYOL optimizer to manage tau parameter.
    """
    global_step = jnp.minimum(global_step, max_steps)
    cosine_decay_value = 0.5 * (1 + jnp.cos(jnp.pi * global_step / max_steps))
    decayed_learning_rate = initial_value * cosine_decay_value
    return decayed_learning_rate
