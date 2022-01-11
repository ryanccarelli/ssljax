from typing import Union

import flax
import jax
import optax
from flax import traverse_util
from optax._src import base
from optax._src.alias import _scale_by_learning_rate
from ssljax.core import ModelParamFilter, get_from_register, register
from ssljax.train.postprocess import PostProcess
from ssljax.train.scheduler import Scheduler

ScalarOrSchedule = Union[float, base.Schedule]


@register(PostProcess, "ema")
def ema_builder(online_module_names, target_module_names, tau):
    """
    Constructs exponential moving average function. Target module parameters are
    updated as a moving average of Online module parameters.

    Args:
        online_module_names (list[str]): indexes of online modules in state.params
        target_branch_names (list[str]): indexes of target modules in state.params
        tau (ScalarOrSchedule): ema decay rate schedule
    """

    def ema(params, global_step, tau=tau):
        tau = tau(global_step)

        def ema_update(online_module, target_module):
            # target_branch_sum = jax.tree_map(summ, target_branch, online_branch)
            return jax.tree_map(
                lambda x, y: tau * x + (1 - tau) * y,
                online_module,
                target_module,
            )

        for online_module_name, target_module_name in zip(online_module_names, target_module_names):
            updated_target_params = ema_update(
                params[online_module_name], params[target_module_name]
            )
            params[target_module_name] = updated_target_params

        return params

    return ema


def summ(*args):
    total = 0
    for val in args:
        total += val
    return total
