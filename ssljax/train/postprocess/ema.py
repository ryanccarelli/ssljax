from typing import Union

import flax
import jax
import optax
from flax import traverse_util
from optax._src import base
from optax._src.alias import _scale_by_learning_rate
from ssljax.core.utils.pytrees import ModelParamFilter
from ssljax.core.utils.register import (get_from_register, print_registry,
                                        register)
from ssljax.train.postprocess import PostProcess
from ssljax.train.scheduler import Scheduler

ScalarOrSchedule = Union[float, base.Schedule]


@register(PostProcess, "ema")
def ema_builder(online_branch_name, target_branch_name, tau, remove_from_online):
    """
    Constructs exponential moving average function. Target branch parameters are
    updated as a moving average of Online branch parameters.

    Args:
        online_branch_name (str): index of online branch in state.params
        target_branch_name (str): index of target branch in state.params
        remove_from_online (list): modules in online branch that will not be updated
        tau (ScalarOrSchedule): ema decay rate schedule
    """

    def remove_key_fn(path, value):
        for key in remove_from_online:
            if key in path:
                return False
        return True

    def ema(params, global_step, tau=tau):
        tau = tau(global_step)

        def ema_update(online_branch, target_branch):
            # target_branch_sum = jax.tree_map(summ, target_branch, online_branch)
            return jax.tree_map(
                lambda x, y: tau * x + (1 - tau) * y,
                online_branch,
                target_branch,
            )

        model_params_filter = ModelParamFilter(remove_key_fn)
        online_filtered_params = model_params_filter.update(
            lambda x: x, params[online_branch_name]
        )
        updated_target_params = ema_update(
            online_filtered_params, params[target_branch_name]
        )
        params[target_branch_name] = updated_target_params
        return params

    return ema


def summ(*args):
    total = 0
    for val in args:
        total += val
    return total
