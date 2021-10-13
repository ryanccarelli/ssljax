import flax
import jax
from ssljax.core.utils.register import print_registry, register
from ssljax.train.postprocess import PostProcess


def summ(*args):
    total = 0
    for val in args:
        total += val
    return total


@register(PostProcess, "ema")
def ema_builder(online_branch_name, target_branch_name, tau):
    """
    Constructs exponential moving average function. Target branch parameters are
    updated as a moving average of Online branch parameters.

    Args:
        online_branch_name (str): index of online branch in state.params
        target_branch_name (str): index of target branch in state.params
    """

    def ema(params):
        def ema_update(online_branch, target_branch):
            online_branch_sum = jax.tree_map(summ, online_branch, target_branch)
            return jax.tree_map(
                lambda x, y: tau * x + (1 - tau) * y, online_branch, target_branch
            )

        updated_target_params = ema_update(
            params[online_branch_name], params[target_branch_name]
        )
        params[target_branch_name] = updated_target_params
        return params

    return ema
