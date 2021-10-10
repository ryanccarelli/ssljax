from ssljax.core.utils.register import register, print_registry
from ssljax.train.postprocess import PostProcess

import jax
import flax

def summ(*args):
    total = 0
    for val in args:
        total += val
    return total



@register(PostProcess, "ema")
def ema_builder(online_branch_name, target_branch_name, tau):

    def ema(params):

        def ema_update(online_branch, target_branch):
            online_branch_sum = jax.tree_map(summ, online_branch, target_branch)
            return jax.tree_map(lambda x, y : tau * x + (1 - tau) *y, online_branch, target_branch )

        updated_target_params = ema_update(params[online_branch_name],
                                           params[target_branch_name])

        params[target_branch_name] = updated_target_params
        return params

    return ema
