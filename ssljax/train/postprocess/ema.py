from ssljax.core.utils.register import register, print_registry
from ssljax.train.postprocess import PostProcess
from ssljax.core.utils.pytrees import ModelParamFilter
import jax
import flax
from flax import traverse_util
import pprint

def summ(*args):
    total = 0
    for val in args:
        total += val
    return total



@register(PostProcess, "ema")
def ema_builder(online_branch_name, target_branch_name, tau, remove_from_online):

    def remove_key_fn(path, value):
        for key in remove_from_online:
            if key in path:
                return False
        return True


    def ema(params):
        def ema_update(online_branch, target_branch):
            #target_branch_sum = jax.tree_map(summ, target_branch, online_branch)
            return jax.tree_map(lambda x, y: tau * x + (1 - tau) *y, online_branch, target_branch)


        model_params_filter = ModelParamFilter(remove_key_fn)
        online_filtered_params = model_params_filter.update(lambda x: x, params[online_branch_name])

        debug_flatten = lambda params : ['/'.join(k) for k, _ in traverse_util.flatten_dict(params).items()]
        updated_target_params = ema_update(online_filtered_params,
                                           params[target_branch_name])
        params[target_branch_name] = updated_target_params
        return params

    return ema
