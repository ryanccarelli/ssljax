import flax
from flax import traverse_util
from flax.traverse_util import (Traversal, _EmptyNode, _get_params_dict,
                                _sorted_items, empty_node, flatten_dict,
                                unflatten_dict)
import fnmatch
from typing import  AnyStr, List
def add_prefix_to_dict_keys(d, prefix="branches_"):
    return {prefix + str(k): v for k, v in d.items()}


def flattened_traversal(fn):
    """
    Traverse flattened pytree.
    """

    def mask(data):
        flat = {"/".join(k): v for k, v in traverse_util.flatten_dict(data).items()}
        x = traverse_util.unflatten_dict(
            {tuple(k.split("/")): fn(k, v) for k, v in flat.items()}
        )
        return x

    return mask


class ModelParamFilter(Traversal):
    """Select model parameters using a name filter.

    This traversal operates on a nested dictionary of parameters and selects a
    subset based on the `filter_fn` argument.

    """

    def __init__(self, filter_fn):
        """
        Construct a ModelParamTraversal.

        Args:
          filter_fn: a function that takes a parameter's full name and its value and
            returns whether this parameter should be selected or not. The name of a
            parameter is determined by the module hierarchy and the parameter name
            (for example: '/module/sub_module/parameter_name').
        """
        self._filter_fn = filter_fn

    def iterate(self, inputs):
        params = _get_params_dict(inputs)
        flat_dict = flatten_dict(params)
        for key, value in _sorted_items(flat_dict):
            path = "/" + "/".join(key)
            if self._filter_fn(path, value):
                yield value

    def update(self, fn, inputs):
        params = _get_params_dict(inputs)
        flat_dict = flatten_dict(params, keep_empty_nodes=True)
        new_dict = {}
        for key, value in _sorted_items(flat_dict):
            # empty_node is not an actual leave. It's just a stub for empty nodes
            # in the nested dict.
            if value is not empty_node:
                path = "/" + "/".join(key)
                if self._filter_fn(path, value):
                    value = fn(value)
                    new_dict[key] = value
            new_params = unflatten_dict(new_dict)
        if isinstance(inputs, flax.nn.base.Model):
            return inputs.replace(params=new_params)
        elif isinstance(inputs, flax.core.FrozenDict):
            return flax.core.FrozenDict(new_params)
        else:
            return new_params



def inc_filter_path(path: AnyStr, patterns: List[AnyStr]) -> bool:
    """
    Returns True if the given path string matches any in the pattern
    Args:
      path (Str): String.  This is assumed to be a path (e.g., conv_1/bias)
      patterns (List[Str]): List of patterns against which the path is compared


    Example::

      path = "a/b/c"
      pats = ["*/c"]
      inc_filter_path(path, pats)
      >> True
    """
    for pat in patterns:
        if len(fnmatch.filter([path], pat)) != 0:
               return True
    else:
        return False

def exc_filter_path(path: AnyStr, patterns: List[AnyStr]) -> bool:
    """
    Returns False if the given path string matches any in the pattern
    Args:
      path (Str): This is assumed to be a path (e.g., conv_1/bias)
      patterns (List[Str]): List of patterns against which the path is compared


    Example::

      path = "a/b/c"
      pats = ["*/c"]
      inc_filter_path(path, pats)
      >> True
    """
    return not inc_filter_path(path, patterns)

def inc_exc_filter(path, inc_filters, exc_filters):
    """
    If the path is covered by the inc_filters, it then checks if it is in the
    exclude list. If it is covered by the exc_filters, then the function
    returns False. Else, it returns a True. If it is not present in
    inc_filters, it returns a False

    Args:
      path (Str): This is assumed to be a path like conv_1/bias.
      inc_filters (List[Str]): List of inclusionary patterns
      exc_filters (List[Str]): List of exclusionary patterns
    """
    if inc_filter_path(path, inc_filters):
        if exc_filters is not None:
            return exc_filter_path(path, exc_filters)
        else:
            return True
    else:
        return False


def flattened_traversal_with_filter(fn):
  def mask(data):
    flat = traverse_util.flatten_dict(data)
    ret_dict = {}
    for k, v in flat.items():
        if fn("/".join(k)):
            ret_dict[k] =  True
        else:
            ret_dict[k] =  False
    return traverse_util.unflatten_dict(ret_dict)
  return mask
