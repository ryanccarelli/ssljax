from flax import traverse_util

def add_prefix_to_dict_keys(d, prefix="branches_"):
    return {prefix+str(k): v for k, v in d.items()}

def flattened_traversal(fn):
  def mask(data):
      flat = {'/'.join(k): v for k, v in traverse_util.flatten_dict(data).items()}
      x = traverse_util.unflatten_dict({tuple(k.split('/')): fn(k, v) for k, v in flat.items()})
      return  x
  return mask
