# functions for ssljax.augment.augmentation.ColorTransform
import jax
import jax.numpy as jnp

def rgb_to_hsv(r, g, b):
    """Converts R, G, B  values to H, S, V values.
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.
    Args:
      r: A tensor representing the red color component as floats.
      g: A tensor representing the green color component as floats.
      b: A tensor representing the blue color component as floats.
    Returns:
      H, S, V values, each as tensors of shape [...] (same as the input without
      the last dimension).
    """
    vv = jnp.maximum(jnp.maximum(r, g), b)
    range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
    sat = jnp.where(vv > 0, range_ / vv, 0.0)
    norm = jnp.where(range_ != 0, 1.0 / (6.0 * range_), 1e9)

    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0

    hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
    hue = hue * (range_ > 0)
    hue = hue + (hue < 0)
    return hue, sat, vv


def hsv_to_rgb(h, s, v):
    """Converts H, S, V values to an R, G, B tuple.
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.
    Args:
      h: A float tensor of arbitrary shape for the hue (0-1 values).
      s: A float tensor of the same shape for the saturation (0-1 values).
      v: A float tensor of the same shape for the value channel (0-1 values).
    Returns:
      An (r, g, b) tuple, each with the same dimension as the inputs.
    """
    c = s * v
    m = v - c
    dh = (h % 1.0) * 6.0
    fmodu = dh % 2.0
    x = c * (1 - jnp.abs(fmodu - 1))
    hcat = jnp.floor(dh).astype(jnp.float32)
    rr = (
        jnp.where(
            (hcat == 0) | (hcat == 5), c, jnp.where((hcat == 1) | (hcat == 4), x, 0)
        )
        + m
    )
    gg = (
        jnp.where(
            (hcat == 1) | (hcat == 2), c, jnp.where((hcat == 0) | (hcat == 3), x, 0)
        )
        + m
    )
    bb = (
        jnp.where(
            (hcat == 3) | (hcat == 4), c, jnp.where((hcat == 2) | (hcat == 5), x, 0)
        )
        + m
    )
    return rr, gg, bb


def adjust_brightness(rgb_tuple, delta):
    return jax.tree_map(lambda x: x + delta, rgb_tuple)


def adjust_contrast(image, factor):
    def _adjust_contrast_channel(channel):
        mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
        return factor * (channel - mean) + mean

    return jax.tree_map(_adjust_contrast_channel, image)


def adjust_saturation(h, s, v, factor):
    return h, jnp.clip(s * factor, 0.0, 1.0), v


def adjust_hue(h, s, v, delta):
    # Note: this method exactly matches TF"s adjust_hue (combined with the hsv/rgb
    # conversions) when running on GPU. When running on CPU, the results will be
    # different if all RGB values for a pixel are outside of the [0, 1] range.
    return (h + delta) % 1.0, s, v


def _random_brightness(rgb_tuple, rng, max_delta):
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return adjust_brightness(rgb_tuple, delta)


def _random_contrast(rgb_tuple, rng, max_delta):
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta
    )
    return adjust_contrast(rgb_tuple, factor)


def _random_saturation(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta
    )
    return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


def _random_hue(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return hsv_to_rgb(*adjust_hue(h, s, v, delta))


def _to_grayscale(image):
    rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
    grayscale = jnp.tensordot(image, rgb_weights, axes=(-1, -1))[..., jnp.newaxis]
    return jnp.tile(grayscale, (1, 1, 3))  # Back to 3 channels.
