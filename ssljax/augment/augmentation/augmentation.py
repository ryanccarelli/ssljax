import functools

import jax
import jax.numpy as jnp
from ssljax.augment.augmentation.colortransform import (_random_brightness,
                                           _random_contrast, _random_hue,
                                           _random_saturation, _to_grayscale,
                                           adjust_brightness, adjust_contrast,
                                           adjust_hue, adjust_saturation,
                                           hsv_to_rgb, rgb_to_hsv)
from ssljax.core.utils import register


class Augmentation:
    """
    An augmentation is a function applied to images.
    """

    def __init__(self, prob=1.0):
        self.prob = prob

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, x, rng):
        raise NotImplementedError


class AugmentationDistribution:
    """
    A distribution of augmentations to be sampled
    """

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def sample(self, rng):
        key, subkey = jax.random.split(rng)
        sampledIndex = jax.random.choice(subkey, a=len(self.augmentations), p=jnp.array([aug.prob for aug in self.augmentations]))
        return self.augmentations[sampledIndex]


# byol augmentations
@register(Augmentation, "randomflip")
class RandomFlip(Augmentation):
    """
    Randomly flip image.
    Modified from https://github.com/deepmind/deepmind-research/blob/master/byol/utils/augmentations.py

    Args:
        x (jnp.array): an NHWC tensor (with C=3).
        rng (jnp.array): a single PRNGKey.
    """

    def __call__(self, x, rng):
        return jax.vmap(self._random_flip_single_image)(x, rng)

    @staticmethod
    def _random_flip_single_image(image, rng):
        _, flip_rng = jax.random.split(rng)
        should_flip_lr = jax.random.uniform(flip_rng, shape=()) <= self.prob
        image = jax.lax.cond(
            should_flip_lr,
            image,
            jnp.fliplr,
            image,
            lambda x: x,
        )
        return image


@register(Augmentation, "randomgaussianblur")
class RandomGaussianBlur(Augmentation):
    """
    Randomly apply gaussian blur.
    Modified from https://github.com/deepmind/deepmind-research/blob/master/byol/utils/augmentations.py

    Args:
        x (jnp.array): an NHWC tensor (with C=3).
    """

    def __init__(
            self,
            prob,
            kernel_size,
            padding,
            sigma_min,
            sigma_max,
    ):
        super().__init__(prob)
        self.kernel_size = kernel_size
        self.padding = padding
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x, rng):
        """Applies a random gaussian blur."""
        apply_rng, transform_rng = jax.random.split(rng)

        def _apply(x):
            (sigma_rng,) = jax.random.split(transform_rng, 1)
            sigma = jax.random.uniform(
                sigma_rng,
                shape=(),
                minval=self.sigma_min,
                maxval=self.sigma_max,
                dtype=jnp.float32,
            )
            return self._gaussian_blur_single_image(
                x, self.kernel_size, self.padding, sigma
            )

        return self._maybe_apply(_apply, x, apply_rng, self.prob)

    def _maybe_apply(self, apply_fn, inputs, rng, apply_prob):
        should_apply = jax.random.uniform(rng, shape=()) <= apply_prob
        return jax.lax.cond(should_apply, inputs, apply_fn, inputs, lambda x: x)

    def _depthwise_conv2d(self, inputs, kernel, strides, padding):
        """Computes a depthwise conv2d in Jax.
        Args:
          inputs: an NHWC tensor with N=1.
          kernel: a [H", W", 1, C] tensor.
          strides: a 2d tensor.
          padding: "SAME" or "VALID".
        Returns:
          The depthwise convolution of inputs with kernel, as [H, W, C].
        """
        return jax.lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding,
            feature_group_count=inputs.shape[-1],
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    def _gaussian_blur_single_image(self, image, kernel_size, padding, sigma):
        """Applies gaussian blur to a single image, given as NHWC with N=1."""
        radius = int(kernel_size / 2)
        kernel_size_ = 2 * radius + 1
        x = jnp.arange(-radius, radius + 1).astype(jnp.float32)
        blur_filter = jnp.exp(-(x ** 2) / (2.0 * sigma ** 2))
        blur_filter = blur_filter / jnp.sum(blur_filter)
        blur_v = jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1])
        blur_h = jnp.reshape(blur_filter, [1, kernel_size_, 1, 1])
        num_channels = image.shape[-1]
        blur_h = jnp.tile(blur_h, [1, 1, 1, num_channels])
        blur_v = jnp.tile(blur_v, [1, 1, 1, num_channels])
        expand_batch_dim = len(image.shape) == 3
        if expand_batch_dim:
            image = image[jnp.newaxis, ...]
        blurred = self._depthwise_conv2d(image, blur_h, strides=[1, 1], padding=padding)
        blurred = self._depthwise_conv2d(
            blurred, blur_v, strides=[1, 1], padding=padding
        )
        blurred = jnp.squeeze(blurred, axis=0)
        return blurred


@register(Augmentation, "colortransform")
class ColorTransform(Augmentation):
    """Applies color jittering and/or grayscaling to a batch of images.
    Args:
      images: an NHWC tensor, with C=3.
      rng: a single PRNGKey.
      brightness: the range of jitter on brightness.
      contrast: the range of jitter on contrast.
      saturation: the range of jitter on saturation.
      hue: the range of jitter on hue.
      color_jitter_prob: the probability of applying color jittering.
      to_grayscale_prob: the probability of converting the image to grayscale.
      apply_prob: the probability of applying the transform to a batch element.
      shuffle: whether to apply the transforms in a random order.
    Returns:
      A NHWC tensor of the transformed images.
    """

    def __init__(
            self,
            prob=1.0,
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
            color_jitter_prob=0.8,
            to_grayscale_prob=0.2,
            apply_prob=1.0,
            shuffle=True,
    ):
        print("prob", prob)
        super().__init__(prob)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter_prob = color_jitter_prob
        self.to_grayscale_prob = to_grayscale_prob
        self.apply_prob = apply_prob
        self.shuffle = shuffle

    def __call__(
            self,
            x,
            rng,
    ):
        rngs = jax.random.split(rng, x.shape[0])
        jitter_fn = functools.partial(
            self._color_transform_single_image,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
            color_jitter_prob=self.color_jitter_prob,
            to_grayscale_prob=self.to_grayscale_prob,
            apply_prob=self.apply_prob,
            shuffle=self.shuffle,
        )
        return jax.vmap(jitter_fn)(x, rngs)

    def _color_transform_single_image(
            self,
            image,
            rng,
            brightness,
            contrast,
            saturation,
            hue,
            to_grayscale_prob,
            color_jitter_prob,
            apply_prob,
            shuffle,
    ):
        """Applies color jittering to a single image."""
        apply_rng, transform_rng = jax.random.split(rng)
        perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(
            transform_rng, 7
        )

        # Whether the transform should be applied at all.
        should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
        # Whether to apply grayscale transform.
        should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
        # Whether to apply color jittering.
        should_apply_color = jax.random.uniform(cj_rng, shape=()) <= color_jitter_prob

        # Decorator to conditionally apply fn based on an index.
        def _make_cond(fn, idx):
            def identity_fn(x, unused_rng, unused_param):
                return x

            def cond_fn(args, i):
                def clip(args):
                    return jax.tree_map(lambda arg: jnp.clip(arg, 0.0, 1.0), args)

                out = jax.lax.cond(
                    should_apply & should_apply_color & (i == idx),
                    args,
                    lambda a: clip(fn(*a)),
                    args,
                    lambda a: identity_fn(*a),
                )
                return jax.lax.stop_gradient(out)

            return cond_fn

        random_brightness_cond = _make_cond(_random_brightness, idx=0)
        random_contrast_cond = _make_cond(_random_contrast, idx=1)
        random_saturation_cond = _make_cond(_random_saturation, idx=2)
        random_hue_cond = _make_cond(_random_hue, idx=3)

        def _color_jitter(x):
            rgb_tuple = tuple(jax.tree_map(jnp.squeeze, jnp.split(x, 3, axis=-1)))
            if shuffle:
                order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
            else:
                order = range(4)
            for idx in order:
                if brightness > 0:
                    rgb_tuple = random_brightness_cond(
                        (rgb_tuple, b_rng, brightness), idx
                    )
                if contrast > 0:
                    rgb_tuple = random_contrast_cond((rgb_tuple, c_rng, contrast), idx)
                if saturation > 0:
                    rgb_tuple = random_saturation_cond(
                        (rgb_tuple, s_rng, saturation), idx
                    )
                if hue > 0:
                    rgb_tuple = random_hue_cond((rgb_tuple, h_rng, hue), idx)
            return jnp.stack(rgb_tuple, axis=-1)

        out_apply = _color_jitter(image)
        out_apply = jax.lax.cond(
            should_apply & should_apply_gs,
            out_apply,
            _to_grayscale,
            out_apply,
            lambda x: x,
        )
        return jnp.clip(out_apply, 0.0, 1.0)


@register(Augmentation, "solarize")
class Solarize(Augmentation):
    """Applies solarization.
    Args:
        x: an NHWC tensor (with C=3).
        rng: a single PRNGKey.
        threshold: the solarization threshold.
        apply_prob: the probability of applying the transform to a batch element.
    Returns:
        A NHWC tensor of the transformed images.
    """

    def __init__(self, prob=1.0, threshold=0.5):
        super().__init__(prob)
        self.threshold = threshold

    def __call__(self, x, rng):
        rngs = jax.random.split(rng, x.shape[0])
        solarize_fn = functools.partial(
            self._solarize_single_image, threshold=self.threshold, apply_prob=self.prob
        )
        return jax.vmap(solarize_fn)(x, rngs)

    def _solarize_single_image(self, image, rng, threshold, apply_prob):
        def _apply(image):
            return jnp.where(image < threshold, image, 1.0 - image)

        return self._maybe_apply(_apply, image, rng, apply_prob)

    def _maybe_apply(self, apply_fn, inputs, rng, apply_prob):
        should_apply = jax.random.uniform(rng, shape=()) <= apply_prob
        return jax.lax.cond(should_apply, inputs, apply_fn, inputs, lambda x: x)


@register(Augmentation, "clip")
class Clip(Augmentation):
    """
    Wrap jnp.clip.

    Args:
        x_min(float): Minimum value.
        x_max(float): Maximum value.
    """
    def __init__(self, x_min=0., x_max=1.):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x):
        x = jnp.clip(x, x_min, x_max)
        return x


if __name__ == "__main__":
    augs = AugmentationDistribution([])
    print(augs)
