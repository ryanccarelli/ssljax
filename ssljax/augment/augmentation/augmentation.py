import functools

import jax
import jax.numpy as jnp
from ssljax.augment.augmentation.colortransform import (_random_brightness,
                                                        _random_contrast,
                                                        _random_hue,
                                                        _random_saturation,
                                                        _to_grayscale,
                                                        adjust_brightness,
                                                        adjust_contrast,
                                                        adjust_hue,
                                                        adjust_saturation,
                                                        hsv_to_rgb, rgb_to_hsv)
from ssljax.core import register


class Augmentation:
    """
    An augmentation applies a function to data.

    Args:
        prob (float): Probability that augmentation will be executed.
    """

    def __init__(self, prob=1.0):
        assert isinstance(prob, float), "prob must be of type float"
        self.prob = prob

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, x, rng):
        """
        Apply function to data.

        Args:
            x: input data
            rng (jnp.array): Jax PRNG.
        """
        raise NotImplementedError

# byol augmentations
@register(Augmentation, "RandomFlip")
class RandomFlip(Augmentation):
    """
    Randomly flip image.
    Modified from https://github.com/deepmind/deepmind-research/blob/master/byol/utils/augmentations.py

    Args:
        x (jnp.array): an NHWC tensor (with C=3).
        rng (jnp.array): a single PRNGKey.
    """

    def __init__(
        self,
        prob=1.0,
    ):
        super().__init__(prob)

    def __call__(self, x, rng):
        rngs = jax.random.split(rng, x.shape[0])
        return jax.vmap(self._random_flip_single_image, in_axes=0)(x, rngs)

    def __repr__(self):
        return "RandomFlip"

    def _random_flip_single_image(self, image, rng):
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


@register(Augmentation, "GaussianBlur")
class GaussianBlur(Augmentation):
    """
    Applies a gaussian blur to a batch of images
    Modified from https://github.com/deepmind/deepmind-research/blob/master/byol/utils/augmentations.py

    Args:
        x (jnp.array): an NHWC tensor (with C=3).
    """

    def __init__(
        self,
        prob,
        blur_divider=10.0,
        sigma_min=0.1,
        sigma_max=2.0,
        padding="SAME",
    ):
        super().__init__(prob)
        self.prob = prob
        self.blur_divider = blur_divider
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.padding = padding

    def __call__(self, x, rng):
        rngs = jax.random.split(rng, x.shape[0])
        self.kernel_size = x.shape[1] / self.blur_divider
        blur_fn = functools.partial(
            self._random_gaussian_blur,
            kernel_size=self.kernel_size,
            padding=self.padding,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            apply_prob=self.prob,
        )
        return jax.vmap(blur_fn)(x, rngs)

    def _random_gaussian_blur(
        self, x, rng, kernel_size, padding, sigma_min, sigma_max, apply_prob
    ):
        """Applies a random gaussian blur."""
        apply_rng, transform_rng = jax.random.split(rng)

        def _apply(x):
            (sigma_rng,) = jax.random.split(transform_rng, 1)
            sigma = jax.random.uniform(
                sigma_rng,
                shape=(),
                minval=sigma_min,
                maxval=sigma_max,
                dtype=jnp.float32,
            )
            return self._gaussian_blur_single_image(x, kernel_size, padding, sigma)

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

    def __repr__(self):
        return "GaussianBlur"


@register(Augmentation, "ColorTransform")
class ColorTransform(Augmentation):
    """
    Applies color jittering and/or grayscaling to a batch of images.

    Args:
        images (jnp.array): an NHWC tensor, with C=3.
        rng (jnp.array): Jax PRNGKey
        brightness (float): the range of jitter on brightness.
        contrast (float): the range of jitter on contrast.
        saturation (float): the range of jitter on saturation.
        hue (float): the range of jitter on hue.
        color_jitter_prob (float): the probability of applying color jittering.
        to_grayscale_prob (float): the probability of converting the image to grayscale.
        apply_prob (float): the probability of applying the transform to a batch element.
        shuffle (bool): whether to apply the transforms in a random order.

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
                order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.float32))
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

    def __repr__(self):
        return "ColorTransform"


@register(Augmentation, "Solarize")
class Solarize(Augmentation):
    """
    Applies solarization.

    Args:
        x (jnp.array): an NHWC tensor (with C=3).
        rng (jnp.array): Jax PRNG
        threshold (float): the solarization threshold.
        apply_prob (float): the probability of applying the transform to a batch element.
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

    def __repr__(self):
        return "Solarize"


@register(Augmentation, "Clip")
class Clip(Augmentation):
    """
    Wrap jnp.clip.

    Args:
        x_min (float): Minimum value.
        x_max (float): Maximum value.
    """

    def __init__(self, prob=1.0, x_min=0.0, x_max=1.0):
        super().__init__(prob)
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x, rng):
        x = jnp.clip(x, self.x_min, self.x_max)
        return x

    def __repr__(self):
        return "Clip"


@register(Augmentation, "Identity")
class Identity(Augmentation):
    """
    Apply identity map.

    Args:
        prob (float): probability of execution.
    """

    def __init__(self, prob=1.0):
        super().__init__(prob)

    def __call__(self, x, rng):
        return x

    def __repr__(self):
        return "Identity"

@register(Augmentation, "RandomCrop")
class RandomCrop(Augmentation):
    """
    Apply a random crop.
    # TODO: THIS IS NOT STOCHASTIC DO NOT USE

    Args:
        prob (float): probability of execution.
    """

    def __init__(
        self,
        prob=1.0,
        height=224,
        width=224,
    ):
        super().__init__(prob)
        self.height = height
        self.width = width

    def __call__(self, x, rng):
        rngs = jax.random.split(rng, x.shape[0])
        random_crop_fn = functools.partial(
            self._random_crop, crop_height=self.height, crop_width=self.width
        )
        return jax.vmap(random_crop_fn)(x, rngs)

    def _random_crop(
        self,
        img: jnp.ndarray,
        rngs: jnp.array,
        crop_height: int,
        crop_width: int,
        h_start: float = 0,
        w_start: float = 0,
    ):
        height, width = img.shape[:2]
        if height < crop_height or width < crop_width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=crop_height,
                    crop_width=crop_width,
                    height=height,
                    width=width,
                )
            )
        x1, y1, x2, y2 = self._get_random_crop_coords(
            height, width, crop_height, crop_width, h_start, w_start
        )
        img = img[y1:y2, x1:x2]
        return img

    def _get_random_crop_coords(
        self,
        height: int,
        width: int,
        crop_height: int,
        crop_width: int,
        h_start: float,
        w_start: float,
    ):
        y1 = int((height - crop_height) * h_start)
        y2 = y1 + crop_height
        x1 = int((width - crop_width) * w_start)
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    def __repr__(self):
        return "RandomCrop"

@register(Augmentation, "CenterCrop")
class CenterCrop(Augmentation):
    """
    Apply a center crop.

    Args:
        prob (float): probability of execution.
    """

    def __init__(
        self,
        prob=1.0,
        height=224,
        width=224,
    ):
        super().__init__(prob)
        self.height = height
        self.width = width

    def __call__(self, x, rng):
        rngs = jax.random.split(rng, x.shape[0])
        center_crop_fn = functools.partial(
            self._center_crop, crop_height=self.height, crop_width=self.width
        )
        return jax.vmap(center_crop_fn)(img=x)

    def _get_center_crop_coords(
        self, height: int, width: int, crop_height: int, crop_width: int
    ):
        y1 = (height - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (width - crop_width) // 2
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    def _center_crop(self, img: jnp.ndarray, crop_height: int, crop_width: int):
        height, width = img.shape[:2]
        if height < crop_height or width < crop_width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=crop_height,
                    crop_width=crop_width,
                    height=height,
                    width=width,
                )
            )
        x1, y1, x2, y2 = self._get_center_crop_coords(
            height, width, crop_height, crop_width
        )
        img = img[y1:y2, x1:x2]
        return img

    def __repr__(self):
        return "CenterCrop"
