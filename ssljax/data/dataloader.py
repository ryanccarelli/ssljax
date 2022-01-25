import logging
from typing import Optional, Tuple

import cv2
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import omegaconf
import tensorflow.compat.v1 as tf
from scenic.dataset_lib.big_transfer import utils
from scenic.dataset_lib.big_transfer.registry import Registry as ScenicRegistry
from scenic.dataset_lib.datasets import get_dataset
from ssljax.core import register
from torch.utils import data
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

logger = logging.getLogger(__name__)


def numpy_collate(batch):
    """
    Collate function for pytorch datasets with NumPy arrays.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class FlattenAndCast:
    def __call__(self, pic):
        # return np.expand_dims(np.ravel(np.array(pic, dtype=jnp.float32)), axis=-1)
        return np.ravel(np.array(pic, dtype=jnp.float32))


class Cast:
    def __call__(self, pic):
        # return np.expand_dims(np.ravel(np.array(pic, dtype=jnp.float32)), axis=-1)
        return np.array(pic, dtype=jnp.float32)


class ScenicData:
    """
    Class wrapping distributed tfds dataloaders implemented
    according to conventions in https://github.com/google-research/scenic
    """

    pass


@register(ScenicData, "Base")
def scenic(
    config: omegaconf.DictConfig,
    data_rng: jax.random.PRNGKey,
    *,
    dataset_service_address: Optional[str] = None,
):
    device_count = jax.device_count()

    builder = get_dataset(config.dataset_name)
    batch_size = config.batch_size
    if batch_size % device_count > 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by the "
            f"number of devices ({device_count})"
        )
    if "eval_batch_size" in config:
        eval_batch_size = config.eval_batch_size
    else:
        eval_batch_size = config.batch_size
        if eval_batch_size % device_count > 0:
            raise ValueError(
                f"Eval batch size ({eval_batch_size}) must be divisible "
                f"by the number of devices ({device_count})"
            )
    local_batch_size = batch_size // jax.process_count()
    eval_local_batch_size = eval_batch_size // jax.process_count()
    device_batch_size = batch_size // device_count
    if "shuffle_seed" in config:
        shuffle_seed = config.shuffle_seed
    else:
        shuffle_seed = None
    if dataset_service_address and shuffle_seed is not None:
        raise ValueError(
            "Using dataset service with a random seed causes each "
            "worker to produce exactly the same data. Add "
            "config.shuffle_seed = None to your config if you want "
            "to run with dataset service."
        )
    dataset = builder(
        batch_size=local_batch_size,
        eval_batch_size=eval_local_batch_size,
        num_shards=jax.local_device_count(),
        dtype_str=config.data_dtype_str,
        rng=data_rng,
        shuffle_seed=shuffle_seed,
        dataset_configs=config.dataset_configs,
        dataset_service_address=dataset_service_address,
    )
    return dataset

@ScenicRegistry.register("preprocess_ops.random_resized_crops", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_resized_crops(
    height_small: float,
    width_small: float,
    height_large: float,
    width_large: float,
    n_small: int,
    n_large: int,
    prob: float = 1.0,
):
    def _random_resized_crops(image, p=1.0):
        """Randomly crop and resize an image.
        Args:
            image: `Tensor` representing an image of arbitrary size.
            height_small: Height of small crop.
            width_small: Width of small crop.
            height_large: Height of large crop.
            width_large: Width of large crop.
            n_small: Number of small crops.
            n_large: Number of large crops.
            p: Probability of applying this transformation.
        Returns:
            A preprocessed image `Tensor`.
        """
        def _transform(image):  # pylint: disable=missing-docstring
            images_small = [_crop_and_resize(image, height_small, width_small) for i in range(n_small)]
            images_large = [_crop_and_resize(image, height_large, width_large) for i in range(n_large)]
            images = images_small + images_large
            # combine these images together in a way that can be returned...
            # tf will not return a list
            # can't stack because images have different sizes
            return images
        return _random_apply(_transform, p=p, x=image)

    def _crop_and_resize(image, height, width):
        """Make a random crop and resize it to height `height` and width `width`.
        Args:
            image: Tensor representing the image.
            height: Desired image height.
            width: Desired image width.
        Returns:
            A `height` x `width` x channels Tensor holding a random crop of `image`.
        """
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        aspect_ratio = width / height
        image = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
            area_range=(0.08, 1.0),
            max_attempts=100,
            scope=None,
        )
        return tf.image.resize_bicubic([image], [int(height), int(width)])[0]

    def _random_apply(func, p, x):
        """Randomly apply function func to x with probability p."""
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x)

    return _random_resized_crops

def distorted_bounding_box_crop(
    image,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
    scope=None,
):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Borrowed from SIMCLR ground truth implementation.
    Args:
        image: `Tensor` of image data.
        bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
            where each coordinate is [0, 1) and the coordinates are arranged
            as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding
            box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
            scope: Optional `str` for name scope.
    Returns:
        (cropped image `Tensor`, distorted bbox `Tensor`).
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, target_height, target_width)

        return image
