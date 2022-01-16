import logging
from typing import Optional, Tuple

import cv2
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import omegaconf
import tensorflow as tf
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


@ScenicRegistry.register("identity", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_identity():
    """Passthrough function for scenic bit preprocessing"""

    def _identity(image):
        return image

    return _identity


@ScenicRegistry.register("identity", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_resized_crops(
    height: float,
    width: float,
    prob: float = 1.0,
    scale: Tuple[float] = (0.08, 1.0),
    ratio: Tuple[float] = (0.75, 1.3333333333333333),
    interpolation: str = "bilinear",
):
    def _random_resized_crops(image):
        coords = _get_coords(image)
        # TODO: random crop?
        boxes = [
            coords["h_start"],
            coords["w_start"],
            coords["h_start"] + coords["crop_height"],
            coords["w_start"] + coords["crop_width"],
        ]
        # TODO: multiple times? return tf.stack([copies])
        image = tf.image.crop_and_resize(
            image, boxes=boxes, crop_size=(height, width), method=interpolation
        )
        return image

    def _get_coords(image):
        area = image.shape[0] * image.shape[1]

        for _attempt in range(10):
            target_area = tf.random.uniform(minval=scale[0], maxval=scale[1]) * area
            log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[0]))
            # TODO: shape?
            aspect_ratio = tf.math.exp(
                tf.random.uniform(minval=log_ratio[0], maxval=log_ratio[1])
            )
            w = int(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)).item())
            h = int(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)).item())

            if 0 < w <= x.shape[1] and 0 < h <= x.shape[0]:
                i = tf.random.uniform(
                    minval=0, maxval=(image.shape[0] - h), dtype=tf.dtypes.int64
                )
                j = tf.random.uniform(
                    minval=0, maxval=(image.shape[1] - w), dtype=tf.dtypes.int64
                )
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (image.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (image.shape[1] - w + 1e-10),
                }

        # fallback to central crop
        in_ratio = image.shape[1] / image.shape[0]
        if in_ratio < min(ratio):
            w = image.shape[1]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = image.shape[0]
            w = int(round(h * max(ratio)))
        else:
            w = image.shape[1]
            h = image.shape[0]
        i = (image.shape[0] - h) // 2
        j = (image.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (image.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (image.shape[1] - w + 1e-10),
        }

    return _random_resized_crops
