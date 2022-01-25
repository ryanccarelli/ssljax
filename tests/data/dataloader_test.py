import jax.random
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from ssljax.data.dataloader import ScenicData, scenic


@pytest.mark.parametrize("batch_size", [16, 32])
def test_dataloader(batch_size):
    rng = jax.random.PRNGKey(0)
    loader = scenic(
        OmegaConf.create(
            {
                "dataset_name": "mnist",
                "batch_size": batch_size,
                "data_dtype_str": "float32",
                "dataset_configs": {},
            }
        ),
        data_rng=rng,
    )
    data = next(iter(loader.train_iter))
    assert isinstance(data["inputs"], np.ndarray)


@pytest.mark.parametrize("batch_size", [16, 32])
def test_bit(batch_size):
    rng = jax.random.PRNGKey(0)
    # test passthrough
    loader = scenic(
        OmegaConf.create(
            {
                "experiment_name": "test",
                "dataset_name": "bit",
                "data_dtype_str": "float32",
                "batch_size": batch_size,
                "dataset_configs": {
                    "pp_train": (
                        "decode_jpeg_and_inception_crop(224)"
                        "|flip_lr"
                        "|randaug(2,15)"
                        "|value_range(-1, 1)"
                        "|onehot(1000, key='label', key_result='labels')"
                        "|keep('image', 'labels')"
                    ),
                    "pp_eval": (
                        "decode"
                        "|resize_small(256)"
                        "|central_crop(224)"
                        "|value_range(-1, 1)"
                        "|onehot(1000, key='label', key_result='labels')"
                        "|keep('image', 'labels')"
                    ),
                    "dataset": "imagenet2012",
                    "num_classes": 1000,
                    "train_split": "train",
                    "val_split": "validation",
                    "prefetch_to_device": 2,
                },
            }
        ),
        data_rng=rng,
    )
    data = next(iter(loader.train_iter))
    assert isinstance(data["inputs"], jnp.ndarray)

@pytest.mark.parametrize("batch_size", [16, 32])
def test_random_resized_crops(batch_size):
    rng = jax.random.PRNGKey(0)
    # test rrc
    loader = scenic(
        OmegaConf.create(
            {
                "dataset_name": "bit",
                "batch_size": batch_size,
                "data_dtype_str": "float32",
                "dataset_configs": {
                    "dataset": "imagenet2012",
                    "num_classes": 1000,
                    "train_split": "train",
                    "val_split": "validation",
                    "pp_train": (
                        "decode"
                        "|value_range(-1, 1)"
                        "|onehot(1000, key='label', key_result='labels')"
                        "|random_resized_crops(height_large=224., width_large=224., height_small=224., width_small=224., n_small=6, n_large=2)"
                        "|keep('image', 'labels')"
                    ),
                    "pp_eval": (
                        "decode"
                        "|resize_small(256)"
                        "|central_crop(224)"
                        "|value_range(-1, 1)"
                        "|onehot(1000, key='label', key_result='labels')"
                        "|keep('image', 'labels')"
                    ),
                    "prefetch_to_device": 2,
                    "shuffle_buffer_size": 250_000
                },
            }
        ),
        data_rng=rng,
    )
    data = next(iter(loader.train_iter))
    print(data["inputs"].shape)
    assert isinstance(data["inputs"], jnp.ndarray)
    assert False
