import jax.random
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from ssljax.data.dataloader import ScenicData, scenic


"""
# TODO: test dataloader arguments
@pytest.mark.parametrize("batch_size", [10, 20])
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


@pytest.mark.parametrize("batch_size", [10, 20])
def test_bit(batch_size):
    rng = jax.random.PRNGKey(0)
    # test passthrough
    loader = scenic(
        OmegaConf.create(
            {
                "dataset_name": "bit",
                "batch_size": batch_size,
                "data_dtype_str": "float32",
                "dataset_configs": {"pp_train": ("identity()"), "dataset": "imagenet2012", "num_classes": 1000, "train_split": "train", "val_split": "validation"},
            }
        ),
        data_rng=rng,
    )
    data = next(iter(loader.train_iter))
    print(data.shape)
    assert isinstance(data["inputs"], np.ndarray)

"""
@pytest.mark.parametrize("batch_size", [10, 20])
def test_random_resized_crop(batch_size):
    rng = jax.random.PRNGKey(0)
    # test passthrough
    loader = scenic(
        OmegaConf.create(
            {
                "dataset_name": "bit",
                "batch_size": batch_size,
                "data_dtype_str": "float32",
                "dataset_configs": {"pp_train": ("random_resized_crop(height=224., width=224.)"), "dataset": "imagenet2012", "num_classes": 1000, "train_split": "train", "val_split": "validation"},
            }
        ),
        data_rng=rng,
    )
    data = next(iter(loader.train_iter))
    print(data.shape)
    assert isinstance(data["inputs"], np.ndarray)
