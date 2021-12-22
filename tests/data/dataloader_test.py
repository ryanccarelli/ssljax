import jax.random
import numpy as np
import pytest
from ssljax.data.dataloader import Scenic, ScenicData


# TODO: test dataloader arguments
@pytest.mark.parametrize("batch_size", [10, 20])
def test_dataloader(batch_size):
    rng = jax.random.PRNGKey(0)
    loader = Scenic({"dataset_name": "mnist", "batch_size": batch_size}, data_rng=rng)
    assert isinstance(loader, ScenicData)
    train, _ = next(iter(loader))
    assert isinstance(train, np.ndarray)
