import numpy as np
import pytest
from ssljax.data.dataloader import DataLoader, MNISTLoader


# TODO: test dataloader arguments
@pytest.mark.parametrize("batch_size", [10, 20])
def test_dataloader(batch_size):
    loader = MNISTLoader(batch_size)
    assert isinstance(loader, DataLoader)
    train, _ = next(iter(loader))
    assert isinstance(train, np.ndarray)
