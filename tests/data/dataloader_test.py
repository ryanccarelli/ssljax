import numpy as np
import pytest
from ssljax.data.dataloader import DataLoader, TorchData, TorchMNIST


# TODO: test dataloader arguments
@pytest.mark.parametrize("batch_size", [10, 20])
def test_dataloader(batch_size):
    loader = TorchMNIST(batch_size)
    assert isinstance(loader, TorchData)
    train, _ = next(iter(loader))
    assert isinstance(train, np.ndarray)
