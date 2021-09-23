import numpy as np
import pytest
from ssljax.data.dataloaders import MNISTLoader, NumpyLoader


# TODO: test dataloader arguments
class TestDataloader:
    @pytest.mark.parameterize("dataloader", [MNISTLoader])
    @pytest.mark.parameterize("batch_size", [10, 20])
    def test_dataloader(dataloader, batch_size):
        loader = dataloader(batch_size)
        assert isinstance(loader, NumpyLoader)
        train, _ = next(iter(loader))
        assert isinstance(train, np.ndarray)
