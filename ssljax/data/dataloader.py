import logging

import jax.numpy as jnp
import numpy as np
from ssljax.core.utils.register import register
from torch.utils import data
from torchvision.datasets import MNIST

logger = logging.getLogger(__name__)


class DataLoader(data.DataLoader):
    """
    SSLJax enforces Pytorch dataloaders inheriting from NumpyLoader.
    NumpyLoader implements collate for numpy arrays.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        **kwargs,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class FlattenAndCast:
    def __call__(self, pic):
        return np.expand_dims(np.ravel(np.array(pic, dtype=jnp.float32)), axis=-1)


# packaged dataloaders here


@register(DataLoader, "mnist")
def MNISTLoader(batch_size, **kwargs):
    mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())
    return DataLoader(mnist_dataset, batch_size=batch_size, num_workers=0, **kwargs)
