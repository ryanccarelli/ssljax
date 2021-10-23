import logging

import jax.numpy as jnp
import numpy as np
from ssljax.core.utils.register import register
from torch.utils import data
from torchvision.datasets import CIFAR10, MNIST

logger = logging.getLogger(__name__)


class DataLoader(data.DataLoader):
    """
    SSLJax enforces Pytorch dataloaders inheriting from DataLoader.
    Dataloader collates numpy arrays.

    Args:
        dataset (torch.data.Dataset): torch dataset to load
        batch_size (int): batch size
        shuffle (bool): whether data will be shuffled
        num_workers (int): number of workers for distributed data
        pin_memory (bool): whether to pin memory
        drop_last (bool): whether to drop the last sample
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


# TODO. Add transformation composition class/operator

# packaged dataloaders here
@register(DataLoader, "mnist")
def MNISTLoader(batch_size, flatten=True, num_workers=0, **kwargs):
    """
    Dataloader for MNIST dataset.
    See http://www.pymvpa.org/datadb/mnist.html
    """
    if flatten:
        mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())
    else:
        mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=Cast())
    return DataLoader(
        mnist_dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )


@register(DataLoader, "cifar10")
def CIFAR10Loader(batch_size, flatten=False, num_workers=0, **kwargs):
    """
    Dataloader for CIFAR10 dataset.
    """
    if flatten:
        cifar_dataset = CIFAR10(
            "/tmp/cifar10/", download=True, transform=FlattenAndCast()
        )
    else:
        cifar_dataset = CIFAR10("/tmp/cifar10/", download=True, transform=Cast())
    return DataLoader(cifar_dataset, batch_size=batch_size, num_workers=0, **kwargs)


@register(DataLoader, "cifar100")
def CIFAR100Loader(batch_size, flatten=False, num_workers=0, **kwargs):
    """
    Dataloader for CIFAR100 dataset.
    """
    if flatten:
        cifar_dataset = CIFAR100(
            "/tmp/cifar100/", download=True, transform=FlattenAndCast()
        )
    else:
        cifar_dataset = CIFAR100("/tmp/cifar100/", download=True, transform=Cast())
    return DataLoader(
        cifar_dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
