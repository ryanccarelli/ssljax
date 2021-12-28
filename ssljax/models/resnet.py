# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CONTENT HAS BEEN MODIFIED FROM ORIGINAL SOURCE

from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn
from omegaconf import DictConfig
from ssljax.core import register
from ssljax.models.model import Model
from scenic.projects.baselines.resnet import ResNet as Resnet
from scenic.projects.baselines.resnet import BLOCK_SIZE_OPTIONS
from scenic.projects.baselines.bit_resnet import BitResNet as Bitresnet
from scenic.projects.baselines.axial_resnet import AxialResNet as Axialresnet

@register(Model, "ResNet")
class ResNet(Model):
    """
    Flax implementation of a resnet.
    We wrap the ResNet model in `<scenic> https://github.com/google-research/scenic`_.

    Args:
        config (omegaconf.DictConfig): configuration
    """

    config: DictConfig

    def setup(self):
        self.model = Resnet(**self.config)
        self.num_blocks = len(BLOCK_SIZE_OPTIONS[self.config.num_layers])

    @nn.compact
    def __call__(self, x):
        x = self.model(x)
        # this is directly from https://github.com/google-research/scenic/blob/c2140913a9a3fb7b7c54d50c6db7df0e6cf92ba1/scenic/projects/baselines/resnet.py#L126
        return x[f"stage_{self.num_blocks+1}"]

"""
# from https://github.com/google-research/scenic/blob/c2140913a9a3fb7b7c54d50c6db7df0e6cf92ba1/scenic/projects/baselines/resnet.py#L147
BLOCK_SIZE_OPTIONS = {
    5: ([1], True),  # Only strided blocks. Total stride 4.
    8: ([1, 1], True),  # Only strided blocks. Total stride 8.
    11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
    14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
    9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True)
}
"""

@register(Model, "BitResNet")
class BitResNet(Model):
    """
    Flax implementation of a resnet.
    We wrap the BitResNet model in `<scenic> https://github.com/google-research/scenic`_.

    Args:
        config (omegaconf.DictConfig): configuration
    """

    config: DictConfig

    def setup(self):
        self.model = Bitresnet(**self.config)

    @nn.compact
    def __call__(self, x):
        return self.model(x)


@register(Model, "AxialResNet")
class AxialResNet(Model):
    """
    Flax implementation of a resnet.
    We wrap the BitResNet model in `<scenic> https://github.com/google-research/scenic`_.

    Args:
        config (omegaconf.DictConfig): configuration
    """

    config: DictConfig

    def setup(self):
        self.model = Axialresnet(**self.config)

    @nn.compact
    def __call__(self, x):
        return self.model(x)
