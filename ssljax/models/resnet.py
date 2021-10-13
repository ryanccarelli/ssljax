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
from ssljax.core.utils.register import register
from ssljax.models.model import Model

from ssljax.models.utils import BottleneckResNetBlock, ResNetBlock

@register(Model, "ResNet")
class ResNet(Model):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: str
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    def setup(self):
        if self.block_cls == "ResNetBlock":
            self.block_cls = ResNetBlock
        elif self.block_cls == "BottleneckResNetBlock":
            self.block_cls = BottleneckResNetBlock
        else:
            raise KeyError("must pass {ResNetBlock, BottleneckResNetBlock}")

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = conv(
            self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
