from typing import Any, Callable, Iterable, List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import lecun_normal, zeros
from flax.training import train_state
from jax import core, dtypes
from jax.random import truncated_normal
from omegaconf import DictConfig
from ssljax.core.utils import register
from ssljax.models.model import Model

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


@register(Model, "MLP")
class MLP(Model):
    """
    Base implementation of multilayer perceptron.

    Args:
        layer_dims(List[int]): list indicating number of neurons in each layer
        dtype: jnp datatype
        dropout_prob(float): dropout rate hyperparameteri
        batch_norm(bool): whether to use batchnorm between layers
        batch_norm_params(dict): params to be passed to nn.BatchNorm
        activation_name(str): activation function
    """

    layer_dims: List[int]
    batch_norm_params: dict
    dtype: jax._src.numpy.lax_numpy._ScalarMeta = jnp.float32
    dropout_prob: float = 0.0
    batch_norm: bool = False
    activation_name: Callable = "relu"

    def setup(self):
        if self.activation_name == "relu":
            self.activation = nn.relu
        elif self.activation_name == "gelu":
            self.activation = nn.gelu
        else:
            raise KeyError("activation must be in {relu, gelu}")
        layers = []
        for layer in self.layer_dims[:-1]:
            layers.append(nn.Dense(layer, dtype=self.dtype))
            if self.batch_norm:
                layers.append(nn.BatchNorm(**self.batch_norm_params))
            layers.append(self.activation)
            if self.dropout_prob:
                layers.append(nn.Dropout(rate=self.dropout_prob))
        layers.append(nn.Dense(self.layer_dims[-1], dtype=self.dtype))
        self.layers = layers

    @nn.compact
    def __call__(self, x, train=False):
        for layer in self.layers:
            if isinstance(layer, flax.linen.stochastic.Dropout):
                x = layer(x, deterministic=True)
            else:
                x = layer(x)
        return x


def _truncated_normal(lower, upper, mean=0, stddev=1, dtype=jnp.float_):
    """
    Sample random values from zero-centered, truncated normal distribution.
    """

    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        key_tn, key = jax.random.split(key)
        return (
            truncated_normal(
                key=key_tn, lower=lower/stddev, upper=upper/stddev, shape=shape, dtype=dtype
            )
            * stddev + mean
        )

    return init


@register(Model, "DINOMLP")
class DINOMLP(Model):
    """
    DINO implementation of multilayer perceptron.

    Args:
        layer_dims(List[int]): list indicating number of neurons in each layer
        dtype: jnp datatype
        dropout_prob(float): dropout rate hyperparameteri
        batch_norm(bool): whether to use batchnorm between layers
        batch_norm_params(dict): params to be passed to nn.BatchNorm
        activation_name(str): activation function
        kernel_init(Callable[[PRNGKey, Shape, Dtype], Array]): linear layer kernel init function
        bias_init(Callable[[PRNGKey, Shape, Dtype], Array]): linear layer bias init function
    """

    layer_dims: List[int]
    batch_norm_params: dict
    dtype: jax._src.numpy.lax_numpy._ScalarMeta = jnp.float32
    dropout_prob: float = 0.0
    batch_norm: bool = False
    activation_name: Callable = "gelu"
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = _truncated_normal(
        stddev=0.02, lower=-2.0, upper=2.0
    )
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    def setup(self):
        if self.activation_name == "relu":
            self.activation = nn.relu
        elif self.activation_name == "gelu":
            self.activation = nn.gelu
        else:
            raise KeyError("activation must be in {relu, gelu}")
        layers = []
        for layer in self.layer_dims[:-1]:
            layers.append(
                nn.Dense(
                    layer,
                    dtype=self.dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            if self.batch_norm:
                layers.append(nn.BatchNorm(**self.batch_norm_params))
            layers.append(self.activation)
            if self.dropout_prob:
                layers.append(nn.Dropout(rate=self.dropout_prob))
        layers.append(nn.Dense(self.layer_dims[-1], dtype=self.dtype, use_bias=False))
        self.layers = layers

    @nn.compact
    def __call__(self, x, train=False):
        for layer in self.layers:
            if isinstance(layer, flax.linen.stochastic.Dropout):
                x = layer(x, deterministic=True)
            else:
                x = layer(x)
        return x


@register(Model, "DINOHead")
class DINOHead(Model):
    """
    Adapted from https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257
    Compose with 3 layer GELU MLP, no batchnorm, no dropout for paper-version DINO.

    Args:
        out_dim(int): dimension of output
        dtype: jnp datatype
        kernel_init(Callable[[PRNGKey, Shape, Dtype], Array]): linear layer kernel init function
    """

    out_dim: int
    dtype: jax._src.numpy.lax_numpy._ScalarMeta = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = _truncated_normal(
        stddev=0.02, lower=-2.0, upper=2.0
    )

    def setup(self):
        self.linear = nn.Dense(
            dtype=self.dtype, kernel_init=self.kernel_init, use_bias=False
        )

    @nn.compact
    def __call__(self, x):
        x = jnp.linalg.norm(x, ord=2, axis=-1)
        x = self.linear(x)
        return jnp.linalg.norm(x, ord=1, axis=-1)
