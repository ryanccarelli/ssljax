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
from ssljax.core import register
from ssljax.models.model import Model

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


@register(Model, "MLP")
class MLP(Model):
    """
    Base implementation of multilayer perceptron. Args format is for consistency
    with Scenic.

    Args:
        config (omegaconf.DictConfig): configuration containing
            layer_dims(List[int]): list indicating number of neurons in each layer
            dtype: jnp datatype
            dropout_prob(float): dropout rate hyperparameteri
            batch_norm(bool): whether to use batchnorm between layers
            batch_norm_params(dict): params to be passed to nn.BatchNorm
            activation_name(str): activation function
            batch_norm_final_layer(bool): whether to use batchnorm after the final layer
            batch_norm_final_layer_params(dict): params to be passed to nn.BatchNorm
    """

    config: DictConfig

    def setup(self):
        assert (
            isinstance(self.config.layer_dims, list) and isinstance(i, int)
            for i in self.config.layer_dims
        ), "layer dimensions must be a list of integers"
        assert self.config.activation_name in [
            "relu",
            "gelu",
        ], "supported activations are {'relu', 'gelu'}"
        assert ("batch_norm_final_layer" in self.config or not (len(self.config.layer_dims) == 1 and (self.config.batch_norm or self.config.dropout_prob))), \
            "Single layer mlp does not permit batch norm or dropout"
        if self.config.activation_name == "relu":
            self.activation = nn.relu
        elif self.config.activation_name == "gelu":
            self.activation = nn.gelu
        else:
            raise KeyError("activation must be in {relu, gelu}")
        if "layer_bias" in self.config:
            self.layer_bias = self.config.layer_bias
        else:
            self.layer_bias = True
        dtypedict = {
            "float32": jnp.float32,
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
        }
        self.dtype = dtypedict[self.config.dtype]
        layers = []
        for layer in self.config.layer_dims[:-1]:
            layers.append(nn.Dense(layer, self.layer_bias, dtype=self.config.dtype))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm(**self.config.batch_norm_params))
            layers.append(self.activation)
            if self.config.dropout_prob:
                layers.append(nn.Dropout(rate=self.config.dropout_prob))
        layers.append(
            nn.Dense(
                self.config.layer_dims[-1], self.layer_bias, dtype=self.config.dtype
            )
        )
        if "batch_norm_final_layer" in self.config:
            layers.append(nn.BatchNorm(**self.config.batch_norm_final_layer_params))
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
                    key=key_tn,
                    lower=lower / stddev,
                    upper=upper / stddev,
                    shape=shape,
                    dtype=dtype,
                )
                * stddev
                + mean
        )

    return init


@register(Model, "DINOMLP")
class DINOMLP(MLP):
    """
    DINO implementation of multilayer perceptron.

    Dense layer kernel init is truncated normal.
    Final dense layer has bias.

    Paper composes 3 layer DINOMLP with GELU, no batchnorm, no dropout
    with DINOProj

    Args:
        config (omegaconf.DictConfig): configuration containing
            layer_dims(List[int]): list indicating number of neurons in each layer
            dtype: jnp datatype
            dropout_prob(float): dropout rate hyperparameteri
            batch_norm(bool): whether to use batchnorm between layers
            batch_norm_params(dict): params to be passed to nn.BatchNorm
            activation_name(str): activation function
        kernel_init(Callable[[PRNGKey, Shape, Dtype], Array]): linear layer kernel init function
        bias_init(Callable[[PRNGKey, Shape, Dtype], Array]): linear layer bias init function
    """

    config: DictConfig
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = _truncated_normal(
        stddev=0.02, lower=-2.0, upper=2.0
    )
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    def setup(self):
        assert (
            isinstance(self.config.layer_dims, list) and isinstance(i, int)
            for i in self.config.layer_dims
        ), "layer dimensions must be a list of integers"
        assert self.config.activation_name in [
            "relu",
            "gelu",
        ], "supported activations are {'relu', 'gelu'}"
        if self.activation_name == "relu":
            self.activation = nn.relu
        elif self.activation_name == "gelu":
            self.activation = nn.gelu
        else:
            raise KeyError("activation must be in {relu, gelu}")
        dtypedict = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}
        self.dtype = dtypedict[self.config.dtype]
        layers = []
        for layer in self.layer_dims[:-1]:
            layers.append(
                # default kernel is glorot
                # bias is default, same as MLP
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


@register(Model, "DINOProj")
class DINOProj(Model):
    """
    Adapted from https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257
    Compose with 3 layer GELU MLP, no batchnorm, no dropout for paper-version DINO.

    Args:
        out_dim(int): dimension of output
        dtype: jnp datatype
        kernel_init(Callable[[PRNGKey, Shape, Dtype], Array]): linear layer kernel init function
    """

    config: DictConfig
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = _truncated_normal(
        stddev=0.02, lower=-2.0, upper=2.0
    )

    def setup(self):
        dtypedict = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}
        self.dtype = dtypedict[self.config.dtype]
        self.linear = nn.Dense(
            self.config.out_dim, dtype=self.dtype, kernel_init=self.kernel_init, use_bias=False
        )

    @nn.compact
    def __call__(self, x):
        x = jnp.linalg.norm(x, ord=2, axis=-1)
        x = self.linear(x)
        return jnp.linalg.norm(x, ord=1, axis=-1)
