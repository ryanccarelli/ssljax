import pytest
import jax.numpy as jnp
from ssljax.models import ResNet, ViT, MLP
from omegaconf import OmegaConf


@pytest.fixture
def multi_resolution_data_list():
    # these are data shapes for a standard dino configuration
    data = [jnp.ones((94, 94, 3)) if x < 6 else jnp.ones((224, 224, 3)) for x in range(8)]
    return data

@pytest.fixture
def minimal_resnet():
    resnet_config = {
        "num_classes": None,
        "num_filters": 64,
        "num_layers": 5,
    }
    resnet = ResNet(OmegaConf.create(resnet_config))
    return resnet


@pytest.fixture
def minimal_vit():
    vit_config = {
        "patch_size": 16,
        "num_classes": 20,
        "mlp_dim": 1024,
        "num_layers": 8,
        "num_heads": 8,
        "hidden_size": 384,
        "representation_size": None,
        "dropout_rate": 0.1,
        "attention_dropout_rate": 0.,
        "classifier": "token",
        "dtype": "float32",
    }
    vit = ViT(OmegaConf.create(vit_config))
    return vit
