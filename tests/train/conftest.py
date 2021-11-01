import flax.linen as nn
import hydra
import pytest
from hydra import compose, initialize
from ssljax.augment.augmentation.augmentation import (Augmentation,
                                                      AugmentationDistribution)
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.core.utils import register


@pytest.fixture
def cputestconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="cpu_conf.yaml")
    return cfg


@pytest.fixture
def cputestdynamicscalingconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="cpu_conf_dynamic_scaling.yaml")
    return cfg


@pytest.fixture
def cputestpretrainedconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="cpu_conf_pretrained.yaml")
    return cfg


@pytest.fixture
def byoltestconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="byol_conf.yaml")
    return cfg
