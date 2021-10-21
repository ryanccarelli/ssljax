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
def byoltestconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="byol_conf.yaml")
    return cfg


class Identity(Augmentation):
    """
    Map image by identity.
    """

    def __call__(self, x, rng):
        return x


@register(Pipeline, "CPUPipeline")
class CPUPipeline(Pipeline):
    def __init__(self):
        super().__init__([AugmentationDistribution([Identity()])])
