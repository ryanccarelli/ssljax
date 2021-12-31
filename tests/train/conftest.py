import hydra
import pytest
from hydra import compose, initialize


@pytest.fixture
def basecpuconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="base_cpu_test.yaml")
    return cfg


@pytest.fixture
def dynamicscalingconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="dynamic_scaling_test.yaml")
    return cfg


@pytest.fixture
def pretrainedconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="pretrained_test.yaml")
    return cfg


@pytest.fixture
def byolconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="byol_test.yaml")
    return cfg
