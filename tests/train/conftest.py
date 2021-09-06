import pytest
from hydra import compose, initialize


@pytest.fixture
def nullconfig():
    initialize(config_path="../train/conf/null_conf"):
    cfg = compose(config_name="config")
    return cfg

class CPUOnlineBranch:
    pass
