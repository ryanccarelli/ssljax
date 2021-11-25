# test ssljax/models/branch/branch.py

import pytest
from omegaconf import DictConfig
# TODO: fix this
from ssljax.models.branch.branch import Branch


# how to mock config here?
#
@pytest.mark.parametrize("stage_name", [None])
class TestBranch:
    def test_setup_call(self, stage_name):
        stage_name
        config = DictConfig()
        branch = Branch(config=config)
