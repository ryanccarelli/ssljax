import os

import pytest
from hydra import compose, initialize
from ssljax.train.task import Task


class TestTrainer:
    """
    def test_train_cpu(self, cputestconfig):
        task = Task(cputestconfig)
        trainer = task.trainer
        trainer.train()

    def test_train_cpu_dynamic_scaling(self, cputestdynamicscalingconfig):
        task = Task(cputestdynamicscalingconfig)
        trainer = task.trainer
        trainer.train()
    """

    # def test_train_vit(self, cputestconfig):
    #     cputestconfig["model"]["branches"][0]["params"]["body"]["name"] = "VIT"
    #     cputestconfig["model"]["branches"][0]["params"]["body"]["params"] = {
    #         "config": "ViT-B_32",
    #         "num_classes": 2,
    #     }
    #     task = Task(cputestconfig)
    #     trainer = task.trainer
    #     trainer.train()


    def test_train_byol(self, byoltestconfig):
        task = Task(byoltestconfig)
        trainer = task.trainer
        trainer.train()



def byoltestconfig():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../train/conf")
    cfg = compose(config_name="byol_conf.yaml")
    return cfg
