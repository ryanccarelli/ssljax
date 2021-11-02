import os

import pytest
from hydra import compose, initialize
from ssljax.train.task import Task
from ssljax.core.utils import download_from_url


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

    def test_train_cpu_pretrained(self, cputestpretrainedconfig):
        task = Task(cputestpretrainedconfig)
        trainer = task.trainer
        trainer.train()

    def test_train_vit(self, cputestconfig):
        cputestconfig["model"]["branches"][0]["params"]["body"]["name"] = "VIT"
        cputestconfig["model"]["branches"][0]["params"]["body"]["params"] = {
            "config": "ViT-B_32",
            "num_classes": 2,
        }
        task = Task(cputestconfig)
        trainer = task.trainer
        trainer.train()
    """

    def test_train_byol(self, byoltestconfig):
        download_from_url("https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k", "tests/train/conf")
        task = Task(byoltestconfig)
        trainer = task.trainer
        trainer.train()
