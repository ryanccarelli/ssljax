import os

import pytest
from hydra import compose, initialize
from ssljax.core import download_from_url
from ssljax.train.task import Task


class TestTrainer:
    def test_cpu(self, basecpuconfig):
        task = Task(basecpuconfig)
        trainer = task.trainer
        trainer.train()

    def test_dynamic_scaling(self, dynamicscalingconfig):
        task = Task(dynamicscalingconfig)
        trainer = task.trainer
        trainer.train()

    def test_train_cpu_pretrained(self, pretrainedconfig):
        task = Task(pretrainedconfig)
        trainer = task.trainer
        trainer.train()

    @pytest.mark.gpu
    def test_train_byol(self, byolconfig):
        download_from_url(
            "https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k",
            "tests/train/conf",
        )
        task = Task(byolconfig)
        trainer = task.trainer
        trainer.train()
    """
    @pytest.mark.gpu
    def test_train_dino_vit(self, dinovitconfig):
        # TODO: download imagenet
        task = Task(dinovitconfig)
        trainer = task.trainer
        trainer.train()
    """
