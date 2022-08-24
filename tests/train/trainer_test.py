import os

import pytest
from hydra import compose, initialize
from ssljax.core import download_from_url
from ssljax.train.task import Task


class TestTrainer:
    def test_cpu(self, basecpuconfig):
        os.system("python ssljax/main.py --config='../train/conf/base_cpu_test.yaml'")

    def test_dynamic_scaling(self, dynamicscalingconfig):
        os.system(
            "python ssljax/main.py --config='../train/conf/dynamic_scaling_test.yaml'"
        )

    def test_train_cpu_pretrained(self, pretrainedconfig):
        os.system("python ssljax/main.py --config='../train/conf/pretrained_test.yaml'")

    @pytest.mark.gpu
    def test_train_byol(self, byolconfig):
        download_from_url(
            "https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k",
            "tests/train/conf",
        )
        os.system("python ssljax/main.py --config='../train/conf/byol_test.yaml'")

    """
    @pytest.mark.gpu
    def test_train_dino_vit(self, dinovitconfig):
        download_from_url(
            "https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k",
            "tests/train/conf",
        )
        os.system("python ssljax/main.py --config='../train/conf/dino_vit_test.yaml'")
    """
