"""Main entrypoint. Modified from Scenic"""

from pathlib import Path

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
from absl import flags
from hydra import compose, initialize
from omegaconf import DictConfig
from scenic import app
from scenic.model_lib import models
from scenic.train_lib import train_utils, trainers

from ssljax.train.task import Task

FLAGS = flags.FLAGS


def main(
    rng: jnp.ndarray,
    config: str,
):
    """Main function for the Scenic."""

    # Enable wrapping of all module calls in a named_call for easier profiling:
    nn.enable_named_call()

    config = Path(config)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config.parents[0])
    cfg = compose(config_name=config.name)
    task = Task(cfg)
    # TODO: support train and eval configs
    task.trainer.train()


if __name__ == "__main__":
    app.run(main=main)
