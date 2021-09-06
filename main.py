import hydra
from omegaconf import DictConfig, OmegaConf
from ssljax.train import Trainer, Task

@hydra.main(config_path="ssljax/config/default-conf", config_name="config")
def entry(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    task = Task(config=cfg)


if __name__ == "__main__":
    entry()
