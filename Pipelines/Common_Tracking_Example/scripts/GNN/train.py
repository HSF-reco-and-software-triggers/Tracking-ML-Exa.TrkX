import sys
import argparse
import yaml
import time

import torch
import numpy
import random
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

sys.path.append("../../")
from LightningModules.GNN.Models.checkpoint_pyramid import CheckpointedPyramid
from LightningModules.GNN.Models.interaction_gnn import InteractionGNN

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule

from pytorch_lightning import seed_everything
import wandb


class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()


def set_random_seed(seed):
    torch.random.manual_seed(seed)
    print("Random seed:", seed)
    seed_everything(seed)


def main():
    print("Running main")
    print(time.ctime())

    default_config_path = "default_config.yaml"

    with open(default_config_path) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=default_configs, project=default_configs["project"])
    config = wandb.config

    if "random_seed" in dict(config).keys():
        set_random_seed(dict(config)["random_seed"])

    print("Initialising model")
    print(time.ctime())
    model_name = eval(dict(config)["model"])
    model = model_name(dict(config))
    logger = WandbLogger(save_dir=default_configs["artifacts"])
    logger.watch(model, log="all")

    trainer = Trainer(gpus=1, max_epochs=default_configs["max_epochs"], logger=logger)
    trainer.fit(model)


if __name__ == "__main__":

    main()
