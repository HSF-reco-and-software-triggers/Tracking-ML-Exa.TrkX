import sys
import argparse
import yaml
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

sys.path.append("../../")
from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="default_config.yaml")
    return parser.parse_args()


def main():
    print("Running main")
    print(time.ctime())

    args = parse_args()

    with open(args.config) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    print("Initialising model")
    print(time.ctime())
    model = LayerlessEmbedding(default_configs)

    logger = WandbLogger(
        project=default_configs["project"],
        save_dir=default_configs["artifacts"],
    )
    logger.watch(model, log="all")

    trainer = Trainer(
        gpus=default_configs["gpus"], 
        max_epochs=default_configs["max_epochs"], 
        logger=logger, 
        strategy="ddp"
    )
    trainer.fit(model)


if __name__ == "__main__":

    main()
