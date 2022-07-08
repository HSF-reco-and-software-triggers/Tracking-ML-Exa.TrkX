import sys
import argparse
import yaml
import time
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append("../../")
from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from LightningModules.SuperEmbedding.Models.gravmetric import UndirectedGravMetric, DirectedGravMetric
from LightningModules.SuperEmbedding.Models.halftwin import UndirectedHalfTwinEmbedding
from LightningModules.SuperEmbedding.Models.halftwin import DirectedHalfTwinEmbedding
from LightningModules.EmbeddingSandbox.Models.undirected_embedding import UndirectedEmbedding
from LightningModules.EmbeddingSandbox.Models.directed_embedding import DirectedEmbedding

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="default_config.yaml")
    add_arg("root_dir", nargs="?", default=None)
    add_arg("checkpoint", nargs="?", default=None)
    return parser.parse_args()


def main():
    print("Running main")
    print(time.ctime())

    args = parse_args()

    with open(args.config) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    print("Initialising model")
    print(time.ctime())
    model_name = eval(default_configs["model"])
    model = model_name(default_configs)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=2, save_last=True
    )

    logger = WandbLogger(
        project=default_configs["project"],
        save_dir=default_configs["artifacts"],
    )
    logger.watch(model, log="all")

    if args.root_dir is None: 
        if "SLURM_JOB_ID" in os.environ:
            default_root_dir = os.path.join(".", os.environ["SLURM_JOB_ID"])
        else:
            default_root_dir = None
    else:
        default_root_dir = os.path.join(".", args.root_dir)

    trainer = Trainer(
        gpus=default_configs["gpus"], 
        max_epochs=default_configs["max_epochs"], 
        logger=logger, 
        strategy="ddp",
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir
    )
    trainer.fit(model)


if __name__ == "__main__":

    main()
