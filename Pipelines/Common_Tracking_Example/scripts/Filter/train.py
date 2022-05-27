import sys
import os
import argparse
import yaml
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append("../../")
from LightningModules.Filter.Models.vanilla_filter import VanillaFilter
from LightningModules.Filter.Models.pyramid_filter import PyramidFilter
from LightningModules.Filter.Models.hetero_pyramid import HeteroPyramidFilter

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_gnn.py")
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

    if args.checkpoint is not None:
        default_configs = torch.load(args.checkpoint)["hyper_parameters"]
    
    wandb.init(config=default_configs, project=default_configs["project"])
    config = wandb.config

    print("Initialising model")
    print(time.ctime())
    model_name = eval(default_configs["model"])
    model = model_name(dict(config))
        
    checkpoint_callback = ModelCheckpoint(
        monitor="auc", mode="max", save_top_k=2, save_last=True
    )

    logger = WandbLogger(
        project=default_configs["project"],
        save_dir=default_configs["artifacts"],
    )

    if args.root_dir is None: 
        if "SLURM_JOB_ID" in os.environ:
            default_root_dir = os.path.join(".", os.environ["SLURM_JOB_ID"])
        else:
            default_root_dir = None
    else:
        default_root_dir = os.path.join(".", args.root_dir)
    
    trainer = Trainer(
        gpus=default_configs["gpus"], 
        num_nodes=default_configs["nodes"],
        max_epochs=default_configs["max_epochs"], 
        logger=logger, 
        strategy="ddp",
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir
    )
    trainer.fit(model, ckpt_path=args.checkpoint)



if __name__ == "__main__":

    main()
