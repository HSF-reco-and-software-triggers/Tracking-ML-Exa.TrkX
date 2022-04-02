import sys
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

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_gnn.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="default_config.yaml")
    add_arg("checkpoint", nargs="?", default=None)
    return parser.parse_args()


def main():
    
    print("Running main")
    print(time.ctime())

    args = parse_args()

    with open(args.config) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    # if args.checkpoint is not None:
    #     default_configs = torch.load(args.checkpoint)["hyper_parameters"]
    
    wandb.init(config=default_configs, project=default_configs["project"])
    config = wandb.config

    print("Initialising model")
    print(time.ctime())
    model_name = eval(default_configs["model"])
    model = model_name(dict(config))
        
    logger = WandbLogger(
        project=default_configs["project"],
        save_dir=default_configs["artifacts"],
    )
    
    trainer = Trainer(
        gpus=default_configs["gpus"], 
        max_epochs=default_configs["max_epochs"], 
        logger=logger, 
        strategy="ddp"
    )
    trainer.fit(model, ckpt_path=args.checkpoint)



if __name__ == "__main__":

    main()
