"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
import sys
import os
import argparse
import logging
import yaml
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

import torch

sys.path.append("../../")
# sys.path.append('./')
from Pipelines.Common_Tracking_Example.LightningModules.Filter.Models.pyramid_filter import PyramidFilter
from utils import headline
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_filter_network.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Running filter training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    filter_configs = all_configs["filter_configs"]

    logging.info(headline("a) Initialising model"))

    model = PyramidFilter(filter_configs)

    logging.info(headline("b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "filter")
    os.makedirs(save_directory, exist_ok=True)
    logger = []
    for lg in filter_configs.get('loggers', []):
        if lg == 'CSVLogger':
            logger.append(CSVLogger(save_directory, name=common_configs["experiment_name"]))
        if lg == 'WandbLogger':
            logger.append(WandbLogger(name=common_configs['experiment_name'] + '_filter', project='TrackML', group='filter'))

    trainer = Trainer(
        strategy='ddp',
        accelerator='gpu',
        num_nodes=filter_configs.get('num_nodes') or os.environ.get('num_nodes') or 1,
        devices=common_configs["gpus"],
        max_epochs=filter_configs["max_epochs"],
        logger=logger
    )
    start = datetime.now()
    print(trainer.strategy.launcher is None)
    trainer.fit(model)
    logging.info(f"Training takes {datetime.now() - start}")

    logging.info(headline("c) Saving model") )

    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    trainer, model = train(config_file)    

