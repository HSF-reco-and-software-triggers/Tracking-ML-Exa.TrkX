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
from Pipelines.TrackML_Example_Full.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils import headline
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 1: Running metric learning training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(headline("a) Initialising model"))

    model = LayerlessEmbedding(metric_learning_configs)

    logging.info(headline("b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    logger = []
    for lg in metric_learning_configs.get('loggers', []):
        if lg == 'CSVLogger':
            logger.append(CSVLogger(save_directory, name=common_configs["experiment_name"]))
        if lg == 'WandbLogger':
            logger.append(WandbLogger(name=common_configs['experiment_name'], project='TrackML'))

    trainer = Trainer(
        strategy='ddp',
        accelerator='gpu',
        num_nodes=metric_learning_configs.get('num_nodes') or os.environ.get('num_nodes') or 1,
        devices=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=logger
    )
    start = datetime.now()
    trainer.fit(model)
    logging.info(f"Training takes {datetime.now() - start}")

    logging.info(headline("c) Saving model") )

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    trainer, model = train(config_file)    

