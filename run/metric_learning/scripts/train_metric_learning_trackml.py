"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import sys
import os
import argparse
import logging
import yaml
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

import torch

sys.path.append("/global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX/")
# sys.path.append('./')
from Pipelines.TrackML_Example_Dev.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils import headline
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    # add_arg('resume_version', default=None, help="")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 1: Running metric learning training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(headline("a) Initialising model"))

    model = LayerlessEmbedding(metric_learning_configs)

    save_directory = os.path.join(common_configs["artifact_directory"])
    os.makedirs(save_directory, exist_ok=True)
    logger = []
    for lg in metric_learning_configs.get('loggers', []):
        if lg == 'CSVLogger':
            csv_logger = CSVLogger(save_directory, name=common_configs["experiment_name"])
            logger.append(csv_logger)
        if lg == 'WandbLogger':
            wandb_logger = WandbLogger(project=common_configs.get('project', 'Itk'), group="metric_learning", save_dir=save_directory)
            logger.append(wandb_logger)

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        num_nodes=os.environ.get('SLURM_JOB_NUM_NODES', 1), # metric_learning_configs.get('num_nodes') or os.environ.get('num_nodes') or 1,
        devices=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=logger
    )

    logging.info(headline("b) Running training" ))
    start = datetime.now()
    trainer.fit(model)
    logging.info(f"Training takes {datetime.now() - start}")

    logging.info(headline("c) Saving model") )

    model_name = common_configs['experiment_name'] + f'{ ("_" + wandb_logger.name) if "WandbLogger" in metric_learning_configs.get("loggers", []) else ""}' + f'{ ("_version_" + str(csv_logger.version)) if "CSVLogger" in metric_learning_configs.get("loggers", []) else "" }' + '.ckpt'
    trainer.save_checkpoint(os.path.join(save_directory, model_name))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    trainer, model = train(config_file)    

