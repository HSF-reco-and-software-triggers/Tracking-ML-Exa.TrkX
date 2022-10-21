"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import sys
import os
import argparse
import logging
import yaml
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import wandb
import numpy as np
import torch

sys.path.append("/global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX/")
# sys.path.append('./')
from Pipelines.TrackML_Example_Dev.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from Pipelines.TrackML_Example_Dev.LightningModules.Embedding.embedding_base import WeightRampCallback
from utils import headline
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    add_arg('--load_ckpt', required=False, help="")
    add_arg('--load_model', required=False)
    return parser.parse_args()


def train(config_file="pipeline_config.yaml", load_ckpt=None, load_model=None):

    logging.info(headline("Step 1: Running metric learning training"))

    start = datetime.now()

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]
    metric_learning_configs['load_model'] = load_model
    metric_learning_configs['load_ckpt'] = load_ckpt

    logging.info(headline("a) Initialising model"))

    if load_model is not None:
        logging.info(f"Loading model from {load_model}")
        model = LayerlessEmbedding.load_from_checkpoint(load_model, hparams=metric_learning_configs)
    else:
        model = LayerlessEmbedding(metric_learning_configs)
    

    save_directory = os.path.join(common_configs["artifact_directory"], 'metric_learning')
    os.makedirs(save_directory, exist_ok=True)

    logger = []
    for lg in metric_learning_configs.get('loggers', []):
        if lg == 'CSVLogger':
            csv_logger = CSVLogger(save_dir=save_directory, name=common_configs["experiment_name"], version=str(start))
            logger.append(csv_logger)
        if lg == 'WandbLogger':
            wandb_logger = WandbLogger(project=common_configs.get('project', 'Itk'), group="metric_learning", save_dir=save_directory, log_model=True)
            logger.append(wandb_logger)
        if lg == 'TensorBoardLogger':
            tb_logger = TensorBoardLogger(save_dir=save_directory, name=common_configs['experiment_name'], version=str(start))
            logger.append(tb_logger)
    callback = []
    for cb in metric_learning_configs.get('callbacks', []):
        if cb == 'WeightRampCallback':
            callback.append(WeightRampCallback())


    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu' if torch.cuda.is_available() else None,
        num_nodes=os.environ.get('SLURM_JOB_NUM_NODES') or 1, # metric_learning_configs.get('num_nodes') or os.environ.get('num_nodes') or 1,
        devices=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=logger,
        callbacks=callback
    )

    logging.info(headline("b) Running training" ))
    
    trainer.fit(model, ckpt_path=load_ckpt)
    logging.info(f"Training takes {datetime.now() - start}")

    logging.info(headline("c) Saving model") )

    if torch.cuda.current_device() == 0:
        model_name = common_configs['experiment_name'] + f'{ ("_" + wandb.run.name) if "WandbLogger" in metric_learning_configs.get("loggers", []) else ""}' + f'{ ("_version_" + str(csv_logger.version)) if "CSVLogger" in metric_learning_configs.get("loggers", []) else "" }' + '.ckpt'
        model_save_dir = 'models'
        os.makedirs(model_save_dir, exist_ok=True)
        trainer.save_checkpoint(os.path.join(model_save_dir, model_name))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    load_ckpt = args.load_ckpt
    load_model = args.load_model

    trainer, model = train(config_file, load_ckpt, load_model)    

