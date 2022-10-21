"""
This script runs step 5 of the exatrkx pipeline: Training the graph neural network.
"""
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule
import sys
import os
import argparse
import logging
import yaml
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import wandb

import torch

sys.path.append(os.environ['EXATRKX_WD'])
os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

from Pipelines.TrackML_Example.LightningModules.GNN.Models.interaction_gnn import InteractionGNN
# from Pipelines.Common_Tracking_Example.scripts.GNN.train_gnn import CustomDDPPlugin
from run.utils.convenience_utils import headline
from datetime import datetime

class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("3_Train_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="configs/gnn_configs_trackml.yaml")
    add_arg('--load_ckpt', required=False)
    add_arg('--load_model', required=False)
    return parser.parse_args()


def train(config_file="pipeline_config.yaml", load_ckpt=None, load_model=None):

    logging.info(headline(" Step 5: Running GNN training "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]

    logging.info(headline("a) Initialising model" ))

    model = InteractionGNN(gnn_configs)
    if load_model is not None:
        model = model.load_from_checkpoint(load_model, **gnn_configs)

    logging.info(headline( "b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], 'gnn')
    os.makedirs(save_directory, exist_ok=True)
    logger = []
    for lg in gnn_configs.get('loggers', []):
        if lg == 'CSVLogger':
            csv_logger = CSVLogger(save_directory, name=common_configs["experiment_name"])
            logger.append(csv_logger)
        if lg == 'WandbLogger':
            wandb_logger = WandbLogger(project=common_configs.get('wandb_project', 'TrackML'), group="gnn", save_dir=save_directory)
            logger.append(wandb_logger)

    trainer = Trainer(
        strategy=CustomDDPPlugin(find_unused_parameters=False),
        accelerator='gpu',
        num_nodes=os.environ.get('SLURM_JOB_NUM_NODES', 1), # metric_learning_configs.get('num_nodes') or os.environ.get('num_nodes') or 1,
        devices=common_configs["gpus"],
        max_epochs=gnn_configs["max_epochs"],
        logger=logger
    )

    logging.info(headline("b) Running training" ))
    start = datetime.now()
    trainer.fit(model, ckpt_path=load_ckpt)
    logging.info(f"Training takes {datetime.now() - start}")

    logging.info(headline("c) Saving model") )

    model_name = common_configs['experiment_name'] + f'{ ("_" + wandb.run.name) if "WandbLogger" in gnn_configs.get("loggers", []) else ""}' + f'{ ("_version_" + str(csv_logger.version)) if "CSVLogger" in gnn_configs.get("loggers", []) else "" }' + '.ckpt'
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(model_save_dir, model_name))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    load_ckpt = args.load_ckpt
    load_model = args.load_model

    trainer, model= train(config_file, load_ckpt, load_model)    

