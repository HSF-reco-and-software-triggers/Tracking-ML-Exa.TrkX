"""
This script runs step 3 of the TrackML Quickstart example: Training the graph neural network.
"""

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.GNN.Models.interaction_gnn import InteractionGNN
from utils.convenience_utils import headline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("3_Train_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline(" Step 3: Running GNN training "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]

    logging.info(headline("a) Initialising model" ))

    model = InteractionGNN(gnn_configs)

    logging.info(headline( "b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "gnn")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"])

    trainer = Trainer(
        gpus=common_configs["gpus"],
        max_epochs=gnn_configs["max_epochs"],
        logger=logger
    )

    trainer.fit(model)

    logging.info(headline( "c) Saving model" ))
    
    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file)    

