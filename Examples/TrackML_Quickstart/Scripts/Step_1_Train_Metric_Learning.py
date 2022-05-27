"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""

import sys
import os
import yaml
import argparse
import logging

from pytorch_lightning import Trainer

sys.path.append("../../")
# sys.path.append('./')
from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils import headline

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

    trainer = Trainer(
        gpus=common_configs["gpus"],
        max_epochs=common_configs["max_epochs"]
    )

    trainer.fit(model)

    logging.info(headline("c) Saving model") )

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file)    

