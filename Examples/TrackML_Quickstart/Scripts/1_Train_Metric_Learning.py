"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""

import sys
import os
import yaml
import argparse
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

sys.path.append("../../")
from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(["-"]*20 + " Step 1: Running metric learning training " + ["-"]*20)

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(["-"]*20 + "a) Initialising model" + ["-"]*20)

    model = LayerlessEmbedding(metric_learning_configs)

    logging.info(["-"]*20 + "b) Running training" + ["-"]*20)

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"])

    trainer = Trainer(
        gpus=common_configs["gpus"],
        max_epochs=common_configs["max_epochs"],
        logger=logger
    )

    trainer.fit(model)

    logging.info(["-"]*20 + "c) Saving model" + ["-"]*20)

    os.makedirs(save_directory, exist_ok=True)
    model.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file)    

