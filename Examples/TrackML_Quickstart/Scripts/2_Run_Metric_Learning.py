"""
This script runs step 2 of the TrackML Quickstart example: Inferencing the metric learning to construct graphs.
"""

import sys
import os
import yaml
import argparse
import logging
import torch

sys.path.append("../../")
from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from Pipelines.TrackML_Example.notebooks.build_embedding import EmbeddingInferenceBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("2_Run_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(["-"]*20 + " Step 2: Constructing graphs from metric learning model " + ["-"]*20)

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(["-"]*20 + "a) Loading trained model" + ["-"]*20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayerlessEmbedding.load_from_checkpoint(os.path.join(common_configs["artifact_directory"], "metric_learning", common_configs["experiment_name"]+".ckpt")).to(device)

    logging.info(["-"]*20 + "b) Running inferencing" + ["-"]*20)
    graph_builder = EmbeddingInferenceBuilder(model, metric_learning_configs["train_split"], overwrite=True, knn_max=1000, radius=metric_learning_configs["r_test"])
    graph_builder.build()



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file) 