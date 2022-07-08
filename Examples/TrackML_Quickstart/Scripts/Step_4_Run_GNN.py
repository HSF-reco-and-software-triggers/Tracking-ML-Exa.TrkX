"""
This script runs step 4 of the TrackML Quickstart example: Inferencing the GNN to score edges in the event graphs.
"""

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import torch

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.GNN.Models.interaction_gnn import InteractionGNN
from Pipelines.TrackML_Example.notebooks.build_gnn import GNNInferenceBuilder
from utils.convenience_utils import headline, delete_directory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("4_Run_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

def train(config_file="pipeline_config.yaml"):

    logging.info(headline( "Step 4: Scoring graph edges using GNN " ))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]

    logging.info(headline( "a) Loading trained model" ))

    if common_configs["clear_directories"]:
        delete_directory(gnn_configs["output_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionGNN.load_from_checkpoint(os.path.join(common_configs["artifact_directory"], "gnn", common_configs["experiment_name"]+".ckpt")).to(device)
    model.setup_data()

    logging.info(headline( "b) Running inferencing" ))
    graph_scorer = GNNInferenceBuilder(model)
    graph_scorer.infer()

if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file) 