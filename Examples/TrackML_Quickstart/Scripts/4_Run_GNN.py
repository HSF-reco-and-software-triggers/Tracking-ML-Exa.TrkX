"""
This script runs step 4 of the TrackML Quickstart example: Inferencing the GNN to score edges in the event graphs.
"""

import sys
import os
import yaml
import argparse
import logging
import torch

sys.path.append("../../")
from Pipelines.TrackML_Example.LightningModules.GNN.Models.interaction_gnn import InteractionGNN
from Pipelines.TrackML_Example.notebooks.build_gnn import GNNInferenceBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("4_Run_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

def train(config_file="pipeline_config.yaml"):

    logging.info(["-"]*20 + " Step 4: Scoring graph edges using GNN " + ["-"]*20)

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]

    logging.info(["-"]*20 + "a) Loading trained model" + ["-"]*20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionGNN.load_from_checkpoint(os.path.join(common_configs["artifact_directory"], "gnn", common_configs["experiment_name"]+".ckpt")).to(device)
    model.setup()

    logging.info(["-"]*20 + "b) Running inferencing" + ["-"]*20)
    graph_scorer = GNNInferenceBuilder(model)
    graph_scorer.infer()

if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file) 