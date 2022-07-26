"""
This script runs step 2 of the TrackML Quickstart example: Inferencing the metric learning to construct graphs.
"""

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import torch
import numpy as np

sys.path.append("../..")
from Pipelines.TrackML_Example_Dev.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
# from Pipelines.Common_Tracking_Example.LightningModules.Embedding.Models.layerless_embedding import 
from utils import headline
from Pipelines.TrackML_Example_Dev.notebooks.build_embedding import EmbeddingInferenceBuilder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("2_Run_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    add_arg("--load_ckpt", required=False)
    return parser.parse_args()


def train(config_file="pipeline_config.yaml", load_ckpt=None):

    logging.info(headline("Step 2: Constructing graphs from metric learning model "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(headline("a) Loading trained model" ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_ckpt = os.path.join(common_configs["artifact_directory"], "metric_learning", common_configs["experiment_name"]+".ckpt")
    if load_ckpt is not None:
        logging.info(f"Loading checkpoint from {load_ckpt}")
        try:
            model = LayerlessEmbedding.load_from_checkpoint(load_ckpt).to(device)
        except Exception as e:
            logging.warning(f'Unable to load checkpoint due to this exception:\n{e}\nRestoring from default checkpoint at {default_ckpt}')
            model = LayerlessEmbedding.load_from_checkpoint(default_ckpt).to(device)
    else:
        logging.info(f"Loading checkpoint from {default_ckpt}")
        model = LayerlessEmbedding.load_from_checkpoint(default_ckpt).to(device)

    logging.info(headline("b) Running inferencing"))
    split = (np.array(metric_learning_configs['train_split']) / np.sum(metric_learning_configs['train_split']) * metric_learning_configs['n_inference_events']).astype(np.int16)
    split[0] = metric_learning_configs['n_inference_events'] - np.sum(split[1:])
    model.hparams['n_events'] = metric_learning_configs['n_inference_events']
    graph_builder = EmbeddingInferenceBuilder(model, split=split, overwrite=True, knn_max=1000, radius=metric_learning_configs["r_test"])
    graph_builder.build()

    return graph_builder



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    version = args.load_ckpt

    gb = train(config_file, version) 