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
from Pipelines.Common_Tracking_Example.LightningModules.Embedding.utils import build_edges, graph_intersection

class ITk_EmbeddingInferenceBuilder(EmbeddingInferenceBuilder):
    
    def construct_downstream(self, batch, datatype):
        if "ci" in self.model.hparams["regime"]:
            input_data = torch.cat(
                [batch.cell_data[:, : self.model.hparams["cell_channels"]], batch.x],
                axis=-1,
            )
            input_data[input_data != input_data] = 0
            spatial = self.model(input_data)
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0
            spatial = self.model(input_data)

        batch = self.select_data(batch)
        # Make truth bidirectional
        # e_bidir = torch.cat(
        #     [
        #         batch.modulewise_true_edges,
        #         torch.stack(
        #             [batch.modulewise_true_edges[1], batch.modulewise_true_edges[0]],
        #             axis=1,
        #         ).T,
        #     ],
        #     axis=-1,
        # )

        # Build the radius graph with radius < r_test
        # e_spatial = build_edges(
        #     spatial,
        #     spatial,
        #     indices=None,
        #     r_max=self.model.hparams["r_test"],
        #     k_max=self.knn_max,
        # )  # This step should remove reliance on r_val, and instead compute an r_build based on the EXACT r required to reach target eff/pur
        y_cluster, e_spatial, e_bidir = self.get_performance(
            batch=batch, r_max=self.radius, k_max=self.knn_max
        )

        # Arbitrary ordering to remove half of the duplicate edges
        R_dist = torch.sqrt(batch.x[:, 0] ** 2 + batch.x[:, 2] ** 2)
        e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        # Re-introduce random direction, to avoid training bias
        random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
        e_spatial[0, random_flip], e_spatial[1, random_flip] = (
            e_spatial[1, random_flip],
            e_spatial[0, random_flip],
        )

        batch.edge_index = e_spatial
        batch.y = y_cluster

        self.save_downstream(batch, datatype)


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
            model.hparams['regime'] = metric_learning_configs['regime']
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
    graph_builder = ITk_EmbeddingInferenceBuilder(model, split=split, overwrite=True, knn_max=1000)
    graph_builder.build()

    return graph_builder



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    version = args.load_ckpt

    gb = train(config_file, version) 