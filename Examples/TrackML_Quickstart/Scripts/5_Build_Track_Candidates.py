"""
This script runs step 5 of the TrackML Quickstart example: Labelling spacepoints based on the scored graph.
"""

import sys
import os
import yaml
import argparse
import logging
import torch
import numpy as np
import scipy.sparse as sps

from tqdm.contrib.concurrent import process_map
from functools import partial

sys.path.append("../../")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("5_Build_Track_Candidates.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def label_graph(graph, score_cut=0.8, save_dir="datasets/quickstart_track_building_processed"):

    edge_mask = graph.scores > score_cut

    row, col = graph.edge_index[:, edge_mask]
    edge_attr = np.ones(row.size(0))

    N = graph.x.size(0)
    sparse_edges = sps.coo_matrix((edge_attr, (row.numpy(), col.numpy())), (N, N))

    _, candidate_labels = sps.csgraph.connected_components(sparse_edges, directed=False, return_labels=True)  
    graph.labels = torch.from_numpy(candidate_labels).long()

    torch.save(graph, os.path.join(save_dir, graph.event_file[-4:]))


def train(config_file="pipeline_config.yaml"):

    logging.info(["-"]*20 + " Step 5: Building track candidates from the scored graph " + ["-"]*20)

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]
    track_building_configs = all_configs["track_building_configs"]

    logging.info(["-"]*20 + "a) Loading scored graphs" + ["-"]*20)

    all_graphs = []
    for subdir in ["train", "val", "test"]:
        subdir_graphs = os.listdir(os.path.join(gnn_configs["output_dir"], subdir))
        all_graphs += [torch.load(os.path.join(gnn_configs["output_dir"], subdir, graph), map_location="cpu") for graph in subdir_graphs]

    logging.info(["-"]*20 + "b) Labelling graph nodes" + ["-"]*20)

    score_cut = track_building_configs["score_cut"]
    save_dir = track_building_configs["output_dir"]
    
    label_graph_list_partial = partial(label_graph, score_cut=score_cut, save_dir=save_dir)
    all_graphs = process_map(label_graph_list_partial, all_graphs)



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file) 