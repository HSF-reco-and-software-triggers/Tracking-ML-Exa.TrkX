"""
This script runs step 6 of the TrackML Quickstart example: Evaluating the track reconstruction performance.
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("5_Build_Track_Candidates.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

def evaluate_labelled_graph(graph_file, matching_fraction=0.5):

    # 1. Load the labelled graphs as reconstructed dataframes
    graph = torch.load(graph_file, map_location="cpu")
    reconstruction_df = pd.DataFrame({"hit_id": graph.hid, "track_id": graph.labels, "particle_id": graph.pid})

    # Get track lengths
    candidate_lengths = reconstruction_df.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # Get true track lengths
    particle_lengths = reconstruction_df.groupby("particle_id").track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_true_hits"})

    # Get shared hits
    spacepoint_matching = reconstruction_df.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})
    spacepoint_matching = spacepoint_matching.merge(candidate_lengths, on=['track_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particle_lengths, on=['particle_id'], how='left')


    # calculate matching fraction
    spacepoint_matching = spacepoint_matching.assign(
        purity_reco=np.true_divide(spacepoint_matching.n_shared, spacepoint_matching.n_reco_hits))
    spacepoint_matching = spacepoint_matching.assign(
        eff_true = np.true_divide(spacepoint_matching.n_shared, spacepoint_matching.n_true_hits))

    # print(spacepoint_matching)

    # select the best match
    spacepoint_matching['purity_reco_max'] = spacepoint_matching.groupby(
        "track_id")['purity_reco'].transform(max)
    spacepoint_matching['eff_true_max'] = spacepoint_matching.groupby(
        "track_id")['eff_true'].transform(max)

    # Apply matching criteria
    matched_reco_tracks = spacepoint_matching[
        (spacepoint_matching.purity_reco > frac_reco_matched)
    ]

    matched_true_particles = spacepoint_matching[
        (spacepoint_matching.eff_true_max > frac_truth_matched) \
        & (spacepoint_matching.eff_true == spacepoint_matching.eff_true_max)]

    # Double matching: 
    combined_match = matched_true_particles.merge(
        matched_reco_tracks, on=['track_id', 'particle_id'], how='inner')

    # Then make a particles and reconstructed tracks dataframe with matching attributes
    # particles["is_matched"] = particles.particle_id.isin(combined_match.particle_id).values
    # reconstructed["is_matched"] = reconstructed.track_id.isin(matched_reco_tracks.track_id).values

    return particles, reconstructed

def evaluate(config_file="pipeline_config.yaml"):

    logging.info(["-"]*20 + " Step 6: Evaluating the track reconstruction performance " + ["-"]*20)

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)

    common_configs = all_configs["common_configs"]
    track_building_configs = all_configs["track_building_configs"]
    evaluation_configs = all_configs["evaluation_configs"]

    

    logging.info(["-"]*20 + "a) Loading labelled graphs" + ["-"]*20)

    input_dir = track_building_configs["output_dir"]
    output_dir = evaluation_configs["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_graph_files = os.listdir(input_dir)
    all_graph_files = [os.path.join(input_dir, graph) for graph in all_graph_files]

    evaluate_partial_fn = partial(evaluate_labelled_graph, matching_fraction=evaluation_configs["matching_fraction"])
    evaluated_particles = process_map(evaluate_partial_fn, all_graph_files, n_workers=evaluation_configs["n_workers"])

    # Plot the results across pT and eta


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    evaluate(config_file) 