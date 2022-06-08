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
import pandas as pd
import scipy.sparse as sps

from tqdm.contrib.concurrent import process_map
from functools import partial
from utils import headline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("5_Build_Track_Candidates.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

def load_reconstruction_df(file):
    """Load the reconstructed tracks from a file."""
    graph = torch.load(file, map_location="cpu")
    reconstruction_df = pd.DataFrame({"hit_id": graph.hid, "track_id": graph.labels, "particle_id": graph.pid})
    return reconstruction_df

def load_particles_df(file):
    """Load the particles from a file."""
    graph = torch.load(file, map_location="cpu")

    # Get the particle dataframe
    particles_df = pd.DataFrame({"particle_id": graph.pid, "pt": graph.pt})

    # Reduce to only unique particle_ids
    particles_df = particles_df.drop_duplicates(subset=['particle_id'])

    return particles_df

def get_matching_df(reconstruction_df):
    # Get track lengths
    candidate_lengths = reconstruction_df.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # Get true track lengths
    particle_lengths = reconstruction_df.groupby("particle_id").track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_true_hits"})

    spacepoint_matching = reconstruction_df.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})
    spacepoint_matching = spacepoint_matching.merge(candidate_lengths, on=['track_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particle_lengths, on=['particle_id'], how='left')

    return spacepoint_matching

def calculate_matching_fraction(spacepoint_matching_df):
    spacepoint_matching_df = spacepoint_matching_df.assign(
        purity_reco=np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_reco_hits))
    spacepoint_matching_df = spacepoint_matching_df.assign(
        eff_true = np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_true_hits))

    return spacepoint_matching_df

def evaluate_labelled_graph(graph_file, matching_fraction=0.5):

    # Load the labelled graphs as reconstructed dataframes
    reconstruction_df = load_reconstruction_df(graph_file)
    particles_df = load_particles_df(graph_file)

    # Get matching dataframe
    spacepoint_matching_df = get_matching_df(reconstruction_df)  

    # calculate matching fraction
    spacepoint_matching_df = calculate_matching_fraction(spacepoint_matching_df)

    # select the best match
    spacepoint_matching_df['purity_reco_max'] = spacepoint_matching_df.groupby(
        "track_id")['purity_reco'].transform(max)
    spacepoint_matching_df['eff_true_max'] = spacepoint_matching_df.groupby(
        "track_id")['eff_true'].transform(max)

    # Apply matching criteria
    matched_reco_tracks = spacepoint_matching_df[
        (spacepoint_matching_df.purity_reco > matching_fraction)
    ]

    matched_true_particles = spacepoint_matching_df[
        (spacepoint_matching_df.eff_true_max > matching_fraction) \
        & (spacepoint_matching_df.eff_true == spacepoint_matching_df.eff_true_max)]

    # Double matching: 
    combined_match = matched_true_particles.merge(
        matched_reco_tracks, on=['track_id', 'particle_id'], how='inner')

    # Then make a particles and reconstructed tracks dataframe with matching attributes
    particles_df["is_matched"] = particles_df.particle_id.isin(combined_match.particle_id).values
    reconstruction_df["is_matched"] = reconstruction_df.track_id.isin(matched_reco_tracks.track_id).values

    return particles_df, reconstruction_df

def evaluate(config_file="pipeline_config.yaml"):

    logging.info(["-"]*20 + " Step 6: Evaluating the track reconstruction performance " + ["-"]*20)

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)

    common_configs = all_configs["common_configs"]
    track_building_configs = all_configs["track_building_configs"]
    evaluation_configs = all_configs["evaluation_configs"]

    logging.info(headline("a) Loading labelled graphs"))

    input_dir = track_building_configs["output_dir"]
    output_dir = evaluation_configs["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_graph_files = os.listdir(input_dir)
    all_graph_files = [os.path.join(input_dir, graph) for graph in all_graph_files]

    evaluate_partial_fn = partial(evaluate_labelled_graph, matching_fraction=evaluation_configs["matching_fraction"])
    evaluated_events = process_map(evaluate_partial_fn, all_graph_files, n_workers=evaluation_configs["n_workers"])

    # Check this logic!!
    n_true_tracks, n_reco_tracks, n_matched_particles, n_matched_tracks, n_duplicated_tracks, n_single_matched_particles = 0, 0, 0, 0, 0, 0
    for event in evaluated_events:
        particles_df, reconstruction_df = event
        n_true_tracks += len(particles_df)
        n_reco_tracks += len(reconstruction_df)
        n_matched_particles += len(particles_df[particles_df.is_matched])
        n_matched_tracks += len(reconstruction_df[reconstruction_df.is_matched])
        n_duplicated_tracks += len(reconstruction_df[reconstruction_df.is_matched]) - len(reconstruction_df[reconstruction_df.is_matched].drop_duplicates())
        n_single_matched_particles += len(particles_df[particles_df.is_matched]) - len(particles_df[particles_df.is_matched].drop_duplicates())

    # Plot the results across pT and eta
    logging.info(headline("b) Plotting the results"))
    eff = n_matched_tracks / n_reco_tracks
    purity = n_matched_particles / n_true_tracks
    fake_rate = 1 - purity
    dup_rate = n_duplicated_tracks / n_reco_tracks
    logging.info(f"Efficiency: {eff:.3f}")
    logging.info(f"Fake rate: {fake_rate:.3f}")
    logging.info(f"Duplication rate: {dup_rate:.3f}")

    # Plot the results
    



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    evaluate(config_file) 