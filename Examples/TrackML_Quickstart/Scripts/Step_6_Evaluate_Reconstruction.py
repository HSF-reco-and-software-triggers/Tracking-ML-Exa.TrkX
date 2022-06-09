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
from tqdm import tqdm
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

def get_matching_df(reconstruction_df, min_track_length=1, min_particle_length=1):
    # Get track lengths
    candidate_lengths = reconstruction_df.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # Get true track lengths
    particle_lengths = reconstruction_df.drop_duplicates(subset=['hit_id']).particle_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"particle_id", "particle_id": "n_true_hits"})

    spacepoint_matching = reconstruction_df.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})
    spacepoint_matching = spacepoint_matching.merge(candidate_lengths, on=['track_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particle_lengths, on=['particle_id'], how='left')

    # Filter out tracks with too few shared spacepoints
    spacepoint_matching["is_matchable"] = spacepoint_matching.n_reco_hits >= min_track_length
    spacepoint_matching["is_reconstructable"] = spacepoint_matching.n_true_hits >= min_particle_length

    return spacepoint_matching

def calculate_matching_fraction(spacepoint_matching_df):
    spacepoint_matching_df = spacepoint_matching_df.assign(
        purity_reco=np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_reco_hits))
    spacepoint_matching_df = spacepoint_matching_df.assign(
        eff_true = np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_true_hits))

    return spacepoint_matching_df

def evaluate_labelled_graph(graph_file, matching_fraction=0.5, matching_style="ATLAS", min_track_length=1, min_particle_length=1):

    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must be >= 0.5")

    if matching_fraction == 0.5:
        # Add a tiny bit of noise to the matching fraction to avoid double-matched tracks
        matching_fraction += 0.00001

    # Load the labelled graphs as reconstructed dataframes
    reconstruction_df = load_reconstruction_df(graph_file)

    # Get matching dataframe
    matching_df = get_matching_df(reconstruction_df, min_track_length=min_track_length, min_particle_length=min_particle_length) 

    # calculate matching fraction
    matching_df = calculate_matching_fraction(matching_df)

    # Run matching depending on the matching style
    if matching_style == "ATLAS":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = matching_df.purity_reco >= matching_fraction
    elif matching_style == "one_way":
        matching_df["is_matched"] = matching_df.purity_reco >= matching_fraction
        matching_df["is_reconstructed"] = matching_df.eff_true >= matching_fraction
    elif matching_style == "two_way":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (matching_df.purity_reco >= matching_fraction) & (matching_df.eff_true >= matching_fraction)

    return matching_df

def evaluate(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 6: Evaluating the track reconstruction performance"))

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

    # evaluate_partial_fn = partial(evaluate_labelled_graph, matching_fraction=evaluation_configs["matching_fraction"])
    # evaluated_events = process_map(evaluate_partial_fn, all_graph_files, n_workers=evaluation_configs["n_workers"])

    evaluated_events = []
    for graph_file in tqdm(all_graph_files):
        evaluated_events.append(evaluate_labelled_graph(graph_file, 
                                matching_fraction=evaluation_configs["matching_fraction"], 
                                matching_style=evaluation_configs["matching_style"],
                                min_track_length=evaluation_configs["min_track_length"],
                                min_particle_length=evaluation_configs["min_particle_length"]))

    n_reconstructed_particles, n_particles, n_matched_tracks, n_tracks, n_dup_reconstructed_particles = 0, 0, 0, 0, 0
    for event in evaluated_events:
        n_particles += event[event["is_reconstructable"]].particle_id.nunique()
        reconstructed_particles = event[event["is_reconstructable"] & event["is_reconstructed"]]
        n_reconstructed_particles += reconstructed_particles.particle_id.nunique()
        n_tracks += event[event["is_matchable"]].track_id.nunique()
        n_matched_tracks += event[event["is_matchable"] & event["is_matched"]].track_id.nunique()
        n_dup_reconstructed_particles += reconstructed_particles[reconstructed_particles.particle_id.duplicated()].particle_id.nunique()

    # Plot the results across pT and eta
    logging.info(headline("b) Plotting the results"))
    eff = n_reconstructed_particles / n_particles
    fake_rate = 1 - (n_matched_tracks / n_tracks)
    # dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles
    dup_rate = n_dup_reconstructed_particles / n_matched_tracks
    
    logging.info(f"Efficiency: {eff:.3f}")
    logging.info(f"Fake rate: {fake_rate:.3f}")
    logging.info(f"Duplication rate: {dup_rate:.3f}")

    # TODO: Plot the results
    



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    evaluate(config_file) 