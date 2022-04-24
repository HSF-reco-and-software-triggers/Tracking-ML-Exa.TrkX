import os

import numpy as np
import pandas as pd
import torch

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from functools import partial
import random

def rescale_features(merged_coords):
    feature_scale = [1000, np.pi, 1000]
    merged_coords["r"] = np.sqrt(merged_coords["x"] ** 2 + merged_coords["y"] ** 2) / feature_scale[0]
    merged_coords["phi"] = np.arctan2(merged_coords["y"], merged_coords["x"]) / feature_scale[1]
    merged_coords["z"] = merged_coords["z"] / feature_scale[2]
    merged_coords["cluster_r_1"] = np.sqrt(merged_coords["cluster_x_1"] ** 2 + merged_coords["cluster_y_1"] ** 2) / feature_scale[0]
    merged_coords["cluster_phi_1"] = np.arctan2(merged_coords["cluster_y_1"], merged_coords["cluster_x_1"]) / feature_scale[1]
    merged_coords["cluster_z_1"] = merged_coords["cluster_z_1"] / feature_scale[2]
    merged_coords["cluster_r_2"] = np.sqrt(merged_coords["cluster_x_2"] ** 2 + merged_coords["cluster_y_2"] ** 2) / feature_scale[0]
    merged_coords["cluster_phi_2"] = np.arctan2(merged_coords["cluster_y_2"], merged_coords["cluster_x_2"]) / feature_scale[1]
    merged_coords["cluster_z_2"] = merged_coords["cluster_z_2"] / feature_scale[2]

    return merged_coords

def get_truth(event):

    csv_event_file = event.event_file
    truth = pd.read_csv(csv_event_file + "-truth.csv").drop_duplicates(subset=["hit_id"], keep="first").sort_values("hit_id")

    return truth
    
def merge_truth(event, truth):
    hid_df = pd.DataFrame({"hit_id": event.hid})
    merged_coords = hid_df.merge(truth, on="hit_id", how="inner")[["barrel_endcap", "hardware", "x", "y", "z", "cluster_x_1", "cluster_y_1", "cluster_z_1", "cluster_x_2", "cluster_y_2", "cluster_z_2"]]
    
    return merged_coords
    
def assign_volume(merged_coords):

    volume_dict = {"PIXEL": {0: 0, -2: 1, 2: 1}, "STRIP": {0: 2, -2: 3, 2: 3}}

    # Apply volume dict to get volume_id in merged_coords
    merged_coords["volume_id"] = merged_coords.apply(lambda row: volume_dict[row["hardware"]][row["barrel_endcap"]], axis=1)

    return merged_coords

def build_new_data(event, merged_coords):
    new_x = torch.from_numpy(merged_coords[["r", "phi", "z", "cluster_r_1", "cluster_phi_1", "cluster_z_1", "cluster_r_2", "cluster_phi_2", "cluster_z_2"]].to_numpy())
    assert (event.x == new_x[:, :3].float()).all(), "x is not the same"
    event.x = new_x
    event.volume_id = torch.from_numpy(merged_coords["volume_id"].to_numpy())

    return event

def process_event(event_file, save_dir):
    event_name = os.path.split(event_file)[-1]
    save_file = os.path.join(save_dir, event_name)

    if os.path.exists(save_file):
        return
        
    event = torch.load(event_file, map_location="cpu")
    truth = get_truth(event)
    merged_coords = merge_truth(event, truth)
    merged_coords = rescale_features(merged_coords)
    merged_coords = assign_volume(merged_coords)
    event = build_new_data(event, merged_coords)
    
    torch.save(event, save_file)


# Function to read all graphs from a directory, apply hetero functions, and save back to disc
def build_hetero_graphs(base_dir, new_save_path, max_events=1000, use_process_map=False):
    
    # Read all graphs from the base_dir
    subdirs = ["train", "val", "test"]

    for subdir in subdirs:
        # Make save dir if it doesn't exist
        save_dir = os.path.join(new_save_path, subdir)
        partial_process_event = partial(process_event, save_dir=save_dir)
        os.makedirs(save_dir, exist_ok=True)      
        event_dirs = os.listdir(os.path.join(base_dir, subdir))[:max_events]
        event_files = [os.path.join(base_dir, subdir, event_dir) for event_dir in event_dirs]
        random.shuffle(event_files)

        if use_process_map:
            process_map(partial_process_event, event_files, max_workers=8)
        else:
            for event_file in tqdm(event_files):
                partial_process_event(event_file)            
