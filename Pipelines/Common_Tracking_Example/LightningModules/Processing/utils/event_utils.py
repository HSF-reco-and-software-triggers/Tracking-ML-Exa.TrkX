"""Utilities for processing the overall event.

The module contains useful functions for handling data at the event level. More fine-grained utilities are 
reserved for `detector_utils` and `cell_utils`.
    
Todo:
    * Pull module IDs out into a csv file for readability """

# System
import os
import argparse
import logging
import multiprocessing as mp
from functools import partial

# Externals
import yaml
import numpy as np
import pandas as pd
import trackml.dataset

import torch
import itertools

# from .data_utils import Data
from torch_geometric.data import Data


def get_cell_information(hits, cell_features):
    cell_data = hits[cell_features]
    pixel_bool = (hits["hardware"] == "PIXEL").astype(int)

    cell_data = cell_data.assign(pixel=pixel_bool)

    return cell_data


def get_layerwise_edges(hits):

    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    hits.particle_id[hits.particle_id == 0] = np.nan
    hit_list = (
        hits.groupby(["particle_id", "layer"], sort=False)["index"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))
    true_edges = np.array(true_edges).T

    return true_edges, hits


def remap_edges(true_edges, hits):

    unique_hid = np.unique(hits.hit_id)
    hid_mapping = np.zeros(unique_hid.max() + 1).astype(int)
    hid_mapping[unique_hid] = np.arange(len(unique_hid))

    hits = hits.drop_duplicates(subset="hit_id").sort_values("hit_id")
    assert (
        hits.hit_id == unique_hid
    ).all(), "If hit IDs are not sequential, this will mess up graph structure!"

    true_edges = hid_mapping[true_edges]

    return true_edges, hits


def get_modulewise_edges(hits, module_ID = None):

    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]

    # Sort by increasing distance from production
    signal = signal.assign(
        R=np.sqrt(
            (signal.x - signal.vx) ** 2
            + (signal.y - signal.vy) ** 2
            + (signal.z - signal.vz) ** 2
        )
    )
    signal = signal.sort_values("R").reset_index(drop=False)

    # Group by particle ID
    if module_ID is None:
        module_ID = ["barrel_endcap", "hardware", "layer_disk", "eta_module", "phi_module"]
    signal_list = (
        signal.groupby(
            ["particle_id"] + module_ID,
            sort=False,
        )["hit_id"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))

    true_edges = np.array(true_edges).T

    # Remap
    true_edges, hits = remap_edges(true_edges, hits)

    return true_edges, hits


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def select_hits(hits, particles, pt_min=0, endcaps=True, noise=True):

    particles = particles[(particles.pt > pt_min)]
    particles = particles.assign(primary=(particles.barcode < 200000).astype(int))

    if not endcaps:
        hits = hits[hits["barrel_endcap"] == 0]

    if noise:
        hits = hits.merge(
            particles[["particle_id", "pt", "vx", "vy", "vz", "primary"]],
            on="particle_id",
            how="left",
        )
    else:
        hits = hits.merge(
            particles[["particle_id", "pt", "vx", "vy", "vz", "primary"]],
            on="particle_id",
        )

    hits["nhits"] = hits.groupby("particle_id")["particle_id"].transform("count")
    hits.loc[hits.particle_id == 0, "nhits"] = -1

    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    eta = calc_eta(r, hits.z)
    # Select the data columns we need
    hits = hits.assign(r=r, phi=phi, eta=eta)

    return hits

def clean_duplicates(hits):

    noise_hits = hits[hits.particle_id == 0].drop_duplicates(subset="hit_id")
    signal_hits = hits[hits.particle_id != 0]

    non_duplicate_noise_hits = noise_hits[~noise_hits.isin(signal_hits.hit_id)]
    hits = pd.concat([signal_hits, non_duplicate_noise_hits], ignore_index=True)

    return hits

def build_event(
    event_file,
    pt_min,
    feature_scale,
    cell_features,
    cell_information=True,
    endcaps=False,
    modulewise=True,
    layerwise=False,
    noise=False,
):
    # Get true edge list using the ordering by R' = distance from production vertex of each particle
    particles = pd.read_csv(event_file + "-particles.csv")
    hits = pd.read_csv(event_file + "-truth.csv")

    hits = select_hits(
        hits, particles, pt_min=pt_min, endcaps=endcaps, noise=noise
    ).assign(evtid=int(event_file[-9:]))

    hits = clean_duplicates(hits)
    
    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None

    if modulewise:
        modulewise_true_edges, hits = get_modulewise_edges(hits)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_file, modulewise_true_edges.shape
            )
        )

    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_file, layerwise_true_edges.shape
            )
        )

    # Make a unique module ID and attach to hits
    module_lookup = pd.read_pickle(os.path.join(os.path.split(os.path.split(event_file)[0])[0], "module_lookup"))
    hits = hits.merge(module_lookup, on=["hardware", "layer_disk", "eta_module", "phi_module", "barrel_endcap"], how="left")
    module_id = hits.module_index.to_numpy()
    
    if cell_information:
        logging.info("Getting cell info")
        cell_data = get_cell_information(hits, cell_features)

    logging.info("Weights constructed")

    return (
        hits[["r", "phi", "z"]].to_numpy() / feature_scale,
        cell_data.to_numpy(),
        hits.particle_id.to_numpy(),
        modulewise_true_edges,
        layerwise_true_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        hits.primary.to_numpy(),
        hits.nhits.to_numpy(),
        module_id
    )


def prepare_event(
    event_file,
    cell_features,
    progressbar=None,
    output_dir=None,
    pt_min=0,
    endcaps=False,
    modulewise=True,
    layerwise=False,
    noise=False,
    cell_information=True,
    overwrite=False,
    **kwargs
):

    try:
        evtid = int(event_file[-9:])
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))
            feature_scale = [1000, np.pi, 1000]

            (
                X,
                cell_data,
                pid,
                modulewise_true_edges,
                layerwise_true_edges,
                hid,
                pt,
                primary,
                nhits,
                module_id
            ) = build_event(
                event_file,
                pt_min,
                feature_scale,
                cell_features,
                cell_information=cell_information,
                endcaps=endcaps,
                modulewise=modulewise,
                layerwise=layerwise,
                noise=noise,
            )

            hit_data = Data(
                x=torch.from_numpy(X).float(),
                cell_data=torch.from_numpy(cell_data).float(),
                pid=torch.from_numpy(pid),
                event_file=event_file,
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
                primary=torch.from_numpy(primary),
                nhits=torch.from_numpy(nhits),
                modules=torch.from_numpy(module_id)
            )
            if modulewise_true_edges is not None:
                hit_data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)
            if layerwise_true_edges is not None:
                hit_data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)

            with open(filename, "wb") as pickle_file:
                torch.save(hit_data, pickle_file)

        else:
            logging.info("{} already exists".format(evtid))
    except Exception as inst:
        print("Exception with file:", event_file, "Exception:", inst)
