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
from torch_geometric.data import Data

from itertools import permutations
import itertools

# Locals
from .cell_utils import get_one_event

def get_cell_information(data, cell_features, detector_orig, detector_proc, endcaps, noise):

    event_file = data.event_file
    evtid = event_file[-4:]
    
    angles = get_one_event(event_file,
                  detector_orig,
                  detector_proc)
    logging.info("Angles: {}".format(angles))
    hid = pd.DataFrame(data.hid.numpy(), columns = ["hit_id"])
    cell_data = torch.from_numpy((hid.merge(angles, on="hit_id")[cell_features]).to_numpy()).float()
    logging.info("DF merged")
    data.cell_data = cell_data

    return data

def select_hits(hits, truth, particles, pt_min=0, endcaps=False, noise=False):
    # Barrel volume and layer ids
    if endcaps:
        vlids = [(7, 2), (7, 4), (7, 6), (7, 8), (7, 10), (7, 12), (7, 14), (8, 2), (8, 4), (8, 6), (8, 8), (9, 2), (9, 4), (9, 6), (9, 8), (9, 10), (9, 12), (9, 14), (12, 2), (12, 4), (12, 6), (12, 8), (12, 10), (12, 12), (13, 2), (13, 4), (13, 6), (13, 8), (14, 2), (14, 4), (14, 6), (14, 8), (14, 10), (14, 12), (16, 2), (16, 4), (16, 6), (16, 8), (16, 10), (16, 12), (17, 2), (17, 4), (18, 2), (18, 4), (18, 6), (18, 8), (18, 10), (18, 12)]
    else:
        vlids = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    if noise is False:
        # Calculate particle transverse momentum
        pt = np.sqrt(particles.px**2 + particles.py**2)
        # Applies pt cut, removes noise hits
        particles = particles[pt > pt_min]
        truth = (truth[['hit_id', 'particle_id', 'tpx', 'tpy', 'weight']]
                 .merge(particles[['particle_id', 'vx', 'vy', 'vz']], on='particle_id'))
        truth = truth.assign(pt = np.sqrt(truth.tpx**2 + truth.tpy**2))
    else:
        # Calculate particle transverse momentum
        pt = np.sqrt(truth.tpx**2 + truth.tpy**2)
        # Applies pt cut
        truth = truth[pt > pt_min]
        truth.loc[truth['particle_id'] == 0,'particle_id'] = float('NaN')
        truth = truth.assign(pt = pt)
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'x', 'y', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id', 'vx', 'vy', 'vz', 'pt', 'weight']], on='hit_id'))
    # (DON'T) Remove duplicate hits
#     hits = hits.loc[
#         hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
#     ]
    return hits

def build_event(event_file, pt_min, feature_scale, adjacent=True, endcaps=False, layerless=True, layerwise=True, noise=False):
    # Get true edge list using the ordering by R' = distance from production vertex of each particle
    hits, particles, truth = trackml.dataset.load_event(
        event_file, parts=['hits', 'particles', 'truth'])
    hits = select_hits(hits, truth, particles, pt_min=pt_min, endcaps=endcaps, noise=noise).assign(evtid=int(event_file[-9:]))
    layers = hits.layer.to_numpy()

    # Handle which truth graph(s) are being produced
    layerless_true_edges, layerwise_true_edges = None, None

    if layerless:
        hits = hits.assign(R=np.sqrt((hits.x - hits.vx)**2 + (hits.y - hits.vy)**2 + (hits.z - hits.vz)**2))
        hits = hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)
        hit_list = hits.groupby(['particle_id', 'layer'], sort=False)['index'].agg(lambda x: list(x)).groupby(level=0).agg(lambda x: list(x))

        e = []
        for row in hit_list.values:
            for i, j in zip(row[0:-1], row[1:]):
                e.extend(list(itertools.product(i, j)))

        layerless_true_edges = np.array(e).T
        logging.info("Layerless truth graph built for {} with size {}".format(event_file, layerless_true_edges.shape))

    if layerwise:
        # Get true edge list using the ordering of layers
        records_array = hits.particle_id.to_numpy()
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        _, idx_start, _ = np.unique(sorted_records_array, return_counts=True,
                                return_index=True)
        # sets of indices
        res = np.split(idx_sort, idx_start[1:])
        layerwise_true_edges = np.concatenate([list(permutations(i, r=2)) for i in res if len(list(permutations(i, r=2))) > 0]).T
        if adjacent: layerwise_true_edges = layerwise_true_edges[:, (layers[layerwise_true_edges[1]] - layers[layerwise_true_edges[0]] == 1)]
        logging.info("Layerwise truth graph built for {} with size {}".format(event_file, layerwise_true_edges.shape))

    edge_weights = hits.weight.to_numpy()[layerless_true_edges] if layerless else hits.weight.to_numpy()[layerwise_true_edges]
    edge_weight_average = (edge_weights[0] + edge_weights[1])/2
    edge_weight_norm = edge_weight_average / edge_weight_average.mean()
    
    logging.info("Weights constructed")
    
    return hits[['r', 'phi', 'z']].to_numpy() / feature_scale, hits.particle_id.to_numpy(), layers, layerless_true_edges, layerwise_true_edges, hits['hit_id'].to_numpy(), hits.pt.to_numpy(), edge_weight_norm

def prepare_event(event_file, detector_orig, detector_proc, cell_features, progressbar=None, output_dir=None, pt_min=0, adjacent=True, endcaps=False, layerless=True, layerwise=True, noise=False, cell_information=True, overwrite=False, **kwargs):

    try:
        evtid = int(event_file[-9:])
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))
            feature_scale = [1000, np.pi, 1000]

            X, pid, layers, layerless_true_edges, layerwise_true_edges, hid, pt, weights = build_event(event_file, pt_min, feature_scale, adjacent=adjacent, endcaps=endcaps, layerless=layerless, layerwise=layerwise, noise=noise)

            data = Data(
                x = torch.from_numpy(X).float(), 
                pid = torch.from_numpy(pid), 
                layers=torch.from_numpy(layers), 
                event_file=event_file, 
                hid = torch.from_numpy(hid),
                pt = torch.from_numpy(pt),
                weights = torch.from_numpy(weights)
            )
            if layerless_true_edges is not None: data.layerless_true_edges = torch.from_numpy(layerless_true_edges)
            if layerwise_true_edges is not None: data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)
            logging.info("Getting cell info")
            if cell_information:
                data = get_cell_information(data, cell_features, detector_orig, detector_proc, endcaps, noise)

            with open(filename, 'wb') as pickle_file:
                torch.save(data, pickle_file)
                
    
        else:
            logging.info(evtid, "already exists")
    except:
        print("Exception with file:", event_file)