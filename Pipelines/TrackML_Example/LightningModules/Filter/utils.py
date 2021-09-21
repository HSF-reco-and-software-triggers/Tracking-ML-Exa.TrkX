import sys
import os
import logging

import torch
import scipy as sp
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(input_dir, num, pt_background_cut, pt_signal_cut, true_edges, noise):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])
        loaded_events = []
        for event in all_events[:num]:
            try:
                loaded_event = torch.load(event, map_location=torch.device("cpu"))
                loaded_events.append(loaded_event)
                logging.info("Loaded event: {}".format(loaded_event.event_file))
            except:
                logging.info("Corrupted event file: {}".format(event))
        loaded_events = select_data(loaded_events, pt_background_cut, pt_signal_cut, true_edges, noise)
        return loaded_events
    else:
        return None

def get_edge_subset(edges, mask_where, inverse_mask):
    
    included_edges_mask = np.isin(edges, mask_where).all(0)    
    included_edges = edges[:, included_edges_mask]
    included_edges = inverse_mask[included_edges]
    
    return included_edges, included_edges_mask

def select_data(events, pt_background_cut, pt_signal_cut, true_edges, noise):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if (pt_background_cut > 0) or not noise:
        for event in events:
            
            pt_mask = (event.pt > pt_background_cut) & (event.pid == event.pid)
            pt_where = torch.where(pt_mask)[0]
            
            inverse_mask = torch.zeros(pt_where.max()+1).long()
            inverse_mask[pt_where] = torch.arange(len(pt_where))
            
            edge_mask = None    
            event[true_edges], edge_mask = get_edge_subset(event[true_edges], pt_where, inverse_mask)
                        
            if "weights" in event.__dict__.keys():
                if event.weights.shape[0] == event[true_edges].shape[1]:
                    event.weights = event.weights[edge_mask]
            
            event.edge_index, _ = get_edge_subset(event.edge_index, pt_where, inverse_mask)
            
            node_features = ["cell_data", "x", "hid", "pid", "pt", "layers"]
            for feature in node_features:
                if feature in event.__dict__.keys():
                    event[feature] = event[feature][pt_mask]
        
    # Define the signal edges
    for event in events:
        if pt_signal_cut > 0:
            edge_subset = (event.pt[event[true_edges]] > pt_signal_cut).all(0)
            event.signal_true_edges = event[true_edges][:, edge_subset]
        else:
            event.signal_true_edges = event[true_edges]
    
    return events

def graph_intersection(pred_graph, truth_graph):
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2) > 0)).tocoo()

    new_pred_graph = (
        torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col]))
        .long()
        .to(device)
    )
    y = e_intersection.data > 0

    return new_pred_graph, y
