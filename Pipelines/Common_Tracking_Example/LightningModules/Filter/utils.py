import sys
import os
import logging

import torch
import scipy as sp
import numpy as np

from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(input_dir, num):
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
        return loaded_events
    else:
        return None


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

class filter_dataset(Dataset):
    
    def __init__(self, dataset, hparams):
        
        # Setup here
        self.dataset = dataset
        self.hparams = hparams
        
    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        batch = self.dataset[idx]
        
        if "subset" in self.hparams["regime"]:
            subset_mask = np.isin(batch.edge_index, batch.signal_true_edges.unique()).any(0)
            batch.edge_index = batch.edge_index[:, subset_mask]
            batch.y = batch.y[subset_mask]
            
        if self.hparams["ratio"] != 0:
            num_true, num_false = batch.signal_true_edges.shape[1], (~batch.y.bool()).sum()
            
            # Select a subset of fake edges randomly
            start_index = torch.randint(len(batch.y) - 2*self.hparams["ratio"]*num_true, (1,))
            end_index = start_index + 2*self.hparams["ratio"]*num_true
            random_edges = batch.edge_index[:, start_index:end_index]
            combined_edges = torch.cat([batch.signal_true_edges, batch.signal_true_edges.flip(0), random_edges], dim=1)
            combined_y = torch.cat([torch.ones(2*batch.signal_true_edges.shape[1]), batch.y[start_index:end_index]])
            
            # Shuffle in true edges
            shuffle_indices = torch.randperm(combined_edges.shape[1])
            combined_edges = combined_edges[:, shuffle_indices]
            combined_y = combined_y[shuffle_indices]
            
            # Select a further subset in order to handle memory issues
            start_index = torch.randint(len(combined_y) - self.hparams["edges_per_batch"], (1,))
            end_index = start_index + self.hparams["edges_per_batch"]
            
            combined_edges = combined_edges[:, start_index:end_index]
            combined_y = combined_y[start_index:end_index]
            
        
        subbatch = {"x": batch.x, "cell_data": batch.cell_data, "edge_index": combined_edges, "y": combined_y}
        
        return subbatch