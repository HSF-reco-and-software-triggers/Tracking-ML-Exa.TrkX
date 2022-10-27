import os
import logging

import torch
from torch.utils.data import random_split
import scipy as sp
import numpy as np
from torch_geometric.nn import radius

from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
Ideally, we would be using FRNN and the GPU. But in the case of a user not having a GPU, or not having FRNN, we import FAISS as the 
nearest neighbor library
"""

import faiss
import faiss.contrib.torch_utils

try:
    import frnn

    using_faiss = False
except ImportError:
    using_faiss = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    using_faiss = True


def load_dataset(
    input_dir,
    num,
    pt_background_cut,
    pt_signal_cut,
    nhits,
    primary_only,
    true_edges,
    noise,
    eta_cut,
):
    if input_dir is None:
        return None
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])
    loaded_events = []
    for event in all_events[:num]:
        try:
            loaded_event = torch.load(event, map_location=torch.device("cpu"))
            loaded_events.append(loaded_event)
            logging.info(f"Loaded event: {loaded_event.event_file}")
        except Exception:
            logging.info(f"Corrupted event file: {event}")
    loaded_events = select_data(
        loaded_events,
        pt_background_cut,
        pt_signal_cut,
        nhits,
        primary_only,
        true_edges,
        noise,
        eta_cut,
    )
    return loaded_events


def split_datasets(
    input_dir,
    train_split,
    pt_background_cut=0,
    pt_signal_cut=0,
    nhits=0,
    primary_only=False,
    true_edges=None,
    noise=True,
    eta_cut=None,
    seed=None,
    **kwargs,
):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    loaded_events = load_dataset(
        input_dir,
        sum(train_split),
        pt_background_cut,
        pt_signal_cut,
        nhits,
        primary_only,
        true_edges,
        noise,
        eta_cut,
    )
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events


def get_edge_subset(edges, mask_where, inverse_mask):

    included_edges_mask = np.isin(edges, mask_where).all(0)
    included_edges = edges[:, included_edges_mask]
    included_edges = inverse_mask[included_edges]

    return included_edges, included_edges_mask


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def select_data(
    events,
    pt_background_cut,
    pt_signal_cut,
    nhits_min,
    primary_only,
    true_edges,
    noise,
    eta_cut,
):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if pt_background_cut > 0 or not noise:
        for event in events:
            event.eta = calc_eta(event.x[:, 0] * 1000, event.x[:, 2] * 1000)
            pt_mask = (
                (event.pt > pt_background_cut)
                & (event.pid == event.pid)
                & event.primary.bool()
                & (event.eta > -eta_cut)
                & (event.eta < eta_cut)
            )
            pt_where = torch.where(pt_mask)[0]

            inverse_mask = torch.zeros(pt_where.max() + 1).long()
            inverse_mask[pt_where] = torch.arange(len(pt_where))

            event[true_edges], edge_mask = get_edge_subset(
                event[true_edges], pt_where, inverse_mask
            )

            node_features = [
                "cell_data",
                "x",
                "hid",
                "pid",
                "pt",
                "nhits",
                "primary",
                "eta",
            ]
            for feature in node_features:
                event[feature] = event[feature][pt_mask]

    #             print(pt_mask.sum(), event[true_edges].shape)
    for event in events:
        #         print((event.pt[event[true_edges]] > pt_signal_cut).all(0).sum(),
        #              (event.nhits[event[true_edges]] >= nhits_min).all(0).sum(),
        #              (~(primary_only * event.primary[event[true_edges]].bool())).all(0).sum())

        edge_subset = (
            (event.pt[event[true_edges]] > pt_signal_cut).all(0)
            & (event.nhits[event[true_edges]] >= nhits_min).all(0)
            & (event.primary[event[true_edges]].bool().all(0) | (not primary_only))
        )

        event.signal_true_edges = event[true_edges][:, edge_subset]
    #         print(event.signal_true_edges.shape)

    return events


def reset_edge_id(subset, graph):
    subset_ind = np.where(subset)[0]
    filler = -np.ones((graph.max() + 1,))
    filler[subset_ind] = np.arange(len(subset_ind))
    graph = torch.from_numpy(filler[graph]).long()
    exist_edges = (graph[0] >= 0) & (graph[1] >= 0)
    graph = graph[:, exist_edges]

    return graph, exist_edges


def graph_intersection(input_pred_graph, input_truth_graph, return_y_pred=True, return_y_truth=False, return_pred_to_truth=False, return_truth_to_pred=False):
    """
    An updated version of the graph intersection function, which is around 25x faster than the
    Scipy implementation (on GPU). Takes a prediction graph and a truth graph.
    """
    
    unique_edges, inverse = torch.unique(torch.cat([input_pred_graph, input_truth_graph], dim=1), dim=1, sorted=False, return_inverse=True, return_counts=False)

    inverse_pred_map = torch.ones(unique_edges.shape[1], dtype=torch.long) * -1
    inverse_pred_map[inverse[:input_pred_graph.shape[1]]] = torch.arange(input_pred_graph.shape[1])
    
    inverse_truth_map = torch.ones(unique_edges.shape[1], dtype=torch.long) * -1
    inverse_truth_map[inverse[input_pred_graph.shape[1]:]] = torch.arange(input_truth_graph.shape[1])

    pred_to_truth = inverse_truth_map[inverse][:input_pred_graph.shape[1]]
    truth_to_pred = inverse_pred_map[inverse][input_pred_graph.shape[1]:]

    return_tensors = []

    if return_y_pred:
        y_pred = pred_to_truth >= 0
        return_tensors.append(y_pred)
    if return_y_truth:
        y_truth = truth_to_pred >= 0
        return_tensors.append(y_truth)
    if return_pred_to_truth:        
        return_tensors.append(pred_to_truth)
    if return_truth_to_pred:
        return_tensors.append(truth_to_pred)

    return return_tensors if len(return_tensors) > 1 else return_tensors[0]


def build_edges(
    query, database, indices=None, r_max=1.0, k_max=10, return_indices=False, backend="FRNN", self_loop=False
):

    if backend == "FRNN":
        dists, idxs, nn, grid = frnn.frnn_grid_points(
            points1=query.unsqueeze(0),
            points2=database.unsqueeze(0),
            lengths1=None,
            lengths2=None,
            K=k_max,
            r=r_max,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )      

        idxs = idxs.squeeze().int()
        ind = torch.Tensor.repeat(
        torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1
        ).T.int()
        positive_idxs = idxs >= 0
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()

    elif backend == "PYG":
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    if not self_loop:
        edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (edge_list, dists, idxs, ind) if return_indices else edge_list


def build_knn(spatial, k):

    if device == "cuda":
        res = faiss.StandardGpuResources()
        _, I = faiss.knn_gpu(res, spatial, spatial, k)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        _, I = index.search(spatial, k)

    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
    ).T
    edge_list = torch.stack([ind, I])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list

class CustomReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                threshold=1e-4, threshold_mode='rel', cooldown=0,
                min_lr=0, eps=1e-8, verbose=False, ignore_first_n_epochs=0):
        super(CustomReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)

        self.ignore_first_n_epochs = ignore_first_n_epochs

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.ignore_first_n_epochs > 0 and epoch <= self.ignore_first_n_epochs:
            return
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]