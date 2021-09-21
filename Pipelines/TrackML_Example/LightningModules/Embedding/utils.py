import os
import logging

import torch
from torch.utils.data import random_split
import scipy as sp
import numpy as np
import pandas as pd
import trackml.dataset

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


def split_datasets(input_dir, train_split, pt_background_cut=0, pt_signal_cut=0, true_edges=None, noise=False, seed=1):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    """
    torch.manual_seed(seed)
    loaded_events = load_dataset(input_dir, sum(train_split), pt_background_cut, pt_signal_cut, true_edges, noise)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events


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



def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list
        del l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)
    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y


def build_edges(query, database, indices, r_max, k_max, return_distances=False):
    
    if using_faiss:
        edge_list, dist_list = build_edges_faiss(query, database, r_max, k_max)
    else:
        edge_list, dist_list = build_edges_frnn(query, database, r_max, k_max)
        
    # Remove self-loops and correct indices if necessary
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_distances:
        return edge_list, dist_list
    else:
        return edge_list


def build_edges_frnn(query, database, r_max, k_max):
    
    dists, idxs, nn, grid = frnn.frnn_grid_points(points1=query.unsqueeze(0), points2=database.unsqueeze(0), lengths1=None, lengths2=None, K=k_max, r=r_max, grid=None, return_nn=False, return_sorted=True)
    
    idxs = idxs.squeeze()
    ind = torch.Tensor.repeat(torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1).T
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]])
    dist_list = dists.squeeze()[positive_idxs]
    
    return edge_list, dist_list

def build_edges_faiss(query, database, r_max, k_max):
    
    if device == "cuda":
        res = faiss.StandardGpuResources()
        D, I = faiss.knn_gpu(res, query, database, k_max)
    elif device == "cpu":
        index = faiss.IndexFlatL2(database.shape[1])
        index.add(database)
        D, I = index.search(query, k_max)

    ind = torch.Tensor.repeat(torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1).T
    edge_list = torch.stack([ind[D <= r_max**2], I[D <= r_max**2]])
    dist_list = D[D <= r_max**2]
    
    return edge_list, dist_list
    
    

def build_knn(spatial, k):

    if device == "cuda":
        res = faiss.StandardGpuResources()
        _, I = faiss.knn_gpu(res, spatial, spatial, k_max)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        _, I = index.search(spatial, k_max)

    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
    ).T
    edge_list = torch.stack([ind, I])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list


def get_best_run(run_label, wandb_save_dir):
    for (root_dir, dirs, files) in os.walk(wandb_save_dir + "/wandb"):
        if run_label in dirs:
            run_root = root_dir

    best_run_base = os.path.join(run_root, run_label, "checkpoints")
    best_run = os.listdir(best_run_base)
    best_run_path = os.path.join(best_run_base, best_run[0])

    return best_run_path


# -------------------------- Performance Evaluation -------------------


def embedding_model_evaluation(model, trainer, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(model, trainer, fixed_value, fom),
        x0=0.9,
        x1=1.2,
        xtol=0.001,
    )
    print("Seed solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return evaluate_set_metrics(sol.root, model, trainer), sol.root


def evaluate_set_root(r, model, trainer, goal=0.96, fom="eff"):
    eff, pur = evaluate_set_metrics(r, model, trainer)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def get_metrics(test_results, model):

    ps = [len(result["truth"]) for result in test_results]
    ts = [result["truth_graph"].shape[1] for result in test_results]
    tps = [result["truth"].sum() for result in test_results]

    efficiencies = [tp / t for (t, tp) in zip(ts, tps)]
    purities = [tp / p for (p, tp) in zip(ps, tps)]

    mean_efficiency = np.mean(efficiencies)
    mean_purity = np.mean(purities)

    return mean_efficiency, mean_purity


def evaluate_set_metrics(r_test, model, trainer):

    model.hparams.r_test = r_test
    test_results = trainer.test(ckpt_path=None)

    mean_efficiency, mean_purity = get_metrics(test_results, model)

    print(mean_purity, mean_efficiency)

    return mean_efficiency, mean_purity
