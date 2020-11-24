import os

import faiss
import torch
from torch.utils.data import random_split
import scipy as sp
import numpy as np
import pandas as pd
import trackml.dataset

if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    device = 'cuda'
else:
    device = 'cpu'

def load_dataset(input_dir, num, pt_cut):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])
        loaded_events = [torch.load(event, map_location=torch.device('cpu')) for event in all_events[:num]]
        loaded_events = filter_hit_pt(loaded_events, pt_cut)
        return loaded_events
    else:
        return None

def split_datasets(input_dir, train_split, pt_cut = 0, seed = 1):
    '''
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    '''
    torch.manual_seed(seed)
    loaded_events = load_dataset(input_dir, sum(train_split), pt_cut)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events    

def fetch_pt(event):
    # Handle event in batched form
    event_file = event.event_file[0] if type(event.event_file) is list else event.event_file
    # Load the truth data from the event directory
    truth = trackml.dataset.load_event(
        event_file, parts=['truth'])[0]
    hid = event.hid[0] if type(event.hid) is list else event.hid
    merged_truth = pd.DataFrame(hid.cpu().numpy(), columns=["hit_id"]).merge(truth, on="hit_id")
    pt = np.sqrt(merged_truth.tpx**2 + merged_truth.tpy**2)
    
    return pt.to_numpy()

def fetch_type(event):
    # Handle event in batched form
    event_file = event.event_file[0] if type(event.event_file) is list else event.event_file
    # Load the truth data from the event directory
    truth, particles = trackml.dataset.load_event(
        event_file, parts=['truth', 'particles'])
    hid = event.hid[0] if type(event.hid) is list else event.hid
    merged_truth = truth.merge(particles, on="particle_id")
    p_type = pd.DataFrame(hid.cpu().numpy(), columns=["hit_id"]).merge(merged_truth, on="hit_id")
    p_type = p_type.particle_type.values
    
    return p_type
    
def filter_edge_pt(events, pt_cut=0):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]
        
    if pt_cut > 0:
        for event in events:
            pt = fetch_pt(event)
            edge_subset = pt[event.edge_index] > pt_cut
            combined_subset = edge_subset[0] & edge_subset[1]
            event.edge_index = event.edge_index[:, combined_subset]
            event.y = event.y[combined_subset]
            event.y_pid = event.y_pid[combined_subset]
    
    return events

def filter_hit_pt(events, pt_cut=0):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]
        
    if pt_cut > 0:
        for event in events:
            pt = fetch_pt(event)
            hit_subset = pt > pt_cut
            event.cell_data = event.cell_data[hit_subset]
            event.hid = event.hid[hit_subset]
            event.x = event.x[hit_subset]
            event.pid = event.pid[hit_subset]
            event.layers = event.layers[hit_subset]
            if 'pt' in event.__dict__.keys():
                event.pt = event.pt[hit_subset]
            if 'layerless_true_edges' in event.__dict__.keys():
                event.layerless_true_edges, remaining_edges = reset_edge_id(hit_subset, event.layerless_true_edges)
                
            if 'layerwise_true_edges' in event.__dict__.keys():
                event.layerwise_true_edges, remaining_edges = reset_edge_id(hit_subset, event.layerwise_true_edges)
                
            if 'weights' in event.__dict__.keys():
                    event.weights = event.weights[remaining_edges]
            
    return events

def reset_edge_id(subset, graph):
    subset_ind = np.where(subset)[0]
    filler = -np.ones((graph.max()+1,))
    filler[subset_ind] = np.arange(len(subset_ind))
    graph = torch.from_numpy(filler[graph]).long()
    exist_edges = (graph[0] >= 0) & (graph[1] >= 0)
    graph = graph[:, exist_edges]
    
    return graph, exist_edges
    
def graph_intersection(pred_graph, truth_graph, using_weights=False, weights_bidir=None):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix((np.ones(l1.shape[1]), l1), shape=(array_size, array_size)).tocsr()
    e_2 = sp.sparse.coo_matrix((np.ones(l2.shape[1]), l2), shape=(array_size, array_size)).tocsr()
    
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2)>0))
    
    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix((weights_list, l2), shape=(array_size, array_size)).tocsr()
        new_weights = weights_sparse[e_intersection.astype('bool')]
        new_weights = torch.from_numpy(np.array(new_weights)[0])
    
    e_intersection = e_intersection.tocoo()    
    new_pred_graph = torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col])).long().to(device)
    y = torch.from_numpy(e_intersection.data > 0).to(device)
    
    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y

def build_edges(spatial, r_max, k_max, res, return_indices=False):

    index_flat = faiss.IndexFlatL2(spatial.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    spatial_np = spatial.cpu().detach().numpy()
    gpu_index_flat.add(spatial_np)

    D, I = search_index_pytorch(gpu_index_flat, spatial, k_max)

    D, I = D[:,1:], I[:,1:]
    ind = torch.Tensor.repeat(torch.arange(I.shape[0]), (I.shape[1], 1), 1).T.to(device)

    edge_list = torch.stack([ind[D <= r_max**2], I[D <= r_max**2]])

    if return_indices:
        return edge_list, D, I, ind
    else:
        return edge_list

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def get_best_run(run_label, wandb_save_dir):
    for (root_dir, dirs, files) in os.walk(wandb_save_dir + "/wandb"):
        if run_label in dirs:
            run_root= root_dir

    best_run_base = os.path.join(run_root, run_label, "checkpoints")
    best_run = os.listdir(best_run_base)
    best_run_path = os.path.join(best_run_base, best_run[0])

    return best_run_path
