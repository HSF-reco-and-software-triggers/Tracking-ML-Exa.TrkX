import os, sys
import logging

import torch.nn as nn
import torch
import pandas as pd
import numpy as np

# import cupy as cp
# import trackml.dataset

# ---------------------------- Dataset Processing -------------------------


def load_dataset(
    input_subdir="",
    num_events=10,
    pt_background_cut=0,
    pt_signal_cut=0,
    noise=False,
    triplets=False,
    **kwargs
):
    if input_subdir is not None:
        all_events = os.listdir(input_subdir)
        all_events = sorted([os.path.join(input_subdir, event) for event in all_events])
        print("Loading events")
        loaded_events = [
            torch.load(event, map_location=torch.device("cpu"))
            for event in all_events[:num_events]
        ]
        print("Events loaded!")
        loaded_events = process_data(
            loaded_events, pt_background_cut, pt_signal_cut, noise, triplets
        )
        print("Events processed!")
        return loaded_events
    else:
        return None


def process_data(events, pt_background_cut, pt_signal_cut, noise, triplets):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if pt_background_cut > 0:
        for i, event in enumerate(events):

            if triplets:  # Keep all event data for posterity!
                event = convert_triplet_graph(event)

            else:
                edge_mask = (event.pt[event.edge_index] > pt_background_cut).all(0)
                event.edge_index = event.edge_index[:, edge_mask]
                event.y = event.y[edge_mask]

                if "y_pid" in event.__dict__.keys():
                    event.y_pid = event.y_pid[edge_mask]

                if "weights" in event.__dict__.keys():
                    if event.weights.shape[0] == edge_mask.shape[0]:
                        event.weights = event.weights[edge_mask]

                if (
                    "signal_true_edges" in event.__dict__.keys()
                    and event.signal_true_edges is not None
                ):
                    signal_mask = (
                        event.pt[event.signal_true_edges] > pt_signal_cut
                    ).all(0)
                    event.signal_true_edges = event.signal_true_edges[:, signal_mask]

    return events


def convert_triplet_graph(event, edge_cut=0.5, directed=True):

    triplet_edges, triplet_y_truth, triplet_y_pid_truth = build_triplets(
        event, edge_cut, directed
    )

    triplet_x, triplet_cell_data = get_symmetric_values(
        event.x, event.edge_index
    ), get_symmetric_values(event.cell_data, event.edge_index)

    (
        event.doublet_edge_index,
        event.doublet_y,
        event.doublet_y_pid,
        event.doublet_x,
        event.doublet_cell_data,
    ) = (event.edge_index, event.y, event.y_pid, event.x, event.cell_data)
    event.edge_index, event.y, event.y_pid, event.x, event.cell_data = (
        triplet_edges,
        triplet_y_truth,
        triplet_y_pid_truth,
        triplet_x,
        triplet_cell_data,
    )

    return event


def build_triplets(graph, edge_cut=0.5, directed=True):

    undir_graph = torch.cat([graph.edge_index, graph.edge_index.flip(0)], dim=-1)

    # apply cut
    passing_edges = undir_graph[:, graph.scores.repeat(2) > edge_cut]
    passing_y_truth = graph.y[graph.scores > edge_cut]
    passing_y_pid_truth = graph.y_pid[graph.scores > edge_cut]

    # convert to cupy
    passing_edges_cp = cp.asarray(passing_edges).astype("float32")

    # make some utility objects
    num_edges = passing_edges.shape[1]
    e_ones = cp.array([1] * num_edges).astype("float32")
    e_arange = cp.arange(num_edges).astype("float32")
    e_max = passing_edges.max().item()

    # build sparse edge array
    passing_edges_csr_in = cp.sparse.coo_matrix(
        (e_ones, (passing_edges_cp[0], e_arange)), shape=(e_max + 1, num_edges)
    ).tocsr()
    passing_edges_csr_out = cp.sparse.coo_matrix(
        (e_ones, (passing_edges_cp[1], e_arange)), shape=(e_max + 1, num_edges)
    ).tocsr()

    # convert to triplets
    triplet_edges = passing_edges_csr_out.T * passing_edges_csr_in
    triplet_edges = triplet_edges.tocoo()

    # convert back to pytorch
    undirected_triplet_edges = torch.as_tensor(
        cp.stack([triplet_edges.row, triplet_edges.col]), device=device
    )

    # convert back to a single-direction edge list
    if directed:
        directed_map = torch.cat(
            [torch.arange(num_edges / 2), torch.arange(num_edges / 2)]
        ).int()
        directed_triplet_edges = directed_map[undirected_triplet_edges.long()].long()
        directed_triplet_edges = directed_triplet_edges[
            :, directed_triplet_edges[0] != directed_triplet_edges[1]
        ]  # Remove self-loops
        directed_triplet_edges = directed_triplet_edges[
            :, directed_triplet_edges[0] < directed_triplet_edges[1]
        ]  # Remove duplicate edges

        return (
            directed_triplet_edges,
            passing_y_truth[directed_triplet_edges].all(0),
            passing_y_pid_truth[directed_triplet_edges].all(0),
        )

    else:
        return (
            undirected_triplet_edges,
            passing_y_truth[undirected_triplet_edges].all(0),
            passing_y_pid_truth[undirected_triplet_edges].all(0),
        )


def get_symmetric_values(x, e):
    x_mean = (x[e[0]] + x[e[1]]) / 2
    x_diff = (x[e[0]] - x[e[1]]).abs()

    return torch.cat([x_mean, x_diff], dim=-1)


def purity_sample(truth, target_purity, regime):

    # Get true edges
    true_edges = torch.where(truth)[0]
    num_true = true_edges.shape[0]

    # Get fake edges
    fake_edges = torch.where(~truth)[0]

    # Sample fake edges
    num_fakes_to_sample = int(num_true * (1 - target_purity) / target_purity)
    fake_edges_sample = fake_edges[
        torch.randperm(len(fake_edges))[:num_fakes_to_sample]
    ]

    # Mix together
    combined_edges = torch.cat([true_edges, fake_edges_sample])
    combined_edges = combined_edges[torch.randperm(len(combined_edges))]

    edge_sample = batch.edge_index[:, combined_edges]
    truth_sample = batch.y_pid[combined_edges]
    return edge_sample, truth_sample


def random_edge_slice_v2(delta_phi, batch):
    """
    Same behaviour as v1, but avoids the expensive calls to np.isin and np.unique, using sparse operations on GPU
    """
    # 1. Select random phi
    random_phi = np.random.rand() * 2 - 1
    e = batch.e_radius.to("cpu").numpy()
    x = batch.x.to("cpu")

    # 2. Find edges within delta_phi of random_phi
    e_average = (x[e[0], 1] + x[e[1], 1]) / 2
    dif = abs(e_average - random_phi)
    subset_edges = ((dif < delta_phi) | ((2 - dif) < delta_phi)).numpy()

    # 3. Find connected edges to this subset
    e_ones = cp.array([1] * e_length).astype("Float32")
    subset_ones = cp.array([1] * subset_edges.sum()).astype("Float32")

    e_csr_in = cp.sparse.coo_matrix(
        (
            e_ones,
            (cp.array(e[0]).astype("Float32"), cp.arange(e_length).astype("Float32")),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    e_csr_out = cp.sparse.coo_matrix(
        (
            e_ones,
            (cp.array(e[0]).astype("Float32"), cp.arange(e_length).astype("Float32")),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    e_csr = e_csr_in + e_csr_out

    subset_csr_in = cp.sparse.coo_matrix(
        (
            subset_ones,
            (
                cp.array(e[0, subset_edges]).astype("Float32"),
                cp.arange(e_length)[subset_edges].astype("Float32"),
            ),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    subset_csr_out = cp.sparse.coo_matrix(
        (
            subset_ones,
            (
                cp.array(e[0, subset_edges]).astype("Float32"),
                cp.arange(e_length)[subset_edges].astype("Float32"),
            ),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    subset_csr = subset_csr_in + subset_csr_out

    summed = (subset_csr.T * e_csr).sum(axis=0)
    subset_edges_extended = (summed > 0)[0].get()

    return subset_edges, subset_edges_extended


def random_edge_slice(delta_phi, batch):
    # 1. Select random phi
    random_phi = np.random.rand() * 2 - 1
    e = batch.e_radius.to("cpu")
    x = batch.x.to("cpu")

    # 2. Find hits within delta_phi of random_phi
    dif = abs(x[:, 1] - random_phi)
    subset_hits = np.where((dif < delta_phi) | ((2 - dif) < delta_phi))[0]

    # 3. Filter edges with subset_hits
    subset_edges_ind = np.isin(e[0], subset_hits) | np.isin(e[1], subset_hits)

    subset_hits = np.unique(e[:, subset_edges_ind])
    subset_edges_extended = np.isin(e[0], subset_hits) | np.isin(e[1], subset_hits)
    nested_ind = np.isin(
        np.where(subset_edges_extended)[0], np.where(subset_edges_ind)[0]
    )

    return subset_edges_ind, subset_edges_extended, nested_ind


def hard_random_edge_slice(delta_phi, batch):

    # 1. Select random phi
    random_phi = np.random.rand() * 2 - 1
    e = batch.e_radius.to("cpu")
    x = batch.x.to("cpu")

    e_average = (x[e[0], 1] + x[e[1], 1]) / 2
    dif = abs(e_average - random_phi)
    subset_edges_ind = ((dif < delta_phi) | ((2 - dif) < delta_phi)).numpy()

    return subset_edges_ind


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def hard_eta_edge_slice(delta_eta, batch):

    e = batch.e_radius.to("cpu")
    x = batch.x.to("cpu")

    etas = calc_eta(x[:, 0], x[:, 2])
    random_eta = (np.random.rand() - 0.5) * 2 * (etas.max() - delta_eta)

    e_average = (etas[e[0]] + etas[e[1]]) / 2
    dif = abs(e_average - random_eta)
    subset_edges_ind = ((dif < delta_eta)).numpy()

    return subset_edges_ind


# ------------------------- Convenience Utilities ---------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


# ----------------------------- Performance Utilities ---------------------------


def graph_model_evaluation(model, trainer, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(model, trainer, fixed_value, fom),
        x0=0.1,
        x1=0.2,
        xtol=0.001,
    )
    print("Seed solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return evaluate_set_metrics(sol.root, model, trainer), sol.root


def evaluate_set_root(edge_cut, model, trainer, goal=0.96, fom="eff"):
    eff, pur = evaluate_set_metrics(edge_cut, model, trainer)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def get_metrics(test_results):

    ps = [result["preds"].sum() for result in test_results[1:]]
    ts = [result["truth"].sum() for result in test_results[1:]]
    tps = [(result["preds"] * result["truth"]).sum() for result in test_results[1:]]

    efficiencies = [tp / t for (t, tp) in zip(ts, tps)]
    purities = [tp / p for (p, tp) in zip(ps, tps)]

    mean_efficiency = np.mean(efficiencies)
    mean_purity = np.mean(purities)

    return mean_efficiency, mean_purity


def evaluate_set_metrics(edge_cut, model, trainer):
    model.hparams.edge_cut = edge_cut
    test_results = trainer.test(ckpt_path=None)

    mean_efficiency, mean_purity = get_metrics(test_results)

    print(mean_purity, mean_efficiency)

    return mean_efficiency, mean_purity
