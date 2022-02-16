"""
TODO:

- Change manual scipy sparse conversion to PyG version for brevity
"""

import os
import logging

import torch
import scipy.sparse.csgraph as scigraph
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix


def label_graph(
    input_file: str, output_dir: str, edge_cut: float = 0.5, **kwargs
) -> None:

    """Loads an input_file and outputs a segmented (i.e. labelled) graph.

    Args:
        input_file: Location of the input graph (a torch pickled file containing a Pytorch Geometric data object).
        edge_cut: The minimum score for an edge to become part of a segment

    """

    try:

        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])

        if not os.path.exists(output_file) or overwrite:

            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location="cpu")

            # apply cut
            passing_edges = graph.edge_index[:, graph.scores > edge_cut]

            # attach labels to data
            graph.labels = label_segments(passing_edges, len(graph.x))

            with open(output_file, "wb") as pickle_file:
                torch.save(data, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)


def labelSegments(input_edges, num_nodes):

    # get connected components
    sparse_edges = sp.coo_matrix(
        (np.ones(input_edges.shape[1]), input_edges.cpu().numpy()),
        shape=(num_nodes, num_nodes),
    )
    connected_components = scigraph.connected_components(sparse_edges)[1]

    return torch.from_numpy(connected_components).type_as(input_edges)


def sparse_score_segments(labels, pids, signal_pids):

    unique_pids, new_pids = pids.unique(return_inverse=True)
    _, new_labels = labels.unique(return_inverse=True)
    signal_segments_pids, unique_signal_segments_pids = get_unique_signal_segments(
        new_labels, new_pids, signal_pids
    )

    iou, segment_count, pid_count = get_jaccard_matrix(
        new_labels, new_pids, signal_segments_pids, unique_signal_segments_pids
    )

    sparse_segment_count = sp.coo_matrix(
        (
            segment_count[unique_signal_segments_pids[0]].cpu(),
            unique_signal_segments_pids.cpu().numpy(),
        )
    ).tocsr()
    sparse_pid_count = sp.coo_matrix(
        (
            pid_count[unique_signal_segments_pids[1]].cpu(),
            unique_signal_segments_pids.cpu().numpy(),
        )
    ).tocsr()

    segment_pur = (
        iou.multiply(sparse_segment_count).sum()
        / segment_count[unique_signal_segments_pids[0]].sum()
    )
    segment_eff = (
        iou.multiply(sparse_pid_count).sum()
        / pid_count[unique_signal_segments_pids[1]].sum()
    )

    segment_f1 = 2 * segment_pur * segment_eff / (segment_pur + segment_eff)

    return segment_pur, segment_eff, segment_f1


def get_jaccard_matrix(labels, pids, signal_segments_pids, unique_signal_segments_pids):

    sparse_intersection = sp.coo_matrix(
        (np.ones(signal_segments_pids.shape[1]), signal_segments_pids.cpu().numpy())
    ).tocsr()

    segment_count = labels.unique(return_counts=True)[1]
    pid_count = pids.unique(return_counts=True)[1]

    union_counts = (
        segment_count[unique_signal_segments_pids[0]]
        + pid_count[unique_signal_segments_pids[1]]
    )
    sparse_sum = sp.coo_matrix(
        (union_counts.cpu(), unique_signal_segments_pids.cpu().numpy())
    ).tocsr()
    sparse_union = sparse_sum - sparse_intersection
    sparse_union.data = 1 / sparse_union.data
    iou = sparse_intersection.multiply(sparse_union)

    return iou, segment_count, pid_count


def get_unique_signal_segments(labels, pids, signal_pids):

    labels_unique, labels_inverse, labels_counts = labels.unique(
        return_counts=True, return_inverse=True
    )

    segments_pids = torch.stack([labels, pids])
    is_signal = torch.isin(pids, pids[signal_pids]) & (
        labels_counts[labels_inverse] >= 3
    )

    signal_segments_pids = segments_pids[:, is_signal]
    unique_signal_segments_pids = signal_segments_pids.unique(dim=1)

    return signal_segments_pids, unique_signal_segments_pids
