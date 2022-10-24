import os, sys
import logging

import torch.nn as nn
import torch
import pandas as pd
import numpy as np

from .segmentation_utils import labelSegments


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
        loaded_events = process_data(loaded_events)
        print("Events processed!")
        return loaded_events
    else:
        return None


def process_data(events):

    # Handle event in batched form
    for i, event in enumerate(events):

        event.labels = labelSegments(event.edge_index[:, event.y.bool()], len(event.x))

        event.long_mask, long_labels, long_pid = get_long_segments(event)
        event.label_pairs, event.pid_pairs = get_segment_pairs(
            event, long_labels, long_pid
        )

    return events


def get_long_segments(event):

    long_segments, segment_inverse, segment_counts = event.labels.unique(
        return_counts=True, return_inverse=True
    )
    long_mask = segment_counts[segment_inverse] >= 3

    long_labels, long_pid = event.labels[long_mask], event.pid[long_mask]

    return long_mask, long_labels, long_pid


def get_segment_pairs(event, long_labels, long_pid):

    label_pairs = torch.combinations(long_labels.unique(), r=2).T
    label_pairs = randomise_pairs(label_pairs)

    matching = pd.DataFrame(
        np.array(
            [
                np.concatenate([[124], long_labels.numpy()]),
                np.concatenate([[0], long_pid.numpy()]),
                np.ones(len(long_labels) + 1),
            ]
        ).T,
        columns=["labels", "pid", "count"],
    )
    max_pid_df = (
        matching.groupby(["labels", "pid"], as_index=False)
        .count()
        .sort_values("count")
        .drop_duplicates(["labels"], keep="last")
    )
    max_pid = max_pid_df[["labels", "pid"]].to_numpy().astype(int)

    dom_pid_map = np.zeros(label_pairs.max() + 1)
    dom_pid_map[max_pid[:, 0]] = max_pid[:, 1]
    pid_pairs = dom_pid_map[label_pairs]

    return label_pairs, torch.from_numpy(pid_pairs).long()


def randomise_pairs(label_pairs):
    # Re-introduce random direction, to avoid training bias
    random_flip = torch.randint(2, (label_pairs.shape[1],)).bool()
    label_pairs[0, random_flip], label_pairs[1, random_flip] = (
        label_pairs[1, random_flip],
        label_pairs[0, random_flip],
    )

    return label_pairs


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
