import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_scatter import scatter
from torch.utils.checkpoint import checkpoint

from ..stitcher_base import StitcherBase
from ..utils.data_utils import make_mlp


class InteractionGNN(StitcherBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        self.segment_network = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        self.output_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def message_step(self, x, start, end, e):

        # Compute new node features
        if self.hparams["aggregation"] == "sum":
            edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0])

        elif self.hparams["aggregation"] == "max":
            edge_messages = scatter_max(e, end, dim=0, dim_size=x.shape[0])[0]

        elif self.hparams["aggregation"] == "sum_max":
            edge_messages = torch.cat(
                [
                    scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                    scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                ],
                dim=-1,
            )

        elif self.hparams["aggregation"] == "mean_max":
            edge_messages = torch.cat(
                [
                    scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                    scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
                ],
                dim=-1,
            )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs)

        x_out = (x + x_out) / 2

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        e_out = (e + e_out) / 2

        return x_out, e_out

    def output_step(self, segment_features, label_pairs):

        pair_features = torch.cat(
            [segment_features[label_pairs[0]], segment_features[label_pairs[1]]], dim=-1
        )

        return self.output_network(pair_features)

    def forward(self, x, edge_index, labels, label_pairs):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1))

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            x, e = checkpoint(self.message_step, x, start, end, e)

        # Compute final node representation
        segment_features = scatter(
            x, labels, dim=0, dim_size=int(labels.max().item() + 1), reduce="mean"
        )
        segment_features = checkpoint(self.segment_network, segment_features)

        return checkpoint(self.output_step, segment_features, label_pairs)
