import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from ..utils import make_mlp


class VanillaCheckResAGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )

        self.edge_network = make_mlp(
            (hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            hparams["hidden"] * concatenation_factor,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        self.node_encoder = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

    def message_step(self, x, start, end):

        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        e = self.edge_network(edge_inputs)
        e = torch.sigmoid(e)

        if self.hparams["aggregation"] == "sum":
            messages = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])

        elif self.hparams["aggregation"] == "max":
            messages = scatter_max(e * x[start], end, dim=0, dim_size=x.shape[0])[0]

        elif self.hparams["aggregation"] == "sum_max":
            messages = torch.cat(
                [
                    scatter_max(e * x[start], end, dim=0, dim_size=x.shape[0])[0],
                    scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0]),
                ],
                dim=-1,
            )

        node_inputs = torch.cat([messages, x], dim=1)
        x_out = self.node_network(node_inputs)

        x_out += x

        return x_out

    def output_step(self, x, start, end):
        edge_inputs = torch.cat([x[start], x[end]], dim=1)

        return self.edge_network(edge_inputs)

    def forward(self, x, edge_index):
        start, end = edge_index

        x.requires_grad = True
        x = checkpoint(self.node_encoder, x)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            x = checkpoint(self.message_step, x, start, end)

        return checkpoint(self.output_step, x, start, end)
