import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
# from fairscale.nn import checkpoint_wrapper

from ..gnn_base import GNNBase
from ..utils import make_mlp


class MessageNet(nn.Module):
    def __init__(self, node_network, edge_network, hparams):

        super(MessageNet, self).__init__()

        self.node_network = node_network
        self.edge_network = edge_network
        self.hparams = hparams

    def forward(self, x, start, end):

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


class OutputNet(nn.Module):
    def __init__(self, edge_network):
        super(OutputNet, self).__init__()

        self.edge_network = edge_network

    def forward(self, x, start, end):
        edge_inputs = torch.cat([x[start], x[end]], dim=1)

        return self.edge_network(edge_inputs)


class CheckpointedPyramid(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        )

        # Setup input network
        self.node_encoder = checkpoint_wrapper(
            make_mlp(
                hparams["spatial_channels"] + hparams["cell_channels"],
                [hparams["hidden"]] * hparams["nb_node_layer"],
                output_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
            )
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            hparams["hidden"] * 2,
            [hparams["hidden"] // (2**i) for i in range(hparams["nb_edge_layer"])]
            + [1],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            hparams["hidden"] * concatenation_factor,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        self.message_net = checkpoint_wrapper(
            MessageNet(self.node_network, self.edge_network, hparams)
        )
        self.output_net = checkpoint_wrapper(OutputNet(self.edge_network))

    def forward(self, x, edge_index):
        start, end = edge_index

        x.requires_grad = True
        x = self.node_encoder(x)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            x = self.message_net(x, start, end)

        return self.output_net(x, start, end)
