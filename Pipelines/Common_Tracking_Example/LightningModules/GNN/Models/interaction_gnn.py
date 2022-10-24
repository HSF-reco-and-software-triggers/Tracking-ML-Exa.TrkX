import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase, LargeGNNBase
from ..utils import make_mlp


class InteractionGNN(LargeGNNBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
        )

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
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
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )


    def message_step(self, x, start, end, e):

        # Compute new node features
        if self.hparams["aggregation"] == "sum":
            edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0])

        elif self.hparams["aggregation"] == "mean":
            edge_messages = scatter_mean(e, end, dim=0, dim_size=x.shape[0])
            
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
        elif self.hparams["aggregation"] == "mean_sum":
            edge_messages = torch.cat(
                [
                    scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
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

        x_out += x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        e_out += e

        return x_out, e_out

    def output_step(self, x, start, end, e):

        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)

        return self.output_edge_classifier(classifier_inputs).squeeze(-1)

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1))

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            x, e = checkpoint(self.message_step, x, start, end, e)

        return self.output_step(x, start, end, e)
