import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GINConv

from ..gnn_base import GNNBase, LargeGNNBase
from ..utils import make_mlp



class PyG_GNN(LargeGNNBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

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
        # self.edge_encoder = make_mlp(
        #     2 * (hparams["hidden"]),
        #     [hparams["hidden"]] * hparams["nb_edge_layer"],
        #     layer_norm=hparams["layernorm"],
        #     batch_norm=hparams["batchnorm"],
        #     output_activation=hparams["output_activation"],
        #     hidden_activation=hparams["hidden_activation"],
        # )

        # The edge network computes new edge features from connected nodes
        # self.edge_network = make_mlp(
        #     3 * hparams["hidden"],
        #     [hparams["hidden"]] * hparams["nb_edge_layer"],
        #     layer_norm=hparams["layernorm"],
        #     batch_norm=hparams["batchnorm"],
        #     output_activation=hparams["output_activation"],
        #     hidden_activation=hparams["hidden_activation"],
        # )

        # The node network computes new node features
        self.node_network = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        self.edge_conv = GINConv(self.node_network, train_eps=True)

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )


    def output_step(self, x, start, end):

        classifier_inputs = torch.cat([x[start], x[end]], dim=1)

        return self.output_edge_classifier(classifier_inputs).squeeze(-1)

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = self.node_encoder(x)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            x = self.edge_conv(x, edge_index)

        # Compute final edge scores; use original edge directions only
        # return checkpoint(self.output_step, x, start, end, e)
        return self.output_step(x, start, end)
