import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..hetero_gnn_base import LargeGNNBase
from ..utils import make_mlp

from .submodels.edge_decoders import HomoDecoder, HeteroDecoder
from .submodels.convolutions import HomoConv, HeteroConv
from .submodels.encoders import HomoEncoder, HeteroEncoder

class HeteroGNN(LargeGNNBase):

    """
    A heterogeneous interaction network class
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

        # Make input network
        if self.hparams["hetero_level"] >= 1:
            self.encoder = HeteroEncoder(hparams)
        else:
            self.encoder = HomoEncoder(hparams)

        # Make message passing network with the following configuration:
        # Check whether the MP network should be homogeneous or heterogeneous (i.e. hetero_level >= 2)
        # Check whether the MP network should share weights between iterations (i.e. is recurrent)
        if "hetero_level" in self.hparams and self.hparams["hetero_level"] >= 2:
            if "recurrent" in self.hparams and not self.hparams["recurrent"]:
                self.conv = nn.ModuleList([HeteroConv(hparams) for _ in range(self.hparams["n_graph_iters"])])
            else:
                self.conv = nn.ModuleList([HeteroConv(hparams)]*self.hparams["n_graph_iters"])
        else:
            if "recurrent" in self.hparams and not self.hparams["recurrent"]:
                self.conv = nn.ModuleList([HomoConv(hparams) for _ in range(self.hparams["n_graph_iters"])])
            else:
                self.conv = nn.ModuleList([HomoConv(hparams)]*self.hparams["n_graph_iters"])        

        # Make output network
        if self.hparams["hetero_level"] >= 3:
            self.decoder = HeteroDecoder(hparams)
        else:
            self.decoder = HomoDecoder(hparams)

    def forward(self, x, edge_index, volume_id):

        # Encode the graph features into the hidden space
        x.requires_grad = True
        encoded_nodes, encoded_edges = self.encoder(x, edge_index, volume_id)
        
        # Loop over iterations of edge and node networks
        for step in range(self.hparams["n_graph_iters"]):

            encoded_nodes, encoded_edges = checkpoint(self.conv[step], encoded_nodes, edge_index, encoded_edges, volume_id)

        # Compute final edge scores
        # TODO: Apply output to SUM of directional edge features (across both directions!)
        return self.decoder(encoded_nodes, edge_index, encoded_edges, volume_id)