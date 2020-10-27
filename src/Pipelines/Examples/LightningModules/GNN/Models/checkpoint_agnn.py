import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from ..utils import make_mlp

class CheckpointedResAGNN(GNNBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.node_encoder = make_mlp(hparams["in_channels"], [hparams["hidden"]],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layernorm"])
   
        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(2*(hparams["in_channels"] + hparams["hidden"]),
                                     [hparams["hidden"]]*hparams["nb_edge_layer"]+[1],
                                     layer_norm=hparams["layernorm"],
                                     output_activation=None,
                                     hidden_activation = hparams["hidden_activation"])

        # The node network computes new node features
        self.node_network = make_mlp((hparams["in_channels"] + hparams["hidden"])*2,
                                     [hparams["hidden"]]*hparams["nb_node_layer"],
                                     layer_norm=hparams["layernorm"],
                                     output_activation=None,
                                     hidden_activation = hparams["hidden_activation"])

    def forward(self, x, edge_index):
        
        # Encode the graph features into the hidden space
        input_x = x
        x = self.node_encoder(x)
        x = torch.cat([x, input_x], dim=-1)
        
        start, end = edge_index

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            # Previous hidden state
            x0 = x

            # Compute new edge score
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = checkpoint(self.edge_network, edge_inputs)
            e = torch.sigmoid(e)
            
            # Sum weighted node features coming into each node
#             weighted_messages_in = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])
#             weighted_messages_out = scatter_add(e * x[end], start, dim=0, dim_size=x.shape[0])
            
            weighted_messages = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0]) + scatter_add(e * x[end], start, dim=0, dim_size=x.shape[0])

            # Compute new node features
#             node_inputs = torch.cat([x, weighted_messages_in, weighted_messages_out], dim=1)
            node_inputs = torch.cat([x, weighted_messages], dim=1)
            x = checkpoint(self.node_network, node_inputs)

            # Residual connection
            x = torch.cat([x, input_x], dim=-1)
            x = x + x0

        # Compute final edge scores; use original edge directions only
        clf_inputs = torch.cat([x[start], x[end]], dim=1)
        return checkpoint(self.edge_network, clf_inputs).squeeze(-1)
    