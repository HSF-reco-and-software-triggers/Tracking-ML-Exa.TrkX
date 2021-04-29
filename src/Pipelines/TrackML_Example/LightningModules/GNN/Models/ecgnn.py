import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint

from ..gnn_contract import GNNContract
from ..utils import make_mlp
from ..EdgePooling import EdgePooling


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2, [output_dim]*nb_layers,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
#         mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
#         mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
#         node_inputs = torch.cat([mi, mo, x], dim=1)
        messages = scatter_add(x[start], end, dim=0, dim_size=x.shape[0]) + scatter_add(x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([messages, x], dim=1)
        return self.network(node_inputs)
       

class ECGNN(GNNContract):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.input_network = make_mlp(hparams["in_channels"], [hparams["hidden"]],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layernorm"])
        # Setup the node layers
        self.node_network = NodeNetwork(hparams["in_channels"] + hparams["hidden"], hparams["hidden"],
                                        hparams["nb_node_layer"], hparams["hidden_activation"], hparams["layernorm"])
        
        # Setup the edge pooling layers
        self.edgepool_network = EdgePooling(hparams["hidden"])
        
        #array for storing unpooling results
        self.unpool = []

    def forward(self, x, edge_index):
    
        input_x = x

        x = self.input_network(x)
        x = torch.cat([x, input_x], dim=-1)
        x = self.node_network(x,edge_index)
        
        batch = torch.zeros(x.size()[0],dtype=torch.int64)
        
        x, edge_index, batch, unpool_info, edge_scores= self.edgepool_network(x,edge_index,batch)
        
        return edge_scores, unpool_info