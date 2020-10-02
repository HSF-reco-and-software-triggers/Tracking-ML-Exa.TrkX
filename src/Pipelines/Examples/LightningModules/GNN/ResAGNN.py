import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.nn import Linear
import sys

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim]*nb_layers+[1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

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
        self.network = make_mlp(input_dim*3, [output_dim]*nb_layers,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)

class ResAGNN(GNN_Base):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.input_network = make_mlp(hparams["in_channels"], [hparams["hidden"]],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layer_norm"])
        # Setup the edge network
        self.edge_network = EdgeNetwork(hparams["in_channels"] + hparams["hidden"], hparams["in_channels"] + hparams["hidden"],
                                        hparams["nb_layer"], hparams["hidden_activation"], hparams["layer_norm"])
        # Setup the node layers
        self.node_network = NodeNetwork(hparams["in_channels"] + hparams["hidden"], hparams["hidden"],
                                        hparams["nb_layer"], hparams["hidden_activation"], hparams["layer_norm"])

    def forward(self, x, edge_index):
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))

            # Apply node network
            x = self.node_network(x, e, edge_index)

            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x

        return self.edge_network(x, edge_index)
