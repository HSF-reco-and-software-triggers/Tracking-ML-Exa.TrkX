import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint

from ..sandbox_base import SandboxEmbeddingBase
from ...GNN.utils import make_mlp


class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        nb_layers,
        hidden_activation="SiLU",
        layer_norm=True,
        batch_norm=True,
    ):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(
            input_dim * 2,
            [hidden_dim] * nb_layers + [1],
            hidden_activation=hidden_activation,
            output_activation=None,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
        )

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
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

    def __init__(
        self,
        input_dim,
        output_dim,
        nb_layers,
        hidden_activation="Tanh",
        layer_norm=True,
        batch_norm=True,
    ):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(
            input_dim * 2,
            [output_dim] * nb_layers,
            hidden_activation=hidden_activation,
            output_activation=None,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
        )

    def forward(self, x, e, edge_index):
        start, end = edge_index
        
        messages = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([messages, x], dim=1)

        return self.network(node_inputs)

class ResAGNN(SandboxEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        self.input_channels = hparams["spatial_channels"] + hparams["cell_channels"]

        # Setup input network
        self.input_network = make_mlp(
            self.input_channels,
            [hparams["feature_hidden"]] * hparams["nb_layer"],
            output_activation=hparams["activation"],
            layer_norm=hparams["layernorm"],
        )

        # Setup the edge network
        self.edge_network = EdgeNetwork(
            hparams["feature_hidden"],
            hparams["feature_hidden"],
            hparams["nb_layer"],
            hparams["activation"],
            hparams["layernorm"],
            hparams["batchnorm"],
        )
        # Setup the node layers
        self.node_network = NodeNetwork(
            hparams["feature_hidden"],
            hparams["feature_hidden"],
            hparams["nb_layer"],
            hparams["activation"],
            hparams["layernorm"],
            hparams["batchnorm"],
        )

        self.output_network = make_mlp(
            hparams["feature_hidden"],
            [hparams["feature_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

    def message_step(self, x, edge_index):

        x_inital = x

        # Apply edge network
        e = torch.sigmoid(self.edge_network(x, edge_index))

        # Apply node network
        x = self.node_network(x, e, edge_index)

        # Residual connection
        return x_inital + x

    def output_step(self, x):

        x_out = self.output_network(x)
        return F.normalize(x_out) if "norm" in self.hparams["regime"] else x_out

    def forward(self, batch):

        edge_index = torch.cat([batch.edge_index, batch.edge_index.flip(0)], dim=1)
        x = self.input_network(torch.cat([batch.x, batch.cell_data[:, : self.hparams["cell_channels"]]], axis=-1))

        # Loop over iterations of edge and node networks
        for _ in range(self.hparams["n_graph_iters"]):

            x = checkpoint(self.message_step, x, edge_index)
            
        return self.output_step(x)
