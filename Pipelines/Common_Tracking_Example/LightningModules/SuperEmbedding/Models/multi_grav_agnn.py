import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..sandbox_base import SandboxEmbeddingBase
from ...GNN.utils import make_mlp


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
        emb_dim,
        nb_layers,
        attention_heads = 1,
        hidden_activation="Tanh",
        layer_norm=True,
        batch_norm=True,
        hparams=None
    ):
        super(NodeNetwork, self).__init__()
        self.hparams = hparams
        self.attention_heads = attention_heads
        
        self.feature_network = make_mlp(
                input_dim + 3*attention_heads * input_dim,
                [output_dim] * nb_layers,
                hidden_activation=hidden_activation,
                output_activation=None,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
            )
        
        self.spatial_network = make_mlp(
                input_dim,
                [input_dim]*nb_layers + [emb_dim * attention_heads],
                hidden_activation=hidden_activation,
                output_activation=None,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
            )

    def grav_pool(self, hidden_features, spatial_features, edge_index):
        start, end = edge_index
        d_weight = torch.sum((spatial_features[start] - spatial_features[end])**2, dim=-1) # euclidean distance
        d_weight = torch.exp(-d_weight)

        if "hidden_norm" in self.hparams and self.hparams["hidden_norm"]:
            hidden_features = F.normalize(hidden_features, p=1, dim=-1)

        hidden_edge_features = hidden_features[start] * d_weight.unsqueeze(-1)

        hidden_features = torch.cat([
            scatter_add(hidden_edge_features, end, dim=0, dim_size=hidden_features.shape[0]),
            scatter_mean(hidden_edge_features, end, dim=0, dim_size=hidden_features.shape[0]),
            scatter_max(hidden_edge_features, end, dim=0, dim_size=hidden_features.shape[0])[0],
        ], dim=1)

        return hidden_features

    def forward(self, x, edge_index):
        start, end = edge_index
        spatial_features = self.spatial_network(x)
        spatial_features = spatial_features.reshape(spatial_features.shape[0], self.attention_heads, -1)

        if "emb_norm" in self.hparams and self.hparams["emb_norm"]:
            spatial_features = F.normalize(spatial_features, dim=-1)

        hidden_features = self.grav_pool(x, spatial_features, edge_index)

        hidden_features = hidden_features.reshape(hidden_features.shape[0], -1)
        hidden_features = torch.cat([x, hidden_features], dim=-1)

        return self.feature_network(hidden_features)


class MultiGravAGNN(SandboxEmbeddingBase):
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

        # Setup the node layers
        self.node_network = nn.ModuleList(
            [NodeNetwork(
                hparams["feature_hidden"],
                hparams["feature_hidden"],
                hparams["emb_dim"],
                hparams["nb_layer"],
                hparams["attention_heads"],
                hparams["activation"],
                hparams["layernorm"],
                hparams["batchnorm"],
                hparams
            ) for _ in range(hparams["n_graph_iters"])]
        )

        self.output_network = make_mlp(
            hparams["feature_hidden"],
            [hparams["feature_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

    def output_step(self, x):

        x_out = self.output_network(x)
        return F.normalize(x_out) if "norm" in self.hparams["regime"] else x_out

    def forward(self, batch):

        edge_index = torch.cat([batch.edge_index, batch.edge_index.flip(0)], dim=1)
        x = self.input_network(torch.cat([batch.x, batch.cell_data[:, : self.hparams["cell_channels"]]], axis=-1))

        # Loop over iterations of edge and node networks
        for net in self.node_network:

            x = checkpoint(net, x, edge_index)
            
        return self.output_step(x)
