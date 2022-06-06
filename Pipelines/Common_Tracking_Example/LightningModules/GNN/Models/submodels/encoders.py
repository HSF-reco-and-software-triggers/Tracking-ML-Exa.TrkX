import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ...utils import make_mlp

class HeteroEncoder(torch.nn.Module):
    """
    The node and edge encoder(s) that can handle heterogeneous models        
    """
    
    def __init__(self, hparams):
        super(HeteroEncoder, self).__init__()

        self.hparams = hparams

        # Make module list
        self.node_encoders = nn.ModuleList([
            make_mlp(
                model["num_features"],
                [hparams["hidden"]] * hparams["nb_node_layer"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            ) for model in hparams["model_ids"]
        ])

        # Make edge encoder combos (this is an N-choose-2 with replacement situation)
        self.all_combos = torch.combinations(torch.arange(len(self.hparams["model_ids"])), r=2, with_replacement=True)
    
        self.edge_encoders = nn.ModuleList([
            make_mlp(
                2 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            ) for _ in self.all_combos
        ])

    def fill_hetero_nodes(self, input_node_features, volume_id):
        """
        Fill the heterogeneous nodes with the corresponding encoders
        """

        features_to_fill = torch.empty((input_node_features.shape[0], self.hparams["hidden"])).to(input_node_features.device)

        for encoder, model in zip(self.node_encoders, self.hparams["model_ids"]):
            node_id_mask = torch.isin(volume_id, torch.tensor(model["volume_ids"]).to(input_node_features.device))
            features_to_fill[node_id_mask] = encoder(input_node_features[node_id_mask, :model["num_features"]])
        
        return features_to_fill

    def fill_hetero_edges(self, input_node_features, start, end, volume_id):
        """
        Fill the heterogeneous edges with the corresponding encoders
        """

        features_to_fill = torch.empty((start.shape[0], self.hparams["hidden"])).to(start.device)

        for encoder, combo in zip(self.edge_encoders, self.all_combos):
            vol_ids_0, vol_ids_1 = torch.tensor(self.hparams["model_ids"][combo[0]]["volume_ids"], device=features_to_fill.device), torch.tensor(self.hparams["model_ids"][combo[1]]["volume_ids"], device=features_to_fill.device)                        
            vol_edge_mask = torch.isin(volume_id[start], vol_ids_0) & torch.isin(volume_id[end], vol_ids_1)
            
            features_to_encode = torch.cat([
                    input_node_features[start[vol_edge_mask]],
                    input_node_features[end[vol_edge_mask]]         
                ], dim=-1)

            features_to_fill[vol_edge_mask] = encoder(features_to_encode)

        return features_to_fill

    def forward(self, x, edge_index, volume_id=None):
        """
        Forward pass of the heterogeneous encoder

        x is the input node features. It is expected to be of shape (num_nodes, max number of input features), where rows that have fewer features than the max are padded with zeros.
        It creates an empty tensor of the same shape as x, and fills it with the encoded nodes, for each node encoder. The same is done for encoded edges.
        """

        start, end = edge_index

        encoded_nodes = self.fill_hetero_nodes(x, volume_id)
        encoded_edges = self.fill_hetero_edges(encoded_nodes, start, end, volume_id)        

        return encoded_nodes, encoded_edges

class HomoEncoder(torch.nn.Module):
    """
    The node and edge encoder(s) that can handle homogeneous models        
    """
    
    def __init__(self, hparams):
        super(HomoEncoder, self).__init__()

        self.hparams = hparams

        # Make module list
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index, volume_id=None):
        """
        Forward pass of the homogeneous encoder
        """
        
        x = x[:, :self.hparams["spatial_channels"]]
        start, end = edge_index

        encoded_nodes = self.node_encoder(x)
        encoded_edges = self.edge_encoder(torch.cat([encoded_nodes[start], encoded_nodes[end]], dim=-1))

        return encoded_nodes, encoded_edges