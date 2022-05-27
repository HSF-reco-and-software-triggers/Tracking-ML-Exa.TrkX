import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ...utils import make_mlp

def get_aggregation(aggregation):
    """
    Factory dictionary for aggregation depending on the hparams["aggregation"]
    """

    aggregation_dict = {
        "sum": lambda e, end, x: scatter_add(e, end, dim=0, dim_size=x.shape[0]),
        "mean": lambda e, end, x: scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
        "max": lambda e, end, x: scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
        "sum_max": lambda e, end, x: torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        ),
        "mean_sum": lambda e, end, x: torch.cat(
            [
                scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        ),
        "mean_max": lambda e, end, x: torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        ),
    }

    return aggregation_dict[aggregation]

class HeteroConv(torch.nn.Module):
    """
    The node and edge GNN convolution that can handle heterogeneous models        
    """

    def __init__(self, hparams):
        super(HeteroConv, self).__init__()

        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
        )

        # Make module list
        self.node_encoders = nn.ModuleList([
            make_mlp(
                concatenation_factor * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_node_layer"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            ) for _ in hparams["model_ids"]
        ])

        # Make edge encoder combos (this is an N-choose-2 with replacement situation)
        self.all_combos = torch.combinations(torch.arange(len(self.hparams["model_ids"])), r=2, with_replacement=True)
    
        self.edge_encoders = nn.ModuleList([
            make_mlp(
                3 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            ) for _ in self.all_combos
        ])

    def forward(self, x, edge_index, e, volume_id=None):

        start, end = edge_index

        # Perform message passing
        edge_messages = get_aggregation(self.hparams["aggregation"])(e, end, x)
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        x_out = self.fill_hetero_nodes(node_inputs, volume_id)
        x_out += x

        # Compute new edge features
        e_out = self.fill_hetero_edges(e, x_out, start, end, volume_id)
        e_out += e

        return x_out, e_out

    def fill_hetero_nodes(self, input_node_features, volume_id):
        """
        Fill the heterogeneous nodes with the corresponding encoders
        """
        features_to_fill = torch.empty((input_node_features.shape[0], self.hparams["hidden"])).to(input_node_features.device)
        
        for encoder, model in zip(self.node_encoders, self.hparams["model_ids"]):
            node_id_mask = torch.isin(volume_id, torch.tensor(model["volume_ids"]).to(input_node_features.device))
            features_to_fill[node_id_mask] = encoder(input_node_features[node_id_mask])
        
        return features_to_fill

    def fill_hetero_edges(self, input_edge_features, input_node_features, start, end, volume_id):
        """
        Fill the heterogeneous edges with the corresponding encoders
        """
        features_to_fill = torch.empty((start.shape[0], self.hparams["hidden"])).to(start.device)

        for encoder, combo in zip(self.edge_encoders, self.all_combos):
            vol_ids_0, vol_ids_1 = torch.tensor(self.hparams["model_ids"][combo[0]]["volume_ids"], device=features_to_fill.device), torch.tensor(self.hparams["model_ids"][combo[1]]["volume_ids"], device=features_to_fill.device)                        
            vol_edge_mask = torch.isin(volume_id[start], vol_ids_0) & torch.isin(volume_id[end], vol_ids_1)
            
            features_to_encode = torch.cat([
                    input_node_features[start[vol_edge_mask]],
                    input_node_features[end[vol_edge_mask]],
                    input_edge_features[vol_edge_mask]         
                ], dim=-1)

            features_to_fill[vol_edge_mask] = encoder(features_to_encode)

        return features_to_fill

class HomoConv(torch.nn.Module):

    """
    A simple message passing convolution (a la Interaction Network) that handles only homogeoneous models
    """

    def __init__(self, hparams):
        super(HomoConv, self).__init__()

        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
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
            
        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index, e, volume_id=None):

        start, end = edge_index

        # Perform message passing
        edge_messages = get_aggregation(self.hparams["aggregation"])(e, end, x)
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        x_out = self.node_network(node_inputs)
        if "concat_output" not in self.hparams or not self.hparams["concat_output"]:
            x_out += x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)
        if "concat_output" not in self.hparams or not self.hparams["concat_output"]:
            e_out += e

        return x_out, e_out