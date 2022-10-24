import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import HeteroConv

from ...utils import make_mlp, get_region

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


class NodeEncoder(torch.nn.Module):
    def __init__(self, hparams, model) -> None:
        super().__init__()
        self.hparams = hparams

        self.network = make_mlp(
            model['num_features'],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def forward(self, x):
        return  self.network( x.float() )
    
class HeteroNodeEncoder(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self.encoders = torch.nn.ModuleDict()
        for model in self.hparams['model_ids']:
            region = get_region(model)
            self.encoders[region] = NodeEncoder(self.hparams, model)
        
    def forward(self, x_dict):
        for model in self.hparams['model_ids']:
            region = get_region(model)
            print(region)
            x_dict[region] = self.encoders[region](x_dict[region][:, : model['num_features']])
        
        return x_dict

class EdgeEncoder(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams

        self.network = make_mlp(
            2 * hparams['hidden'],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        print("Encoding edges")
        if isinstance(x, tuple):
            x1, x2 = x
            x_in = torch.cat([x1[src], x2[dst]], dim=-1)
        else:
            x_in = torch.cat([x[src], x[dst]], dim=-1)

        return  self.network( x_in )

class HeteroEdgeConv(HeteroConv):

    def __init__(self, convs: dict, aggr: str = "sum"):
        super().__init__(convs, aggr)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        *args_dict,
        **kwargs_dict,
    ) -> dict :

        out_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            print(edge_type)
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                value_dict.get(dst, None))

            conv = self.convs[str_edge_type]
            print(kwargs)

            if src == dst:
                out = conv(x_dict[src], edge_index, *args, **kwargs)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index, *args, **kwargs)

            out_dict[edge_type] = out

        return out_dict

class EdgeClassifier(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams['output_activation'],
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index, edge):
        src, dst = edge_index
        if isinstance(x, tuple):
            x1, x2 = x
            classifier_input = torch.cat([x1[src], x2[dst], edge], dim=-1)
        else:
            classifier_input = torch.cat([x[src], x[dst], edge], dim=-1)
        # classifier_input = torch.cat([x[src], x[dst], edge], dim=-1)
        return self.network(classifier_input)