import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
import torch_geometric
from torch_geometric.typing import Adj, EdgeType, NodeType
from typing import Dict, Optional
from torch import Tensor
from torch.nn import Module, ModuleDict

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
        node_inputs = torch.cat([x,  edge_messages], dim=-1)
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

class InteractionHeteroConv(torch_geometric.nn.HeteroConv):
    def __init__(self, convs: Dict[EdgeType, Module], aggr: Optional[str] = "sum"):
        super().__init__(convs, aggr)

    def edge_forward(self,x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_dict,
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:

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
            edge = edge_dict[edge_type]

            out = conv.edge_update((x_dict[src], x_dict[dst]), edge, edge_index, *args, **kwargs)

            # if src == dst:
            #     out = conv.edge_updater(x_dict[src], edge_index, *args, **kwargs)
            # else:
            #     out = conv.edge_update((x_dict[src], x_dict[dst]), edge_index, *args,
            #             **kwargs)

            out_dict[edge_type] = out

        return out_dict

class InteractionMessagePassing(torch_geometric.nn.MessagePassing):
    def __init__(self, hparams, aggr: str = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super().__init__(aggr, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)

        self.hparams=hparams

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
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

    def message(self, edge):
        return edge

    def aggregate(self, out, edge_index):

        src, dst = edge_index
        return self.aggr_module(out, dst)[dst.unique()]
    
    def update(self, agg_message, x, edge_index):
        src, dst = edge_index
        indices_to_add = torch.arange(agg_message.shape[0])
        # print(dst.unique())
        # print(x)
        x[dst.unique()] += agg_message
        
        return x

    def edge_update(self, x, edge, edge_index, *args, **kwargs):
        src, dst = edge_index
        if isinstance(x, tuple):
            x_src, x_dst = x[0][src], x[1][dst]
        else:
            x_src, x_dst = x[src], x[dst]
        out = self.edge_network(torch.cat([x_src, x_dst, edge], dim=-1))
        return out

    def forward(self, x, edge_index, edge):

        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src, x_dst = x, x

        x_dst = self.propagate(edge_index, x=x_dst, edge=edge)

        return x_dst