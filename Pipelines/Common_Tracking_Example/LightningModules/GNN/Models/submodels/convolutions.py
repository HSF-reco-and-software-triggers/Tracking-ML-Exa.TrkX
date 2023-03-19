from turtle import forward
from itertools import product, combinations_with_replacement
import sys
from tabnanny import check

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
from functools import partial
from collections import defaultdict
from typing import Dict, Optional
from torch_geometric.nn.conv.hgt_conv import group

from ...utils import get_region, make_mlp

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
        x_out = x_out + x

        # Compute new edge features
        e_out = self.fill_hetero_edges(e, x_out, start, end, volume_id)
        e_out = e_out + e

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
            x_out = x_out + x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)
        if "concat_output" not in self.hparams or not self.hparams["concat_output"]:
            e_out = e_out + e

        return x_out, e_out

class InteractionHeteroConv(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
        )

        # Make module list
        if self.hparams['hetero_level'] < 3:
            node_encoder = make_mlp(
                concatenation_factor * self.hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_node_layer"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            )
            self.node_encoders = nn.ModuleDict({
                get_region(model): node_encoder for model in hparams["model_ids"]
            })

            edge_encoder = make_mlp(
                concatenation_factor * self.hparams["hidden"],
                [hparams["hidden"]] * self.hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            )
            self.edge_encoders = nn.ModuleDict({
                '__'.join([get_region(model0), "connected_to", get_region(model1)]): edge_encoder for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
            })

        else:
            self.node_encoders = nn.ModuleDict({
                get_region(model): make_mlp(
                    concatenation_factor * self.hparams["hidden"],
                    [hparams["hidden"]] * hparams["nb_node_layer"],
                    output_activation=hparams["output_activation"],
                    hidden_activation=hparams["hidden_activation"],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                ) for model in hparams["model_ids"]
            })

            # # Make edge encoder combos (this is an N-choose-2 with replacement situation)
            # self.all_combos = torch.combinations(torch.arange(len(self.hparams["model_ids"])), r=2, with_replacement=True)
            if self.hparams.get('hetero_reduce'):
                encoders = {}
                for model0, model1 in combinations_with_replacement(self.hparams['model_ids'], r=2):
                    encoder = make_mlp(
                        concatenation_factor * self.hparams["hidden"],
                        [hparams["hidden"]] * self.hparams["nb_edge_layer"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                    )
                    encoders[ '__'.join([get_region(model0), "connected_to", get_region(model1)]) ]  = encoder
                    encoders[ '__'.join([get_region(model1), "connected_to", get_region(model0)]) ]  = encoder
                self.edge_encoders = nn.ModuleDict(encoders)
            else:
                self.edge_encoders = nn.ModuleDict({
                    '__'.join([get_region(model0), "connected_to", get_region(model1)]): make_mlp(
                        concatenation_factor * self.hparams["hidden"],
                        [hparams["hidden"]] * self.hparams["nb_edge_layer"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                    ) for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
                })

    def update_node(self, x_dict: dict, edge_index_dict: dict, edge_dict: dict):
        for node_type, x in x_dict.items():
            e = torch.cat([
                edge_dict[key] for key in [k for k in edge_dict.keys() if node_type==k[-1]]
            ], dim=0)
            end = torch.cat([
                edge_index_dict[key][1] for key in [k for k in edge_dict.keys() if node_type==k[-1]]
            ], dim=0)
            message = get_aggregation(self.hparams['aggregation'])(e, end, x)

            x_in = torch.cat([x, message], dim=-1)         
            if self.hparams.get('checkpoint'):
                x_dict[node_type] = x + checkpoint(self.node_encoders[node_type], x_in)
            else:
                x_dict[node_type] = x + self.node_encoders[node_type](x_in)

        return x_dict

    def update_edge(self, x_dict: dict, edge_index_dict: dict, edge_dict: dict)        :
        for edge_type, edge, in edge_dict.items():
            src, _, dst = edge_type
            x_src, x_dst = x_dict[src][edge_index_dict[edge_type][0]], x_dict[dst][edge_index_dict[edge_type][1]]
            edge_in = torch.cat([x_src, x_dst, edge], dim=-1)
            if self.hparams.get('checkpoint'):
                edge_dict[edge_type] = edge + checkpoint(self.edge_encoders['__'.join(edge_type)], edge_in)
            else:
                edge_dict[edge_type] = edge + self.edge_encoders['__'.join(edge_type)](edge_in)
        return edge_dict

    def forward(self, x_dict, edge_index_dict, edge_dict):
        x_dict = self.update_node(x_dict, edge_index_dict, edge_dict)
        edge_dict = self.update_edge(x_dict, edge_index_dict, edge_dict)
        return x_dict, edge_dict

class NodeOnlyHeteroConv(InteractionHeteroConv):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
        )

        # Make module list
        if self.hparams['hetero_level'] < 3:
            node_encoder = make_mlp(
                concatenation_factor * self.hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_node_layer"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            )
            self.node_encoders = nn.ModuleDict({
                get_region(model): node_encoder for model in hparams["model_ids"]
            })

        else:
            self.node_encoders = nn.ModuleDict({
                get_region(model): make_mlp(
                    concatenation_factor * self.hparams["hidden"],
                    [hparams["hidden"]] * hparams["nb_node_layer"],
                    output_activation=hparams["output_activation"],
                    hidden_activation=hparams["hidden_activation"],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                ) for model in hparams["model_ids"]
            })
            
        edge_encoder = make_mlp(
            concatenation_factor * self.hparams["hidden"],
            [hparams["hidden"]] * self.hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        self.edge_encoders = nn.ModuleDict({
            '__'.join([get_region(model0), "connected_to", get_region(model1)]): edge_encoder for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
        })


class PyGInteractionHeteroConv(torch_geometric.nn.HeteroConv):
    def __init__(self, convs, aggr: Optional[str] = "sum"):
        super().__init__(convs, aggr)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
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

            if src == dst:
                out = conv(x_dict[src], edge_index, *args, **kwargs)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index, *args, **kwargs)

            out_dict[dst].append(out)

            # out_edge_dict[edge_type] = edge_out

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict #, out_edge_dict

class InteractionMessagePassing(torch_geometric.nn.MessagePassing):
    def __init__(self, hparams, aggr: str = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super().__init__(aggr, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)

        self.hparams=hparams

        # self.aggr_module = get_aggregation(self.hparams['aggregation'])

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
        )

        # The edge network computes new edge features from connected nodes
        # self.edge_network = make_mlp(
        #     3 * hparams["hidden"],
        #     [hparams["hidden"]] * hparams["nb_edge_layer"],
        #     layer_norm=hparams["layernorm"],
        #     batch_norm=hparams["batchnorm"],
        #     output_activation=hparams["output_activation"],
        #     hidden_activation=hparams["hidden_activation"],
        # )

        # The node network computes new node features
        self.node_network = make_mlp(
            concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

    def message(self, edge):
        return edge

    def aggregate(self, out, x, edge_index):

        src, dst = edge_index
        return get_aggregation(self.hparams['aggregation'])(out, dst, x)
    
    def update(self, agg_message, x, edge_index):
        x_in = torch.cat([x, agg_message], dim=-1)
        if self.hparams.get('checkpoint'):
            return x + checkpoint(self.node_network, x_in)
        else:
            return x + self.node_network(x_in)

    # def edge_update(self, x, edge, edge_index, *args, **kwargs):
    #     src, dst = edge_index
    #     if isinstance(x, tuple):
    #         x_src, x_dst = x[0][src], x[1][dst]
    #     else:
    #         x_src, x_dst = x[src], x[dst]
    #     if self.hparams.get('checkpoint'):
    #         return edge + checkpoint(self.edge_network, torch.cat([x_src, x_dst, edge], dim=-1))
    #     else:
    #         return edge + self.edge_network(torch.cat([x_src, x_dst, edge], dim=-1))

    def forward(self, x, edge_index, edge):

        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src, x_dst = x, x

        x_dst = self.propagate(edge_index, x=x_dst, edge=edge)

        # edge_out = self.edge_update((x_src, x_dst), edge, edge_index)

        return x_dst#, edge_out