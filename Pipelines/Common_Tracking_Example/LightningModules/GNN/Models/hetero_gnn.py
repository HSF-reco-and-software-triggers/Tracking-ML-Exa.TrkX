import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..hetero_gnn_base import LargeGNNBase
from ..utils import make_mlp

class HeteroGNN(LargeGNNBase):

    """
    A heterogeneous interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        concatenation_factor = (
            3 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]) else 2
        )

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Setup input network

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

        # The edge network computes new edge features from connected nodes

        # Make edge encoder combos
        self.all_combos = torch.combinations(torch.arange(len(self.hparams["model_ids"])), r=2, with_replacement=True)                
        self.edge_encoders = nn.ModuleList([
            make_mlp(
                2 * (hparams["hidden"]),
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            ) for _ in self.all_combos
        ])

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
            concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )


    def message_step(self, x, start, end, e):

        # Compute new node features
        if self.hparams["aggregation"] == "sum":
            edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0])

        elif self.hparams["aggregation"] == "mean":
            edge_messages = scatter_mean(e, end, dim=0, dim_size=x.shape[0])
            
        elif self.hparams["aggregation"] == "max":
            edge_messages = scatter_max(e, end, dim=0, dim_size=x.shape[0])[0]

        elif self.hparams["aggregation"] == "sum_max":
            edge_messages = torch.cat(
                [
                    scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                    scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                ],
                dim=-1,
            )        
        elif self.hparams["aggregation"] == "mean_sum":
            edge_messages = torch.cat(
                [
                    scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
                    scatter_add(e, end, dim=0, dim_size=x.shape[0]),
                ],
                dim=-1,
            )            
        elif self.hparams["aggregation"] == "mean_max":
            edge_messages = torch.cat(
                [
                    scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                    scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
                ],
                dim=-1,
            )
            
        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs)

        x_out += x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        e_out += e

        return x_out, e_out

    def output_step(self, x, start, end, e):

        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)

        return self.output_edge_classifier(classifier_inputs).squeeze(-1)

    def forward(self, x, edge_index, volume_id):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True

        encoded_nodes = torch.empty((x.shape[0], self.hparams["hidden"])).to(self.device)
        for encoder, model in zip(self.node_encoders, self.hparams["model_ids"]):
            node_id_mask = torch.isin(volume_id, torch.tensor(model["volume_ids"]).to(self.device))
            encoded_nodes[node_id_mask] = checkpoint(encoder, x[node_id_mask, :model["num_features"]])

        encoded_edges = torch.empty((edge_index.shape[1], self.hparams["hidden"])).to(self.device)
        for encoder, combo in zip(self.edge_encoders, self.all_combos):
            vol_ids_0, vol_ids_1 = self.hparams["model_ids"][combo[0]]["volume_ids"], self.hparams["model_ids"][combo[1]]["volume_ids"]
            edge_id_mask = (
                (torch.isin(volume_id[edge_index[0]], torch.tensor(vol_ids_0).to(self.device)) & torch.isin(volume_id[edge_index[1]], torch.tensor(vol_ids_1).to(self.device))) 
                | (torch.isin(volume_id[edge_index[0]], torch.tensor(vol_ids_1).to(self.device)) & torch.isin(volume_id[edge_index[1]], torch.tensor(vol_ids_0).to(self.device)))
            )
            encoded_edges[edge_id_mask] = checkpoint(encoder, torch.cat([encoded_nodes[edge_index[0, edge_id_mask]], encoded_nodes[edge_index[1, edge_id_mask]]], dim=-1))

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            encoded_nodes, encoded_edges = checkpoint(self.message_step, encoded_nodes, start, end, encoded_edges)

        # Compute final edge scores; use original edge directions only
        # return checkpoint(self.output_step, x, start, end, e)
        return self.output_step(encoded_nodes, start, end, encoded_edges)
