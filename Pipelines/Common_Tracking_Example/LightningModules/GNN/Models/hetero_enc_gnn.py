import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..hetero_gnn_base import LargeGNNBase
from ..utils import make_mlp

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
                hparams["model_ids"][combo[0]]["num_features"] + hparams["model_ids"][combo[1]]["num_features"] + 2*hparams["cell_channels"],
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            ) for combo in self.all_combos
        ])

    def forward(self, x, edge_index, volume_id):
        """
        Forward pass of the heterogeneous encoder

        x is the input node features. It is expected to be of shape (num_nodes, max number of input features), where rows that have fewer features than the max are padded with zeros.
        It creates an empty tensor of the same shape as x, and fills it with the encoded nodes, for each node encoder. The same is done for encoded edges.
        """

        start, end = edge_index

        encoded_nodes = torch.empty((x.shape[0], self.hparams["hidden"])).to(x.device)
        encoded_nodes = self.fill_hetero_nodes(encoded_nodes, x, volume_id)

        encoded_edges = torch.empty((edge_index.shape[1], self.hparams["hidden"])).to(edge_index.device)
        encoded_edges = self.fill_hetero_edges(encoded_edges, encoded_nodes, start, end, volume_id)        

        return encoded_nodes, encoded_edges

    def fill_hetero_nodes(self, encoded_nodes, x, volume_id):
        """
        Fill the heterogeneous nodes with the corresponding encoders
        """
        for encoder, model in zip(self.node_encoders, self.hparams["model_ids"]):
            node_id_mask = torch.isin(volume_id, torch.tensor(model["volume_ids"]).to(x.device))
            encoded_nodes[node_id_mask] = checkpoint(encoder, x[node_id_mask, :model["num_features"]])
        
        return encoded_nodes

    def fill_hetero_edges(self, encoded_edges, x, cell_x, start, end, volume_id):
        """
        Fill the heterogeneous edges with the corresponding encoders
        """
        for encoder, combo in zip(self.edge_encoders, self.all_combos):
            vol_ids_0, vol_ids_1 = torch.tensor(self.hparams["model_ids"][combo[0]]["volume_ids"], device=encoded_edges.device), torch.tensor(self.hparams["model_ids"][combo[1]]["volume_ids"], device=encoded_edges.device)                        
            vol_edge_mask = torch.isin(volume_id[start], vol_ids_0) & torch.isin(volume_id[end], vol_ids_1)
            
            encoded_edges[vol_edge_mask] = encoder(
                torch.cat([
                    x[start[vol_edge_mask], :self.hparams["model_ids"][combo[0]]["num_features"]],
                    cell_x[start[vol_edge_mask]],
                    x[end[vol_edge_mask], :self.hparams["model_ids"][combo[1]]["num_features"]],
                    cell_x[end[vol_edge_mask]]                    
                ], dim=-1)
            )

        return encoded_edges

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

    def get_aggregation(self):
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

        return aggregation_dict[self.hparams["aggregation"]]


    def forward(self, x, edge_index, e):

        start, end = edge_index

        # Perform message passing
        edge_messages = self.get_aggregation()(e, end, x)
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        x_out = self.node_network(node_inputs)
        x_out += x

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)
        e_out += e

        return x_out, e_out


class HeteroEncGNN(LargeGNNBase):

    """
    A heterogeneous interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Make input network
        self.encoder = HeteroEncoder(hparams)

        # Make message passing network
        self.conv = HomoConv(hparams)        

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def output_step(self, x, edge_index, e):
        
        classifier_inputs = torch.cat([x[edge_index[0]], x[edge_index[1]], e], dim=1)

        return self.output_edge_classifier(classifier_inputs).squeeze(-1)

    def forward(self, x, edge_index, volume_id):

        # Encode the graph features into the hidden space
        x.requires_grad = True
        encoded_nodes, encoded_edges = self.encoder(x, edge_index, volume_id)
        
        # Loop over iterations of edge and node networks
        for _ in range(self.hparams["n_graph_iters"]):

            encoded_nodes, encoded_edges = checkpoint(self.conv, encoded_nodes, edge_index, encoded_edges)

        # Compute final edge scores
        
        # TODO: Apply output to SUM of directional edge features (across both directions!)
        return self.output_step(encoded_nodes, edge_index, encoded_edges)