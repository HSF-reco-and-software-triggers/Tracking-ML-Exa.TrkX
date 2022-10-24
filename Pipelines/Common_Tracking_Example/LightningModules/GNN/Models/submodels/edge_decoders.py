import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ...utils import make_mlp

class HomoDecoder(torch.nn.Module):
    
        """
        A simple decoder that handles only homogeneous models
        """
            
        def __init__(self, hparams):
            super(HomoDecoder, self).__init__()

            self.hparams = hparams
            
            if "concat_output" in self.hparams and self.hparams["concat_output"]:
                self.input_channels = 3 * hparams["hidden"] * hparams["n_graph_iters"]
            else:
                self.input_channels = 3 * hparams["hidden"]
            
            self.output_edge_classifier = make_mlp(
                self.input_channels,
                [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=None,
                hidden_activation=hparams["hidden_activation"],
            )

        def forward(self, x, edge_index, e, volume_id=None):
            
            classifier_inputs = torch.cat([x[edge_index[0]], x[edge_index[1]], e], dim=1)
            
            return self.output_edge_classifier(classifier_inputs).squeeze(-1)

class HeteroDecoder(torch.nn.Module):
    """
    A hetero edge classifier
    """
    
    def __init__(self, hparams):
        super(HeteroDecoder, self).__init__()

        self.hparams = hparams
    
        self.all_combos = torch.combinations(torch.arange(len(self.hparams["model_ids"])), r=2, with_replacement=True)
        
        self.edge_decoders = nn.ModuleList([
            make_mlp(
                3 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=None,
                hidden_activation=hparams["hidden_activation"],
            )
            for _ in self.all_combos
        ])

    def forward(self, x, edge_index, e, volume_id=None):
        """
        Run output edge classifiers on the edge features
        """

        start, end = edge_index

        return self.fill_hetero_edges(e, x, start, end, volume_id)

    def fill_hetero_edges(self, input_edge_features, input_node_features, start, end, volume_id):
        """
        Fill the heterogeneous edges with the corresponding encoders
        """
        features_to_fill = torch.empty((start.shape[0], 1)).to(start.device)

        for decoder, combo in zip(self.edge_decoders, self.all_combos):
            vol_ids_0, vol_ids_1 = torch.tensor(self.hparams["model_ids"][combo[0]]["volume_ids"], device=features_to_fill.device), torch.tensor(self.hparams["model_ids"][combo[1]]["volume_ids"], device=features_to_fill.device)                        
            vol_edge_mask = torch.isin(volume_id[start], vol_ids_0) & torch.isin(volume_id[end], vol_ids_1)
            
            features_to_decode = torch.cat([
                    input_node_features[start[vol_edge_mask]],
                    input_node_features[end[vol_edge_mask]],
                    input_edge_features[vol_edge_mask]         
                ], dim=-1)

            features_to_fill[vol_edge_mask] = decoder(features_to_decode)

        return features_to_fill