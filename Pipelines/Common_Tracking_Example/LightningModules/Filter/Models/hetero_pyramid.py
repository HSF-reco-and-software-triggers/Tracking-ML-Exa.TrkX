# System imports

# 3rd party imports
import torch
import torch.nn as nn

# Local imports
from ..utils import make_mlp
from ..hetero_filter_base import HeteroFilterBase


class HeteroPyramidFilter(HeteroFilterBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

        self.all_combos = torch.combinations(torch.arange(len(self.hparams["model_ids"])), r=2, with_replacement=True)                
        self.edge_encoders = nn.ModuleList([
            make_mlp(
                hparams["model_ids"][combo[0]]["num_features"] + hparams["model_ids"][combo[1]]["num_features"] + 2*hparams["cell_channels"],
                [hparams["hidden"] // (2**i) for i in range(hparams["nb_layer"])] + [1],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=None,
                hidden_activation=hparams["hidden_activation"],
            ) for combo in self.all_combos
        ])
            
        
    def forward(self, x, cell_x, edge_index, volume_id):
        
        start, end = edge_index

        encoded_edges = torch.empty((edge_index.shape[1], 1)).to(edge_index.device)
        encoded_edges = self.fill_hetero_edges(encoded_edges, x, cell_x, start, end, volume_id)   

        return encoded_edges

    def fill_hetero_edges(self, encoded_edges, x, cell_x, start, end, volume_id):
        """
        Fill the heterogeneous edges with the corresponding encoders
        """

        for encoder, combo in zip(self.edge_encoders, self.all_combos):
            vol_ids_0, vol_ids_1 = torch.tensor(self.hparams["model_ids"][combo[0]]["volume_ids"], device=encoded_edges.device), torch.tensor(self.hparams["model_ids"][combo[1]]["volume_ids"], device=encoded_edges.device)
            print(vol_ids_0)
            print(vol_ids_1)
            print(volume_id[start])
            print(volume_id[end])
            edge_id_mask = (
                (torch.isin(volume_id[start], vol_ids_0) & torch.isin(volume_id[end], vol_ids_1)) 
                | (torch.isin(volume_id[start], vol_ids_1) & torch.isin(volume_id[end], vol_ids_0))
            )
            print(edge_id_mask)
            encoded_edges[edge_id_mask] = encoder(torch.cat([
                x[start[edge_id_mask], :self.hparams["model_ids"][combo[0]]["num_features"]], 
                cell_x[start[edge_id_mask]], 
                x[end[edge_id_mask], :self.hparams["model_ids"][combo[1]]["num_features"]],
                cell_x[end[edge_id_mask]]
            ], dim=-1))
        
        print(encoded_edges)

        return encoded_edges
