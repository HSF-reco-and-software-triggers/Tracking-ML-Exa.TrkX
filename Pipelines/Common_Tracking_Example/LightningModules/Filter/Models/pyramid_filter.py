# System imports
import sys
import os
import copy

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ..utils import graph_intersection, make_mlp
from ..filter_base import FilterBase, FilterBaseBalanced, LargeFilterBaseBalanced


class PyramidFilter(LargeFilterBaseBalanced):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

        # Construct the MLP architecture
        self.net = make_mlp(
            (hparams["spatial_channels"] + hparams["cell_channels"] + hparams["emb_channels"]) * 2,
            [hparams["hidden"] // (2**i) for i in range(hparams["nb_layer"])] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
            
        
    def forward(self, x, e, emb=None):
        if emb is not None:
            x = self.net(
                torch.cat([x[e[0]], emb[e[0]], x[e[1]], emb[e[1]]], dim=-1)
            )
        else:
            x = self.net(torch.cat([x[e[0]], x[e[1]]], dim=-1))
        return x
