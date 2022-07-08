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
from ..utils import graph_intersection
from ..filter_base import FilterBase, FilterBaseBalanced


class PyramidFilter(FilterBaseBalanced):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

        # Construct the MLP architecture
        self.input_layer = Linear(
            hparams["in_channels"] * 2 + hparams["emb_channels"] * 2, hparams["hidden"]
        )
        layers = [
            Linear(hparams["hidden"] // (2**i), hparams["hidden"] // (2 ** (i + 1)))
            for i in range(hparams["nb_layer"] - 1)
        ]
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(
            hparams["hidden"] // (2 ** (hparams["nb_layer"] - 1)), 1
        )
        self.act = nn.Tanh()
        batch_norms = [torch.nn.BatchNorm1d(l.out_features) for l in self.layers]
        self.batch_norms = nn.ModuleList(batch_norms)

    def forward(self, x, e, emb=None):
        if emb is not None:
            x = self.input_layer(
                torch.cat([x[e[0]], emb[e[0]], x[e[1]], emb[e[1]]], dim=-1)
            )
        else:
            x = self.input_layer(torch.cat([x[e[0]], x[e[1]]], dim=-1))
        for b, l in zip(self.batch_norms, self.layers):
            x = l(x)
            x = self.act(x)
            if self.hparams["layernorm"]:
                x = F.layer_norm(x, (l.out_features,))  # Option of LayerNorm
            if self.hparams["batchnorm"]:
                x = b(x)  # Option of BatchNorm
        x = self.output_layer(x)
        return x
