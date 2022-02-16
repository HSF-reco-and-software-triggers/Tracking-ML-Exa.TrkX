# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..super_embedding_base import SuperEmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ..utils import graph_intersection


class SuperLayerlessEmbedding(SuperEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        # Construct the MLP architecture
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        self.network1 = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=False,
            layer_norm=True,
        )

        self.network2 = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=False,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x1_out = self.network1(x)
        x2_out = self.network2(x)

        if "norm" in self.hparams["regime"]:
            return F.normalize(x1_out), F.normalize(x2_out)
        else:
            return x1_out, x2_out
