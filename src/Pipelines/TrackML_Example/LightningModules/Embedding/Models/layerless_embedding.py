# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..embedding_base import EmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ..utils import graph_intersection


class LayerlessEmbedding(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        layers = [Linear(hparams["in_channels"], hparams["emb_hidden"])]
        ln = [
            Linear(hparams["emb_hidden"], hparams["emb_hidden"])
            for _ in range(hparams["nb_layer"] - 1)
        ]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(hparams["emb_hidden"], hparams["emb_dim"])
        self.norm = nn.LayerNorm(hparams["emb_hidden"])
        self.act = nn.Tanh()
        self.save_hyperparameters()

    def forward(self, x):
        #         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
        #         x = self.norm(x) #Option of LayerNorm
        x = self.emb_layer(x)
        return x
