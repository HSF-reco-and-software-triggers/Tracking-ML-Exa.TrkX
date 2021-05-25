import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from ..utils import make_mlp, hard_random_edge_slice, hard_eta_edge_slice


class CheckpointedResAGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden"]],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            2 * (hparams["in_channels"] + hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            (hparams["in_channels"] + hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index):

        # Encode the graph features into the hidden space
        input_x = x
        x = self.node_encoder(x)
        x = torch.cat([x, input_x], dim=-1)

        start, end = edge_index

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            # Previous hidden state
            x0 = x

            # Compute new edge score
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = checkpoint(self.edge_network, edge_inputs)
            e = torch.sigmoid(e)

            # Sum weighted node features coming into each node
            #             weighted_messages_in = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])
            #             weighted_messages_out = scatter_add(e * x[end], start, dim=0, dim_size=x.shape[0])

            weighted_messages = scatter_add(
                e * x[start], end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(e * x[end], start, dim=0, dim_size=x.shape[0])

            # Compute new node features
            #             node_inputs = torch.cat([x, weighted_messages_in, weighted_messages_out], dim=1)
            node_inputs = torch.cat([x, weighted_messages], dim=1)
            x = checkpoint(self.node_network, node_inputs)

            # Residual connection
            x = torch.cat([x, input_x], dim=-1)
            x = x + x0

        # Compute final edge scores; use original edge directions only
        clf_inputs = torch.cat([x[start], x[end]], dim=1)
        return checkpoint(self.edge_network, clf_inputs).squeeze(-1)


class SliceCheckpointedResAGNN(CheckpointedResAGNN):
    def __init__(self, hparams):
        super().__init__(hparams)

    def training_step(self, batch, batch_idx):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        if "delta_phi" in self.hparams.keys():
            subset_edge_ind = hard_random_edge_slice(self.hparams["delta_phi"], batch)
        elif "delta_eta" in self.hparams.keys():
            subset_edge_ind = hard_eta_edge_slice(self.hparams["delta_eta"], batch)

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights[subset_edge_ind]
            manual_weights[batch.y[subset_edge_ind] == 0] = 1
        else:
            manual_weights = None

        output = (
            self(
                torch.cat([batch.cell_data, batch.x], axis=-1),
                batch.edge_index[:, subset_edge_ind],
            ).squeeze()
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index[:, subset_edge_ind]).squeeze()
        )

        if "pid" in self.hparams["regime"]:
            y_pid = (
                batch.pid[batch.edge_index[0, subset_edge_ind]]
                == batch.pid[batch.edge_index[1, subset_edge_ind]]
            ).float()
            loss = F.binary_cross_entropy_with_logits(
                output, y_pid.float(), weight=manual_weights, pos_weight=weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                output,
                batch.y[subset_edge_ind].float(),
                weight=manual_weights,
                pos_weight=weight,
            )

        self.log("train_loss", loss)

        return loss
