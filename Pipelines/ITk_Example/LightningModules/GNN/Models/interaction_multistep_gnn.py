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
from ..utils import make_mlp


class InteractionMultistepGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden"], int(hparams["hidden"] / 2)],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            hparams["hidden"],
            [hparams["hidden"], int(hparams["hidden"] / 2)],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"], int(hparams["hidden"] / 2)],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"], int(hparams["hidden"] / 2)],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            int(1.5 * hparams["hidden"]),
            [hparams["hidden"], int(hparams["hidden"] / 2), 1],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        input_x = x
        input_e = e

        edge_outputs = []
        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Cocnatenate with initial latent space
            x = torch.cat([x, input_x], dim=-1)
            e = torch.cat([e, input_e], dim=-1)

            # Compute new node features
            edge_messages = scatter_add(
                e, end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(e, start, dim=0, dim_size=x.shape[0])
            node_inputs = torch.cat([x, edge_messages], dim=-1)
            x = self.node_network(node_inputs)

            # Compute new edge features
            edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
            e = self.edge_network(edge_inputs)
            e = torch.sigmoid(e)

            classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
            edge_outputs.append(
                self.output_edge_classifier(classifier_inputs).squeeze(-1)
            )

        # Compute final edge scores; use original edge directions only
        return edge_outputs

    def training_step(self, batch, batch_idx):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        output = (
            self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index)
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index)
        )

        output = torch.cat(output)

        if "pid" in self.hparams["regime"]:
            y = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        else:
            y = batch.y

        loss = F.binary_cross_entropy_with_logits(
            output,
            y.float().repeat(self.hparams["n_graph_iters"]),
            weight=torch.repeat_interleave(
                (
                    (torch.arange(self.hparams["n_graph_iters"]) + 1)
                    / self.hparams["n_graph_iters"]
                ).to(self.device),
                len(y),
            ),
            pos_weight=weight,
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        if "pid" in self.hparams["regime"]:
            y = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        else:
            y = batch.y

        output = (
            self(
                torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index
            ).squeeze()
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index)
        )

        output = output[-1]

        truth = (
            (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            if "pid" in self.hparams["regime"]
            else batch.y
        )

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth.float(), weight=manual_weights, pos_weight=weight
        )

        # Edge filter performance
        preds = F.sigmoid(output) > self.hparams["edge_cut"]
        edge_positive = preds.sum().float()

        edge_true = truth.sum().float()
        edge_true_positive = (truth.bool() & preds).sum().float()

        eff = torch.tensor(edge_true_positive / edge_true)
        pur = torch.tensor(edge_true_positive / edge_positive)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
        )

        return {
            "loss": loss,
            "preds": preds.cpu().numpy(),
            "truth": truth.cpu().numpy(),
        }


class CheckpointedInteractionMultistepGNN(InteractionMultistepGNN):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        input_x = x
        input_e = e

        edge_outputs = []
        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Cocnatenate with initial latent space
            x = torch.cat([x, input_x], dim=-1)
            e = torch.cat([e, input_e], dim=-1)

            # Compute new node features
            edge_messages = scatter_add(
                e, end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(e, start, dim=0, dim_size=x.shape[0])
            node_inputs = torch.cat([x, edge_messages], dim=-1)
            x = checkpoint(self.node_network, node_inputs)

            # Compute new edge features
            edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
            e = checkpoint(self.edge_network, edge_inputs)
            e = torch.sigmoid(e)

            classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
            edge_outputs.append(
                checkpoint(self.output_edge_classifier, classifier_inputs).squeeze(-1)
            )

        # Compute final edge scores; use original edge directions only
        return edge_outputs
