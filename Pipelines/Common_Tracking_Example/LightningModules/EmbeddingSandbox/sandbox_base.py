"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

import logging

# 3rd party imports
import numpy as np
import torch

# Local Imports
from ..Embedding.utils import build_edges
from ..Embedding.embedding_base import EmbeddingBase

device = "cuda" if torch.cuda.is_available() else "cpu"

class SandboxUndirectedEmbeddingBase(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        if "scheduler" in self.hparams:
            if self.hparams["scheduler"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer[0], T_max=self.hparams["patience"]
                )
            elif self.hparams["scheduler"] == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer[0], step_size=self.hparams["patience"], gamma=self.hparams["factor"]
                )
            elif self.hparams["scheduler"] == "none":
                return optimizer
        else:
            if self.hparams["patience"] > 0:
                scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer[0], step_size=self.hparams["patience"], gamma=self.hparams["factor"]
                    )
            else:
                return optimizer

        scheduler = [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]

        return optimizer, scheduler

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial, r_train=None, knn=None):
        if r_train is None:
            r_train = self.hparams["r_train"]
        if knn is None:
            knn = self.hparams["knn"]
        if "knn_factor" in self.hparams and "knn_patience" in self.hparams:
            # Update knn based on growth rate knn_factor, the number of steps knn_patience, and the current step self.trainer.current_epoch
            knn = int(knn * self.hparams["knn_factor"] ** (np.ceil(
                (self.trainer.current_epoch + 1 - self.hparams["knn_patience"]) / self.hparams["knn_patience"]
                )))

        knn_edges = build_edges(
                query,
                spatial,
                query_indices,
                r_train,
                knn                
            )

        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )

        return e_spatial

    def get_hinge_distance(self, spatial1, spatial2, e_spatial, y_cluster):

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial1[e_spatial[0]]
        neighbors = spatial2[e_spatial[1]]
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        return hinge, d

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        spatial = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        hinge, d = self.get_hinge_distance(spatial, spatial, e_spatial, y_cluster)

        # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + (self.hparams["weight"] * positive_loss)
        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        input_data = self.get_input_data(batch)
        spatial = self(input_data)

        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )       

        R95, R98, R99 = self.get_working_points(spatial, spatial, e_bidir)

        # Build whole KNN graph
        e_spatial_99 = build_edges(
            spatial, spatial, indices=None, r_max=R99, k_max=knn_num
        )

        e_spatial_fixed_radius = build_edges(
            spatial, spatial, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial_99, y_cluster_99 = self.get_truth(batch, e_spatial_99, e_bidir)

        _, d_99 = self.get_hinge_distance(
            spatial, spatial, e_spatial_99.to(self.device), y_cluster_99
        )

        pur_95, pur_98, pur_99 = self.get_working_metrics(e_spatial_99, y_cluster_99, d_99, R95, R98)

        e_spatial_fixed_radius, y_cluster_fixed_radius = self.get_truth(batch, e_spatial_fixed_radius, e_bidir)

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster_fixed_radius.sum()
        cluster_positive = len(e_spatial_fixed_radius[0])

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        hinge, d = self.get_hinge_distance(
            spatial, spatial, e_spatial_fixed_radius.to(self.device), y_cluster_fixed_radius
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "current_lr": current_lr, "R95": R95, "R98": R98, "R99": R99, "pur_99": pur_99, "pur_98": pur_98, "pur_95": pur_95, "eff_fixed_radius": eff, "pur_fixed_radius": pur}
            )

        return {
            "loss": loss,
            "preds": e_spatial_fixed_radius,
            "truth": y_cluster_fixed_radius,
            "truth_graph": e_bidir,
        }

    def get_working_points(self, spatial1, spatial2, truth):
        """
        Args:
            spatial (``torch.tensor``, required): The spatial embedding of the data
            truth (``torch.tensor``, required): The truth graph of the data

        Returns:
            ``torch.tensor``: The R95, R98, R99 values
        """
        margin_multiple = 1
        max_dist = torch.tensor(margin_multiple*self.hparams["margin"]).float().to(self.device)

        # Get the R95, R98, R99 values
        # distances = torch.sum((spatial[truth[0]] - spatial[truth[1]])**2, dim=-1)
        distances = torch.pairwise_distance(spatial1[truth[0]], spatial2[truth[1]])
        # Sort the distances
        distances, indices = torch.sort(distances, descending=False)
        # Get the indices of the 95th, 98th, 99th percentile
        R95 = min(distances[int(len(distances)*0.95)], max_dist)
        R98 = min(distances[int(len(distances)*0.98)], max_dist)
        R99 = min(distances[int(len(distances)*0.99)], max_dist)

        return R95.item(), R98.item(), R99.item()

    def get_working_metrics(self, e_spatial, y_cluster, d, R95, R98):
            edge_mask_98 = d < R98**2
            e_spatial_98, y_cluster_98 = e_spatial[:, edge_mask_98], y_cluster[edge_mask_98]

            edge_mask_95 = d < R95**2
            e_spatial_95, y_cluster_95 = e_spatial[:, edge_mask_95], y_cluster[edge_mask_95]

            cluster_tp_99 = y_cluster.sum()
            cluster_tp_98 = y_cluster_98.sum()
            cluster_tp_95 = y_cluster_95.sum()

            cluster_positive_99 = len(e_spatial[0])
            cluster_positive_98 = len(e_spatial_98[0])
            cluster_positive_95 = len(e_spatial_95[0])
            
            pur_99 = torch.tensor(cluster_tp_99 / cluster_positive_99)
            pur_98 = torch.tensor(cluster_tp_98 / cluster_positive_98)
            pur_95 = torch.tensor(cluster_tp_95 / cluster_positive_95)

            return pur_95, pur_98, pur_99

    

class SandboxDirectedEmbeddingBase(SandboxUndirectedEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        spatial1, spatial2 = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial1)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial2)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial2)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = batch.signal_true_edges

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        hinge, d = self.get_hinge_distance(spatial1, spatial2, e_spatial, y_cluster)

        # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        input_data = self.get_input_data(batch)
        spatial1, spatial2 = self(input_data)

        e_bidir = batch.signal_true_edges

        R95, R98, R99 = self.get_working_points(spatial1, spatial2, e_bidir)

        # Build whole KNN graph
        e_spatial_99 = build_edges(
            spatial1, spatial2, indices=None, r_max=R99, k_max=knn_num
        )

        e_spatial_fixed_radius = build_edges(
            spatial1, spatial2, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial_99, y_cluster_99 = self.get_truth(batch, e_spatial_99, e_bidir)

        _, d_99 = self.get_hinge_distance(
            spatial1, spatial2, e_spatial_99.to(self.device), y_cluster_99
        )

        pur_95, pur_98, pur_99 = self.get_working_metrics(e_spatial_99, y_cluster_99, d_99, R95, R98)

        e_spatial_fixed_radius, y_cluster_fixed_radius = self.get_truth(batch, e_spatial_fixed_radius, e_bidir)

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster_fixed_radius.sum()
        cluster_positive = len(e_spatial_fixed_radius[0])

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        hinge, d = self.get_hinge_distance(
            spatial1, spatial2, e_spatial_fixed_radius.to(self.device), y_cluster_fixed_radius
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "current_lr": current_lr, "R95": R95, "R98": R98, "R99": R99, "pur_99": pur_99, "pur_98": pur_98, "pur_95": pur_95, "eff_fixed_radius": eff, "pur_fixed_radius": pur}
            )

        return {
            "loss": loss,
            "preds": e_spatial_fixed_radius,
            "truth": y_cluster_fixed_radius,
            "truth_graph": e_bidir,
        }