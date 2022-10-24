"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

import logging

# 3rd party imports
import torch

# Local Imports
from ..Embedding.utils import build_edges
from ..Embedding.embedding_base import EmbeddingBase

device = "cuda" if torch.cuda.is_available() else "cpu"

class SuperEmbeddingBase(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial, r_train=None, knn=None):
        if r_train is None:
            r_train = self.hparams["r_train"]
        if knn is None:
            knn = self.hparams["knn"]

        knn_edges = build_edges(
                query,
                spatial,
                query_indices,
                r_train,
                knn,
                remove_self_loops=False
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

        spatial1, spatial2 = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial1)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial2)
            # print(e_spatial.shape[1] / len(query))

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial2)

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

        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Build whole KNN graph
        e_spatial = build_edges(
            spatial1, spatial2, indices=None, r_max=knn_radius, k_max=knn_num, remove_self_loops=False
        )

        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        hinge, d = self.get_hinge_distance(
            spatial1, spatial2, e_spatial.to(self.device), y_cluster
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))

        return {
            "loss": loss,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
        }