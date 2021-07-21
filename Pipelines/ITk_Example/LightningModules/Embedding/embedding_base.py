"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

# System imports
import sys
import os
import logging

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_cluster import radius_graph
import numpy as np

# Local Imports
from .utils import graph_intersection, split_datasets, build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == "fit":
            self.trainset, self.valset, self.testset = split_datasets(
                self.hparams["input_dir"],
                self.hparams["train_split"],
                self.hparams["pt_min"],
                self.hparams["n_hits"],
                self.hparams["primary_only"]
            )

    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

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
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """
        
        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch.modulewise_true_edges, batch.modulewise_true_edges.flip(0)], axis=-1
        )

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)
        
        
        # Forward pass of model, handling whether Cell Information (ci) is included
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], batch.x], axis=-1)
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0
        
        with torch.no_grad():
            spatial = self(input_data)
            
        cut_indices = batch.modulewise_true_edges.unique()
        query = spatial[cut_indices]
#         print("Query", query)
#         print("Database", spatial)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            knn_edges = build_edges(query, spatial, cut_indices, self.hparams["r_train"], self.hparams["knn"])
#             print("KNN:", knn_edges.shape)
#             print("Unique in KNN:", knn_edges.unique().shape)
            e_spatial = torch.cat(
                [
                    e_spatial,
                    knn_edges,
                ],
                axis=-1,
            )
        
        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            
            n_random = int(self.hparams["randomisation"] * len(cut_indices))
            indices_src = torch.randint(0, len(cut_indices), (n_random,), device=self.device)
            indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
            random_pairs = torch.stack([cut_indices[indices_src], indices_dest])
            
            
            e_spatial = torch.cat(
                [
                    e_spatial,
                    random_pairs
                ],
                axis=-1,
            )

        

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            axis=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        new_weights = torch.cat(
            [
                new_weights,
                torch.ones(e_bidir.shape[1], device=self.device)
                * self.hparams["weight"],
            ]
        )
        
        included_hits = e_spatial.unique()
        print("Total shape:", spatial.shape[0], " - Unique shape:", included_hits.shape)
        
        spatial[included_hits] = self(input_data[included_hits])

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        new_weights[
            y_cluster == 0
        ] = 1  # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        d = d * new_weights

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], batch.x], axis=-1)
            input_data[input_data != input_data] = 0
            spatial = self(input_data)
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0
            spatial = self(input_data)

        e_bidir = torch.cat(
            [batch.modulewise_true_edges, batch.modulewise_true_edges.flip(0)], axis=-1
        )

        # Build whole KNN graph
        e_spatial = build_edges(spatial, spatial, indices=None, r_max=knn_radius, k_max=knn_num)
        
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        e_spatial = e_spatial.to(self.device)
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        new_weights[y_cluster == -1] = 1
        d = d # * new_weights THIS IS BETTER TO NOT INCLUDE

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        current_lr = self.optimizers().param_groups[0]["lr"]
        if log:
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)

        return {
            "loss": loss,
            "preds": e_spatial.cpu().numpy(),
            "truth": y_cluster.cpu().numpy(),
            "truth_graph": e_bidir.cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 100, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 500, log=True
        )

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """

        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
