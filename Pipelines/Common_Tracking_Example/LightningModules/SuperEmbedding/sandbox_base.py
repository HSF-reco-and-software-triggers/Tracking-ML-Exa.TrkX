"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

import warnings

# 3rd party imports
import numpy as np
import torch
from torch import Tensor

# Local Imports
from .utils import build_edges
from ..Embedding.embedding_base import EmbeddingBase
from ..Embedding.utils import split_datasets
from .utils import CustomReduceLROnPlateau

device = "cuda" if torch.cuda.is_available() else "cpu"

class SandboxEmbeddingBase(EmbeddingBase):
    """
    Base class for all Lightning Embedding Models
    """

    def __init__(self, hparams):
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        super().__init__(hparams)
        self.save_hyperparameters(hparams)

    def setup(self, stage="fit"):

        if not self.trainset or not self.valset or not self.testset:
            self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

        try:
            print("Defining figures of merit")
            for working_point in self.hparams["working_points"]:
                self.logger.experiment.define_metric(f"output/pur_{working_point}" , summary="max")
            self.logger.experiment.define_metric("output/eff", summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")

    def training_step(self, batch, batch_idx):

        """
        Training step of the embedding, for each set of output spaces.
        Output spaces are assumed to be of one of the following formats:
        - tensor = self(input)
        - (tensor, tensor) = self(input)
        - [tensor, ..., tensor] = self(input)
        - [(tensor, tensor), tensor, ..., (tensor, tensor)] = self(input)
        """

        embedding_objects = self._get_embedding_objects(batch)
        embedding_objects = self.build_training_graphs(embedding_objects, batch)
        loss = self.get_training_loss(embedding_objects)

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_num, log=False):

        """
        Validation step of the embedding, for each set of output spaces.
        Output spaces are assumed to be of one of the following formats:
        - tensor = self(input)
        - (tensor, tensor) = self(input)
        - [tensor, ..., tensor] = self(input)
        - [(tensor, tensor), tensor, ..., (tensor, tensor)] = self(input)
        """

        embedding_objects = self._get_embedding_objects(batch)
        embedding_objects = self.build_validation_graphs(embedding_objects, batch, knn_num)

        metric_dict = self.evaluate_loss_metrics(embedding_objects, batch, log)

        return {
                "loss": metric_dict["val_loss"],
            }

    def _get_embedding_objects(self, batch):
        embeddings = self(batch)
        return self.process_embeddings(embeddings, batch)

    def process_embeddings(self, embeddings, batch):
        """
        Build list of embedding objects. For each embedding:
        1. Check if it's a tensor or a tuple of tensors 
        2. Set to (tensor, tensor)
        3. Set the truth, based on directed or undirected
        4. Set the knn, r, weight, based on whether the last entry or not (i.e. depending on whether a latent, or the output)
        """

        embedding_objects = []
        if isinstance(embeddings, (tuple, torch.Tensor)):
            embeddings = [embeddings]
        elif not isinstance(embeddings, list):
            raise ValueError("Embeddings must be a tensor, tuple of tensors, or list of tensors")

        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, tuple):
                directed = True
                embedding_tuple = embedding 
                truth = batch.signal_true_edges
            elif isinstance(embedding, torch.Tensor):
                directed = False
                embedding_tuple = (embedding, embedding)
                truth = torch.cat(
                    [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
                )
            else:
                raise ValueError(
                    "Embedding must be a tensor or a tuple of tensors"
                )

            if i == len(embeddings) - 1:
                knn = self.hparams["output_k"]
                r = self.hparams["output_r"]
                weight = self.hparams["output_weight"]
                margin = self.hparams["output_margin"]
            else:
                knn = self.hparams["latent_k"]
                r = self.hparams["latent_r"]
                weight = self.hparams["latent_weight"]
                margin = self.hparams["latent_margin"]

            embedding_objects.append(
                {
                    "embedding": embedding_tuple,
                    "truth": truth,
                    "knn": knn,
                    "r": r,
                    "weight": weight,
                    "directed": directed,
                    "margin": margin,
                }
            )


        return embedding_objects

    def get_hinge_distance(self, spatial1, spatial2, e_spatial, y_cluster):

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial1[e_spatial[0]]
        neighbors = spatial2[e_spatial[1]]
        d_sq = torch.sum((reference - neighbors) ** 2, dim=-1)

        return hinge, d_sq

    def build_training_graphs(self, embedding_objects, batch):
        """
        Build the set of edges used for training.
        """

        for embedding_object in embedding_objects:
            # Instantiate empty prediction edge list
            edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)
            query_indices, query = self.get_query_points(batch, embedding_object["embedding"][0])
            # print(query_indices.shape)

            # Append Hard Negative Mining (hnm) with KNN graph
            if "hnm" in self.hparams["regime"]:
                edges = self.append_hnm_pairs(edges, query, query_indices, embedding_object["embedding"][1],
                embedding_object["r"], embedding_object["knn"])

            # Append random edges pairs (rp) for stability
            if "rp" in self.hparams["regime"]:
                edges = self.append_random_pairs(edges, query_indices, embedding_object["embedding"][1])
                
            # Calculate truth from intersection between Prediction graph and Truth graph
            edges, y = self.get_truth(batch, edges, embedding_object["truth"])

            # Append all positive examples and their truth and weighting
            edges, y, _ = self.get_true_pairs(
                edges, y, embedding_object["truth"]
            )

            embedding_object["edges"] = edges
            embedding_object["y"] = y

        return embedding_objects

    def get_training_loss(self, embedding_objects):

        total_loss = torch.tensor(0.0, device=self.device)

        for embedding_object in embedding_objects:
            hinge, d_sq = self.get_hinge_distance(embedding_object["embedding"][0], 
                embedding_object["embedding"][1], 
                embedding_object["edges"].to(self.device),
                embedding_object["y"].to(self.device)
            )

            # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
            negative_loss = torch.nn.functional.hinge_embedding_loss(
                d_sq[hinge == -1],
                hinge[hinge == -1],
                margin=embedding_object["margin"]**2,
                reduction="mean",
            )

            positive_loss = torch.nn.functional.hinge_embedding_loss(
                d_sq[hinge == 1],
                hinge[hinge == 1],
                margin=embedding_object["margin"]**2,
                reduction="mean",
            )

            loss = negative_loss + embedding_object["weight"] * positive_loss

            total_loss += loss

        return total_loss

    def build_validation_graphs(self, embedding_objects, batch, knn_num):
        """
        Build the set of edges used for validation.
        """

        for embedding_object in embedding_objects:
            # Build whole KNN graph
            edges = build_edges(
                embedding_object["embedding"][0], embedding_object["embedding"][1], indices=None, r_max=self.hparams["output_r"], k_max=knn_num, backend="FRNN"
            )

            edges, y = self.get_truth(batch, edges, embedding_object["truth"])

            embedding_object["edges"] = edges
            embedding_object["y"] = y

            if "working_points" in self.hparams and self.hparams["working_points"]:

                # Get the radii required to build a graph of the desired working point efficiencies
                working_point_radii = self.get_working_points(embedding_object["embedding"][0], 
                                                        embedding_object["embedding"][1], 
                                                        embedding_object["truth"],
                                                        embedding_object["margin"])
                
                # Assuming the working point radii are sorted in decreasing efficiency, get the graph of the highest efficiency
                working_point_edges = build_edges(
                    embedding_object["embedding"][0], embedding_object["embedding"][1], indices=None, r_max=working_point_radii[0], k_max=knn_num, backend="FRNN"
                )

                # Get truth for the working point graph
                working_point_edges, working_point_y = self.get_truth(batch, working_point_edges, embedding_object["truth"])

                embedding_object["working_point_edges"] = working_point_edges
                embedding_object["working_point_y"] = working_point_y
                embedding_object["working_point_radii"] = working_point_radii

        return embedding_objects

    def get_working_points(self, spatial1, spatial2, truth, margin):
        """
        Args:
            spatial (``torch.tensor``, required): The spatial embedding of the data
            truth (``torch.tensor``, required): The truth graph of the data

        Returns:
            ``torch.tensor``: The working point values (e.g. [R95, R98, R99])
        """
        margin_multiple = 1 if "margin_multiple" not in self.hparams else self.hparams["margin_multiple"]
        max_dist = torch.tensor(margin_multiple*margin).float().to(self.device)

        distances = torch.pairwise_distance(spatial1[truth[0]], spatial2[truth[1]])
        # Sort the distances
        distances, _ = torch.sort(distances, descending=False)

        return [min(distances[int(len(distances) * (working_point_efficiency / 100))], max_dist).item() for working_point_efficiency in self.hparams["working_points"]]

    def evaluate_loss_metrics(self, embedding_objects, batch, log=True):
        """
        Evaluate the loss of the final embedding object (this is our "target loss"), 
        and log the metrics of each embedding object - including their working points if requested
        """

        log_dict = {}

        for i, embedding_object in enumerate(embedding_objects):

            if i == len(embedding_objects) - 1:
                # Get the target loss
                hinge, d_sq = self.get_hinge_distance(
                    embedding_object["embedding"][0],
                    embedding_object["embedding"][1],
                    embedding_object["edges"].to(self.device),
                    embedding_object["y"].to(self.device),
                )

                loss = torch.nn.functional.hinge_embedding_loss(
                    d_sq, hinge, margin=embedding_object["margin"]**2, reduction="mean"
                )

                log_dict["val_loss"] = loss.item()

                log_prefix = "output/"

            else:
                log_prefix = f"latent_{i}/"

            # Get the values required for the metric calculation 
            eff, pur = self.get_metric_values(embedding_object)
            log_dict[f"{log_prefix}eff"] = eff.item()
            log_dict[f"{log_prefix}pur"] = pur.item()

            # If working points requested, get those values as well
            if "working_points" in self.hparams and self.hparams["working_points"]:
                working_logs = self.get_working_point_metrics(embedding_object, batch, log_prefix)
                log_dict.update(working_logs)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            log_dict["current_lr"] = current_lr
            self.log_dict(log_dict, batch_size=1)

        return log_dict

    def get_working_point_metrics(self, embedding_object, batch, log_prefix):
        _, d_sq = self.get_hinge_distance(
            embedding_object["embedding"][0],
            embedding_object["embedding"][1],
            embedding_object["working_point_edges"].to(self.device),
            embedding_object["working_point_y"].to(self.device),
        )
        
        working_logs = {}

        for i, r in enumerate(embedding_object["working_point_radii"]):
            edge_mask = (d_sq < r**2).to(self.device)
            masked_y = embedding_object["working_point_y"][edge_mask]
            masked_edges = embedding_object["working_point_edges"][:, edge_mask]
            
            num_true_positive = masked_y.sum()
            num_pid_positive = ((batch.pid[masked_edges[0]] == batch.pid[masked_edges[1]]) & (batch.pid[masked_edges[0]] != 0)).sum()
            num_positive = masked_y.shape[0]

            purity = num_true_positive / num_positive
            pid_purity = num_pid_positive / num_positive

            working_logs[f"{log_prefix}pur_{self.hparams['working_points'][i]}"] = purity.item()
            working_logs[f"{log_prefix}PID_pur_{self.hparams['working_points'][i]}"] = pid_purity.item()
            working_logs[f"{log_prefix}R_{self.hparams['working_points'][i]}"] = r

        return working_logs

    def get_metric_values(self, embedding_object, working_point=False):
        """
        Get the values required for the metric calculation 
        """

        num_true = embedding_object["truth"].shape[1]
        num_true_positive = embedding_object["y"].sum()
        num_positive = embedding_object["edges"].shape[1]

        eff = num_true_positive / num_true
        pur = num_true_positive / num_positive

        return eff, pur
                
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
            elif self.hparams["scheduler"] == "plateau":
                scheduler = CustomReduceLROnPlateau(
                    optimizer[0], mode="min", factor=self.hparams["factor"], patience=self.hparams["patience"], verbose=True,
                    ignore_first_n_epochs=2*self.hparams["warmup"]
                )
            elif self.hparams["scheduler"] == "none":
                return optimizer
        elif self.hparams["patience"] > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer[0], step_size=self.hparams["patience"], gamma=self.hparams["factor"]
                )
        else:
            return optimizer

        scheduler = [{"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}]

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
                knn,
                backend="FRNN"
            )

        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )

        return e_spatial

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        knn_val = 1000 if self.trainer.current_epoch > self.hparams["warmup"] else 100
        outputs = self.shared_evaluation(
            batch, batch_idx, knn_val, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(batch, batch_idx, 1000, log=False)